"""Run a repeatable, streaming benchmark: python -m benchmarks.run --help."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from text_redaction.cli import build_parser as build_redaction_parser
from text_redaction.cli import validate_args
from text_redaction.io import batches, run_file
from text_redaction.pipeline import Pipeline

from .monitor import ProcessResourceMonitor as ResourceMonitor


class BenchmarkValidationError(ValueError):
    """An allowlisted, input-independent validation explanation safe to display."""


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _headroom(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number < 1:
        raise argparse.ArgumentTypeError("must be between 0 (inclusive) and 1 (exclusive)")
    return number


def build_parser():
    parser = build_redaction_parser()
    parser.description = "Benchmark redaction throughput and process-tree CPU, RAM and GPU resources."
    parser.add_argument("--warmup", type=_nonnegative_int, default=1, help="Warm-up batches excluded from throughput (default: 1).")
    parser.add_argument("--repeats", type=_positive_int, default=3, help="Full input passes (default: 3).")
    parser.add_argument("--sample-interval", type=_positive_float, default=0.5, help="Resource sample interval in seconds (default: 0.5).")
    parser.add_argument("--report-dir", type=Path, help="New, non-existing directory for report.json and samples.csv.")
    parser.add_argument("--write-output", action="store_true", help="Write redacted output on every pass; final pass replaces prior output.")
    parser.add_argument("--target-records", type=_positive_int, help="Production record count for rough capacity estimates.")
    parser.add_argument("--target-hours", type=_positive_float, help="Completion deadline in hours; requires --target-records.")
    parser.add_argument("--headroom", type=_headroom, default=0.3, help="Fraction of capacity to reserve (default: 0.3).")
    return parser


def _same_path(left, right):
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return False


def _validate_paths(args):
    source = Path(args.input).resolve()
    if not source.is_file():
        raise BenchmarkValidationError("input_must_be_an_existing_regular_file")
    if args.target_hours is not None and args.target_records is None:
        raise BenchmarkValidationError("target_hours_requires_target_records")
    if args.output is not None and not args.write_output:
        raise BenchmarkValidationError("output_requires_write_output")
    report_dir = args.report_dir
    if report_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_dir = Path("reports") / (stamp + "-" + uuid.uuid4().hex[:8])
    report_dir = Path(report_dir).resolve()
    if report_dir.exists():
        raise BenchmarkValidationError("report_directory_already_exists_choose_a_new_directory")
    output = Path(args.output).resolve() if args.output is not None else report_dir / "redacted.txt"
    report_paths = [report_dir / "report.json", report_dir / "samples.csv"]
    protected = [source]
    if getattr(args, "config", None) is not None:
        protected.append(Path(args.config).resolve())
    for path in report_paths + ([output] if args.write_output else []):
        if any(_same_path(path, item) for item in protected):
            raise BenchmarkValidationError("output_or_report_conflicts_with_input_or_config")
    if args.write_output:
        if any(_same_path(output, path) or path in output.parents or output in path.parents for path in report_paths):
            raise BenchmarkValidationError("output_conflicts_with_report_files")
        if output.exists():
            raise BenchmarkValidationError("output_already_exists_choose_a_new_output")
    report_dir.mkdir(parents=True, exist_ok=False)
    return source, report_dir, output if args.write_output else None


def _environment():
    import psutil

    versions = {}
    for package in ("paddlenlp", "paddlepaddle", "paddlepaddle-gpu", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
        "hardware": {
            "processor": platform.processor() or None,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "host_memory_total_bytes": psutil.virtual_memory().total,
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
    }


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _summarize(rounds):
    names = (
        "records", "chars", "batches", "wall_seconds", "processing_seconds",
        "regex_seconds", "model_seconds", "replace_seconds",
    )
    result = {key: sum(item.get(key, 0) for item in rounds) for key in names}
    seconds = result["wall_seconds"]
    result["records_per_second"] = result["records"] / seconds if seconds > 0 else None
    result["chars_per_second"] = result["chars"] / seconds if seconds > 0 else None
    result["completed_rounds"] = len(rounds)
    result["throughput_scope"] = "Input reading, redaction and optional output writing; initialization and warm-up excluded."
    result["record_count_scope"] = "Counts are summed over all measured passes; input records are reported separately."
    result["batch_latency_scope"] = "Per-round latency statistics come from the pipeline's bounded reservoir; inspect each round."
    return result


def _capacity(args, summary, resources):
    if args.target_records is None:
        return None
    measured = summary.get("records_per_second")
    if not measured or measured <= 0:
        return {"status": "unavailable", "reason": "no_measured_throughput"}
    usable = measured * (1 - args.headroom)
    seconds = args.target_records / usable
    workers = max(1, math.ceil(seconds / (args.target_hours * 3600))) if args.target_hours else 1
    metrics = resources.get("overall", {}).get("cpu_memory", {})
    rss = metrics.get("process_tree_rss_bytes", {}).get("max")
    cpu = metrics.get("process_tree_cpu_percent", {}).get("max")
    return {
        "status": "rough_estimate",
        "target_records": args.target_records,
        "target_hours": args.target_hours,
        "reserved_capacity_fraction": args.headroom,
        "observed_records_per_second": measured,
        "planning_records_per_second_per_worker": usable,
        "estimated_hours_one_worker": seconds / 3600,
        "estimated_concurrent_workers": workers,
        "observed_peak_process_tree_rss_bytes": rss,
        "planning_total_rss_bytes_with_headroom": math.ceil(rss * workers / (1 - args.headroom)) if rss is not None else None,
        "observed_peak_cpu_cores": cpu / 100 if cpu is not None else None,
        "notes": [
            "Assumes production text lengths, schema, rule density, hardware and settings resemble this input.",
            "Worker count assumes linear scaling; shared CPU, GPU, storage and memory bandwidth can prevent it.",
            "RSS sizing excludes operating-system and other-service needs; shared pages may be counted repeatedly.",
            "GPU utilization is card-wide and cannot determine GPU count; test concurrent workers and GPU memory separately.",
            "Sampled peaks can miss short spikes. Validate estimates with a sustained representative workload.",
        ],
    }


def _first_warmup_batch(source, args):
    saw_records = False
    for batch in batches(source, args):
        saw_records = True
        if any(text.strip() for text in batch):
            return batch
    if not saw_records:
        raise BenchmarkValidationError("input_has_no_records")
    raise BenchmarkValidationError("input_has_no_nonblank_records")


def run_benchmark(args):
    validate_args(args)
    source, report_dir, output = _validate_paths(args)
    args.report_dir = report_dir
    report_path = report_dir / "report.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(source), "file_bytes": source.stat().st_size},
        "parameters": _json_value(vars(args)),
        "output": str(output) if output is not None else None,
        "rounds": [],
        "measurement_notes": [
            "Repeated passes reuse the same loaded model and may benefit from operating-system file caches.",
            "Resource monitoring and CSV output consume some CPU and I/O, included in the measured workload.",
            "Initialization measures Pipeline construction and may include model download when not cached.",
            "When enabled, every measured pass writes output; the final file contains the last pass.",
            "Record throughput includes blank input lines; blank and nonblank input counts are reported separately.",
            "No input text or extracted sensitive entities are written to the resource report or timeline.",
        ],
    }
    monitor = None
    failure = None
    try:
        report["environment"] = _environment()
        report["environment"]["thread_environment_before_initialization"] = dict(report["environment"].get("thread_environment", {}))
        warmup_batch = _first_warmup_batch(source, args)
        monitor = ResourceMonitor(report_dir / "samples.csv", interval=args.sample_interval)
        monitor.start()
        started = time.perf_counter()
        engine = Pipeline(args)
        report["initialization"] = {
            "seconds": time.perf_counter() - started,
            "requested_device": args.device,
            "effective_device": engine.device,
        }
        report["environment"]["thread_environment"] = {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        }
        monitor.set_phase("warmup")
        started = time.perf_counter()
        for _ in range(args.warmup):
            engine.process(warmup_batch)
        report["warmup"] = {
            "batches": args.warmup,
            "records_per_batch": len(warmup_batch),
            "characters_per_batch": sum(len(text) for text in warmup_batch),
            "blank_records_per_batch": sum(not text.strip() for text in warmup_batch),
            "seconds": time.perf_counter() - started,
            "excluded_from_throughput": True,
        }
        del warmup_batch
        for number in range(1, args.repeats + 1):
            monitor.set_phase("measure", round_number=number)
            stats = run_file(engine, args, output=output)
            report["rounds"].append({"round": number, **stats, "wrote_output": output is not None})
        first = report["rounds"][0]
        blank_records = first.get("blank_records")
        report["input"].update({
            "records": first["records"], "characters": first["chars"],
            "blank_records": blank_records,
            "nonblank_records": first["records"] - blank_records if blank_records is not None else None,
            "mean_characters_per_record": first["chars"] / first["records"],
            "max_characters_per_record": first.get("max_record_chars"),
        })
        report["summary"] = _summarize(report["rounds"])
        report["status"] = "completed"
    except BaseException as exc:
        failure = exc
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc) if isinstance(exc, BenchmarkValidationError) else
                       "Benchmark failed; raw exception text is omitted to prevent input data disclosure.",
        }
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception as exc:
                report["monitor_stop_error"] = {"type": type(exc).__name__}
                report["status"] = "failed"
                failure = failure or exc
            report["resources"] = monitor.report()
        incomplete = bool(report.get("resources", {}).get("sampling_errors"))
        report["resource_measurement_status"] = "degraded" if incomplete else (
            "completed" if report["status"] == "completed" else "incomplete"
        )
        if report["status"] == "completed":
            report["capacity_estimate"] = (
                {"status": "unavailable", "reason": "telemetry_incomplete"} if incomplete
                else _capacity(args, report["summary"], report.get("resources", {}))
            )
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    if failure is not None:
        # Do not propagate a potentially sensitive exception or chained traceback.
        raise RuntimeError("benchmark_failed_see_sanitized_report") from None
    return report_path, report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        path, report = run_benchmark(args)
    except BenchmarkValidationError as exc:
        print(f"Benchmark configuration error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Benchmark failed ({type(exc).__name__}). Check arguments and the sanitized report if created.", file=sys.stderr)
        return 1
    summary = report["summary"]
    print(f"Report: {path}")
    print(f"Throughput: {summary['records_per_second']:.2f} records/s; {summary['chars_per_second']:.2f} chars/s")
    resources = report["resources"]
    if report["resource_measurement_status"] == "degraded":
        print("Resource telemetry is incomplete; capacity estimates are disabled. Inspect sampling_errors.")
    peak_rss = resources.get("overall", {}).get("cpu_memory", {}).get("process_tree_rss_bytes", {}).get("max")
    if peak_rss is not None:
        print(f"Peak process-tree RSS (includes initialization): {peak_rss / 1024**3:.3f} GiB")
    measure_cpu = resources.get("phases", {}).get("measure", {}).get("cpu_memory", {})
    for metric, label in (("process_tree_cpu_percent", "Process-tree CPU (100% = one logical CPU)"), ("host_cpu_percent", "Host CPU")):
        values = measure_cpu.get(metric, {})
        if values.get("mean") is not None:
            print(f"{label}, measured mean/max: {values['mean']:.1f}% / {values['max']:.1f}%")
    print("GPU telemetry: " + ("available" if report["resources"]["gpu_available"] else "unavailable"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
