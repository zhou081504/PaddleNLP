"""Benchmark tests use fake engines and telemetry; they never download models."""

import csv
import ctypes
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import run
from benchmarks.monitor import NvidiaReader, ProcessResourceMonitor, ResourceMonitor, RunningMetric


class FakeMonitor:
    instances = []

    def __init__(self, path, interval):
        self.path = path
        self.stopped = False
        self.phases = ["initialize"]
        self.__class__.instances.append(self)

    def start(self):
        self.path.write_text("phase,cpu\ninitialize,1\n", encoding="utf-8")

    def set_phase(self, phase, round_number=None):
        self.phases.append(phase)

    def stop(self):
        self.stopped = True

    def report(self):
        return {
            "gpu_available": False,
            "overall": {"cpu_memory": {
                "process_tree_rss_bytes": {"max": 1000},
                "process_tree_cpu_percent": {"max": 150.0},
            }},
        }


class FakePipeline:
    instances = []

    def __init__(self, args):
        self.device = "cpu"
        self.calls = []
        self.__class__.instances.append(self)

    def process(self, lines):
        self.calls.append(list(lines))
        return ["MASKED\n" for _ in lines], {"regex_seconds": 0.0, "model_seconds": 0.0, "replace_seconds": 0.0}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.input = self.root / "input.txt"
        self.input.write_text("\nSENSITIVE FIRST\nSENSITIVE SECOND\n", encoding="utf-8")
        self.report_dir = self.root / "report"
        self.patchers = [
            patch.object(run, "ResourceMonitor", FakeMonitor),
            patch.object(run, "Pipeline", FakePipeline),
            patch.object(run, "_environment", return_value={"test": True}),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.directory.cleanup)

    def args(self, *extra):
        return run.build_parser().parse_args([
            "--input", str(self.input), "--report-dir", str(self.report_dir),
            "--mode", "rules", "--batch-records", "1", *extra,
        ])

    def test_report_warmup_and_repeats_do_not_contain_input_text(self):
        path, report = run.run_benchmark(self.args("--repeats", "2", "--target-records", "100", "--target-hours", "1"))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["input"]["records"], 3)
        self.assertEqual(report["input"]["blank_records"], 1)
        self.assertEqual(report["input"]["nonblank_records"], 2)
        self.assertEqual(report["summary"]["records"], 6)
        self.assertEqual(report["summary"]["completed_rounds"], 2)
        self.assertEqual(FakePipeline.instances[-1].calls[0], ["SENSITIVE FIRST\n"])
        self.assertEqual(len(FakePipeline.instances[-1].calls), 7)
        self.assertTrue(FakeMonitor.instances[-1].stopped)
        self.assertNotIn("SENSITIVE", path.read_text(encoding="utf-8"))
        self.assertFalse((self.report_dir / "redacted.txt").exists())
        self.assertEqual(report["capacity_estimate"]["observed_peak_cpu_cores"], 1.5)

    def test_every_pass_writes_output(self):
        with patch.object(run, "run_file", wraps=run.run_file) as run_file:
            _, report = run.run_benchmark(self.args("--write-output", "--repeats", "2"))
        self.assertEqual(len(run_file.call_args_list), 2)
        for call in run_file.call_args_list:
            self.assertEqual(call.kwargs["output"], self.report_dir / "redacted.txt")
        self.assertEqual((self.report_dir / "redacted.txt").read_text(), "MASKED\n" * 3)
        self.assertTrue(all(item["wrote_output"] for item in report["rounds"]))

    def test_failure_stops_monitor_and_omits_exception_message(self):
        with patch.object(FakePipeline, "process", side_effect=RuntimeError("SENSITIVE FAILURE")):
            with self.assertRaisesRegex(RuntimeError, "benchmark_failed"):
                run.run_benchmark(self.args())
        self.assertTrue(FakeMonitor.instances[-1].stopped)
        text = (self.report_dir / "report.json").read_text()
        self.assertNotIn("SENSITIVE", text)
        self.assertEqual(json.loads(text)["status"], "failed")

    def test_sampling_errors_disable_capacity_estimates(self):
        with patch.object(FakeMonitor, "report", return_value={"sampling_errors": {"OSError": 1}}):
            _, report = run.run_benchmark(self.args("--target-records", "100"))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["resource_measurement_status"], "degraded")
        self.assertEqual(report["capacity_estimate"], {"status": "unavailable", "reason": "telemetry_incomplete"})

    def test_empty_input_fails_with_explicit_safe_reason(self):
        self.input.write_text("")
        with self.assertRaises(RuntimeError):
            run.run_benchmark(self.args())
        report = json.loads((self.report_dir / "report.json").read_text())
        self.assertEqual(report["error"]["message"], "input_has_no_records")

    def test_blank_only_input_cannot_inflate_hybrid_benchmark_speed(self):
        self.input.write_text("\n   \n\t\n")
        with self.assertRaises(RuntimeError):
            run.run_benchmark(self.args())
        report = json.loads((self.report_dir / "report.json").read_text())
        self.assertEqual(report["error"]["message"], "input_has_no_nonblank_records")

    def test_thread_environment_is_captured_after_pipeline_initialization(self):
        class ThreadSettingPipeline(FakePipeline):
            def __init__(self, args):
                super().__init__(args)
                os.environ["OMP_NUM_THREADS"] = "2"

        with patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}), patch.object(run, "Pipeline", ThreadSettingPipeline):
            _, report = run.run_benchmark(self.args("--repeats", "1"))
        self.assertEqual(report["environment"]["thread_environment"]["OMP_NUM_THREADS"], "2")

    def test_existing_report_directory_is_never_overwritten(self):
        self.report_dir.mkdir()
        sentinel = self.report_dir / "report.json"
        sentinel.write_text("preserve")
        with self.assertRaisesRegex(run.BenchmarkValidationError, "already_exists"):
            run.run_benchmark(self.args())
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_output_input_and_hardlink_collisions_are_rejected(self):
        hardlink = self.root / "hardlink.txt"
        os.link(self.input, hardlink)
        for output in (self.input, hardlink):
            with self.subTest(output=output), self.assertRaisesRegex(run.BenchmarkValidationError, "conflicts"):
                run.run_benchmark(self.args("--write-output", "--output", str(output)))
        self.assertFalse(self.report_dir.exists())

    def test_output_cannot_replace_report_or_config(self):
        for output in (self.report_dir / "report.json", self.report_dir / "samples.csv", self.report_dir / "report.json" / "child"):
            with self.subTest(output=output), self.assertRaisesRegex(run.BenchmarkValidationError, "conflicts"):
                run.run_benchmark(self.args("--write-output", "--output", str(output)))
        config = self.root / "config.json"
        config.write_text("{}")
        with self.assertRaisesRegex(run.BenchmarkValidationError, "conflicts"):
            run.run_benchmark(self.args("--write-output", "--output", str(config), "--config", str(config)))

    def test_capacity_uses_weighted_throughput_and_reserves_capacity(self):
        args = self.args("--target-records", "36000", "--target-hours", "1", "--headroom", "0.5")
        summary = run._summarize([
            {"records": 100, "chars": 500, "wall_seconds": 10},
            {"records": 100, "chars": 500, "wall_seconds": 20},
        ])
        estimate = run._capacity(args, summary, {"overall": {"cpu_memory": {}}})
        self.assertAlmostEqual(summary["records_per_second"], 200 / 30)
        self.assertEqual(estimate["estimated_concurrent_workers"], 3)
        self.assertIsNone(estimate["planning_total_rss_bytes_with_headroom"])


class MonitoringTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "uses POSIX usleep to emulate native inference holding the GIL")
    def test_process_sampler_runs_while_native_code_holds_parent_gil(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.csv"
            monitor = ProcessResourceMonitor(path, interval=.05, gpu_timeout=.05)
            monitor.start()
            try:
                monitor.set_phase("measure", 1)
                ctypes.PyDLL(None).usleep(400000)
            finally:
                monitor.stop()
            report = monitor.report()
            self.assertEqual(report["sampler_execution"], "separate_process")
            self.assertFalse(report["sampling_errors"])
            self.assertGreaterEqual(report["phases"]["measure"]["samples"], 3)
            self.assertFalse(monitor._process.is_alive())

    def test_running_metric_ignores_unavailable_values(self):
        metric = RunningMetric()
        metric.add(None)
        self.assertEqual(metric.report(), {"samples": 0, "mean": None, "max": None})
        metric.add(2)
        metric.add(6)
        self.assertEqual(metric.report(), {"samples": 2, "mean": 4, "max": 6})

    def test_gpu_unavailable_has_no_fabricated_zero(self):
        reader = NvidiaReader()
        reader.executable = None
        self.assertEqual(reader({1}), {"status": "unavailable", "reason": "nvidia_smi_not_found", "devices": []})
        reader.executable = "nvidia-smi"
        with patch("benchmarks.monitor.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 2)):
            self.assertEqual(reader({1})["reason"], "nvidia_smi_timeout")

    def test_gpu_process_memory_sums_only_the_current_tree(self):
        reader = NvidiaReader()
        reader.executable = "nvidia-smi"
        replies = [
            SimpleNamespace(stdout="0, GPU-one, Test GPU, 73, 512, 8192\n"),
            SimpleNamespace(stdout="GPU-one, 10, 100\nGPU-one, 11, 200\nGPU-one, 99, 99\n"),
        ]
        with patch("benchmarks.monitor.subprocess.run", side_effect=replies):
            result = reader({10, 11})
        device = result["devices"][0]
        self.assertEqual(device["process_tree_memory_mib"], 300)
        self.assertEqual(device["memory_used_mib"], 512)
        self.assertEqual(device["utilization_percent"], 73)

    def test_cpu_sums_process_tree_and_normalizes_by_logical_cpus(self):
        class Process:
            def __init__(self, pid, rss):
                self.pid, self.rss, self.cpu = pid, rss, 0.0

            def create_time(self):
                return 100

            def cpu_times(self):
                return SimpleNamespace(user=self.cpu, system=0)

            def memory_info(self):
                return SimpleNamespace(rss=self.rss)

            def children(self, recursive):
                return [child]

        parent, child = Process(10, 128), Process(11, 256)
        fake_psutil = SimpleNamespace(
            Error=RuntimeError, Process=lambda _: parent, cpu_count=lambda logical: 4,
            cpu_percent=lambda interval: 55,
            virtual_memory=lambda: SimpleNamespace(used=1000, available=3000, total=4000, percent=25),
        )
        monitor = ResourceMonitor(Path("unused.csv"), psutil_module=fake_psutil)
        with patch("benchmarks.monitor.time.monotonic", side_effect=[0, 1]):
            metrics, _ = monitor._process_metrics()
            self.assertIsNone(metrics["process_tree_cpu_percent"])
            self.assertIsNone(metrics["host_cpu_percent"])
            parent.cpu, child.cpu = 0.3, 0.2
            metrics, pids = monitor._process_metrics()
        self.assertEqual(pids, {10, 11})
        self.assertEqual(metrics["process_tree_rss_bytes"], 384)
        self.assertEqual(metrics["process_tree_cpu_percent"], 50)
        self.assertEqual(metrics["process_tree_cpu_normalized_percent"], 12.5)
        self.assertEqual(metrics["host_cpu_percent"], 55)

    def test_monitor_streams_phase_boundaries_and_stops_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.csv"
            monitor = ResourceMonitor(path, gpu_reader=lambda pids: {"status": "unavailable", "reason": "test", "devices": []})
            monitor.start()
            monitor.set_phase("warmup")
            monitor.set_phase("measure", round_number=1)
            monitor.stop()
            self.assertFalse(monitor._thread.is_alive())
            report = monitor.report()
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(set(row["phase"] for row in rows), {"initialize", "warmup", "measure"})
            self.assertTrue(all(row["gpu_utilization_percent"] == "" for row in rows))
            self.assertFalse(report["gpu_available"])
            self.assertGreater(report["overall"]["cpu_memory"]["process_tree_rss_bytes"]["max"], 0)
            self.assertIn("1", report["rounds"])


if __name__ == "__main__":
    unittest.main()
