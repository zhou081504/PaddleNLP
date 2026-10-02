"""Process-tree resource sampling, streamed to CSV without retaining samples."""

from __future__ import annotations

import csv
import math
import multiprocessing
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class RunningMetric:
    """Constant-space sample statistics; the mean is not time-weighted."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = None

    def add(self, value):
        if value is not None:
            self.count += 1
            self.total += value
            self.maximum = value if self.maximum is None else max(self.maximum, value)

    def report(self):
        return {
            "samples": self.count,
            "mean": self.total / self.count if self.count else None,
            "max": self.maximum,
        }


class NvidiaReader:
    """Read card-wide metrics and this process tree's GPU memory, if supported."""

    def __init__(self, timeout=2.0):
        self.executable = shutil.which("nvidia-smi")
        self.timeout = timeout

    def _query(self, query):
        result = subprocess.run(
            [self.executable, query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=True,
        )
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    def __call__(self, process_ids):
        if self.executable is None:
            return {"status": "unavailable", "reason": "nvidia_smi_not_found", "devices": []}
        try:
            rows = self._query("--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total")
        except subprocess.TimeoutExpired:
            return {"status": "unavailable", "reason": "nvidia_smi_timeout", "devices": []}
        except (OSError, subprocess.CalledProcessError):
            return {"status": "unavailable", "reason": "nvidia_smi_query_failed", "devices": []}
        devices = []
        for row in rows:
            if len(row) != 6:
                continue
            devices.append(
                {
                    "index": row[0].strip(),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "utilization_percent": _number(row[3]),
                    "memory_used_mib": _number(row[4]),
                    "memory_total_mib": _number(row[5]),
                    "process_tree_memory_mib": None,
                }
            )
        if not devices:
            return {"status": "unavailable", "reason": "no_gpu_devices_reported", "devices": []}
        process_memory_status = "available"
        try:
            processes = self._query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory")
            gpu_totals = {device["uuid"]: 0.0 for device in devices}
            for row in processes:
                if len(row) != 3:
                    raise ValueError("invalid_gpu_process_row")
                if int(row[1]) not in process_ids:
                    continue
                gpu_uuid = row[0].strip()
                if gpu_uuid not in gpu_totals:
                    # A MIG identifier may not map to a physical-card UUID.
                    raise ValueError("gpu_process_uuid_cannot_be_mapped_to_card")
                value = _number(row[2])
                if value is None:
                    gpu_totals[gpu_uuid] = None
                elif gpu_totals.get(gpu_uuid) is not None:
                    gpu_totals[gpu_uuid] += value
            for device in devices:
                device["process_tree_memory_mib"] = gpu_totals[device["uuid"]]
            if any(value is None for value in gpu_totals.values()):
                process_memory_status = "partially_unavailable"
        except (OSError, subprocess.SubprocessError, ValueError):
            process_memory_status = "unavailable"
        return {
            "status": "available",
            "reason": None,
            "process_memory_status": process_memory_status,
            "devices": devices,
        }


class ResourceMonitor:
    FIELDS = [
        "elapsed_seconds", "phase", "round", "process_count",
        "process_tree_cpu_percent", "process_tree_cpu_normalized_percent", "host_cpu_percent",
        "process_tree_rss_bytes", "host_memory_used_bytes", "host_memory_available_bytes",
        "host_memory_total_bytes", "host_memory_percent", "gpu_status", "gpu_reason",
        "gpu_process_memory_status", "gpu_index", "gpu_uuid", "gpu_name",
        "gpu_utilization_percent", "gpu_memory_used_mib", "gpu_memory_total_mib",
        "process_tree_gpu_memory_mib",
    ]
    CPU_FIELDS = (
        "process_tree_cpu_percent", "process_tree_cpu_normalized_percent", "host_cpu_percent",
        "process_tree_rss_bytes", "host_memory_used_bytes", "host_memory_available_bytes",
        "host_memory_percent",
    )
    GPU_FIELDS = ("utilization_percent", "memory_used_mib", "process_tree_memory_mib")

    def __init__(self, csv_path: Path, interval=0.5, gpu_timeout=2.0, psutil_module=None, gpu_reader=None, root_pid=None):
        if interval <= 0:
            raise ValueError("sample_interval_must_be_positive")
        if psutil_module is None:
            import psutil as psutil_module
        self.psutil = psutil_module
        self.csv_path = Path(csv_path)
        self.interval = interval
        self.gpu_timeout = gpu_timeout
        self.gpu_reader = gpu_reader or NvidiaReader(gpu_timeout)
        self.process = self.psutil.Process(root_pid if root_pid is not None else os.getpid())
        self.logical_cpus = self.psutil.cpu_count(logical=True) or 1
        self._processes = {}
        self._cpu_times = {}
        self._last_cpu_sample = None
        self._host_cpu_threads = set()
        self._phase = "initialize"
        self._round = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._stream = None
        self._writer = None
        self._started = None
        self._groups = {}
        self._gpu_metadata = {}
        self._gpu_status_counts = {}
        self._errors = {}
        self._sample_spacing = RunningMetric()
        self._last_checkpoint = None

    def start(self):
        self._stream = self.csv_path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._started = time.monotonic()
        self.checkpoint()
        self._thread = threading.Thread(target=self._loop, name="redaction-resource-monitor", daemon=True)
        self._thread.start()
        return self

    def _process_metrics(self):
        now = time.monotonic()
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except self.psutil.Error:
            pass
        seen = set()
        next_cpu_times = {}
        rss = 0
        cpu_seconds = 0.0
        for process in processes:
            try:
                # A PID may be reused after a short-lived child exits.
                key = (process.pid, process.create_time())
                values = process.cpu_times()
                cpu_time = values.user + values.system
                next_cpu_times[key] = cpu_time
                if key in self._cpu_times:
                    cpu_seconds += max(0.0, cpu_time - self._cpu_times[key])
                rss += process.memory_info().rss
                seen.add(process.pid)
            except self.psutil.Error:
                continue
        elapsed = now - self._last_cpu_sample if self._last_cpu_sample is not None else None
        cpu = 100.0 * cpu_seconds / elapsed if elapsed and elapsed > 0 else None
        self._cpu_times = next_cpu_times
        self._last_cpu_sample = now
        # psutil maintains the nonblocking host-CPU baseline separately per thread.
        thread_id = threading.get_ident()
        host_cpu = self.psutil.cpu_percent(interval=None)
        if thread_id not in self._host_cpu_threads:
            self._host_cpu_threads.add(thread_id)
            host_cpu = None
        memory = self.psutil.virtual_memory()
        return {
            "process_count": len(seen),
            "process_tree_cpu_percent": cpu,
            "process_tree_cpu_normalized_percent": cpu / self.logical_cpus if cpu is not None else None,
            "host_cpu_percent": host_cpu,
            "process_tree_rss_bytes": rss,
            "host_memory_used_bytes": memory.used,
            "host_memory_available_bytes": memory.available,
            "host_memory_total_bytes": memory.total,
            "host_memory_percent": memory.percent,
        }, seen

    def _accumulate(self, group_name, metrics, gpu):
        group = self._groups.setdefault(group_name, {"samples": 0, "cpu_memory": {}, "gpus": {}})
        group["samples"] += 1
        for name in self.CPU_FIELDS:
            group["cpu_memory"].setdefault(name, RunningMetric()).add(metrics[name])
        for device in gpu["devices"]:
            device_group = group["gpus"].setdefault(device["uuid"], {})
            self._gpu_metadata[device["uuid"]] = {
                key: device[key] for key in ("index", "uuid", "name", "memory_total_mib")
            }
            for name in self.GPU_FIELDS:
                device_group.setdefault(name, RunningMetric()).add(device[name])

    def checkpoint(self):
        """Capture a phase boundary as well as regular background samples."""
        with self._lock:
            if self._stream is None:
                return
            metrics, process_ids = self._process_metrics()
            gpu = self.gpu_reader(process_ids)
            sampled_at = time.monotonic()
            if self._last_checkpoint is not None:
                self._sample_spacing.add(sampled_at - self._last_checkpoint)
            self._last_checkpoint = sampled_at
            status_key = gpu["reason"] or gpu["status"]
            self._gpu_status_counts[status_key] = self._gpu_status_counts.get(status_key, 0) + 1
            base = {
                "elapsed_seconds": time.monotonic() - self._started,
                "phase": self._phase,
                "round": self._round,
                **metrics,
                "gpu_status": gpu["status"],
                "gpu_reason": gpu.get("reason"),
                "gpu_process_memory_status": gpu.get("process_memory_status", "unavailable"),
            }
            for device in gpu["devices"] or [None]:
                row = dict(base)
                if device:
                    row.update({
                        "gpu_index": device["index"], "gpu_uuid": device["uuid"],
                        "gpu_name": device["name"], "gpu_utilization_percent": device["utilization_percent"],
                        "gpu_memory_used_mib": device["memory_used_mib"],
                        "gpu_memory_total_mib": device["memory_total_mib"],
                        "process_tree_gpu_memory_mib": device["process_tree_memory_mib"],
                    })
                self._writer.writerow(row)
            self._stream.flush()
            self._accumulate("overall", metrics, gpu)
            self._accumulate("phase:" + self._phase, metrics, gpu)
            if self._round is not None:
                self._accumulate("round:" + str(self._round), metrics, gpu)

    def set_phase(self, phase, round_number=None):
        with self._lock:
            self.checkpoint()
            self._phase = phase
            self._round = round_number
            # Reset CPU deltas so a new phase does not inherit work from the old phase.
            self._process_metrics()

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.checkpoint()
            except Exception as exc:
                # Exception messages can contain data; only retain the exception type.
                name = type(exc).__name__
                self._errors[name] = self._errors.get(name, 0) + 1

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.gpu_timeout + self.interval + 5)
            if self._thread.is_alive():
                raise RuntimeError("resource_monitor_did_not_stop")
        with self._lock:
            if self._stream is not None:
                try:
                    self.checkpoint()
                finally:
                    self._stream.close()
                    self._stream = None

    def report(self):
        with self._lock:
            groups = {}
            for name, group in self._groups.items():
                groups[name] = {
                    "samples": group["samples"],
                    "cpu_memory": {key: value.report() for key, value in group["cpu_memory"].items()},
                    "gpus": {
                        uuid: {**self._gpu_metadata[uuid], **{key: value.report() for key, value in metrics.items()}}
                        for uuid, metrics in group["gpus"].items()
                    },
                }
            return {
                "sample_interval_seconds": self.interval,
                "actual_sample_spacing_seconds": self._sample_spacing.report(),
                "logical_cpu_count": self.logical_cpus,
                "gpu_available": bool(self._gpu_metadata),
                "gpu_observation_counts": dict(self._gpu_status_counts),
                "sampling_errors": dict(self._errors),
                "overall": groups.get("overall", {}),
                "phases": {key[6:]: value for key, value in groups.items() if key.startswith("phase:")},
                "rounds": {key[6:]: value for key, value in groups.items() if key.startswith("round:")},
                "notes": [
                    "CPU 100% means one logical CPU; normalized CPU divides by logical CPU count.",
                    "Host CPU describes the entire machine, including other workloads, on a 0-100% scale.",
                    "RSS sums the main process and observed descendants; shared pages may be counted more than once.",
                    "GPU utilization and GPU total memory usage describe the entire card, including other workloads.",
                    "Process-tree GPU memory is queried separately and is null when unsupported or unavailable.",
                    "Means are sample averages, including phase-boundary snapshots; peaks are sampled lower bounds.",
                    "CPU counters have OS time granularity; a very short pass can report 0.0% despite using CPU.",
                    "Short-lived child processes between samples and brief resource spikes may be missed.",
                    "CSV has one row per GPU per sample; CPU fields repeat across GPU rows.",
                ],
            }


def _monitor_worker(connection, csv_path, interval, gpu_timeout, root_pid):
    """A separate interpreter keeps sampling while native inference holds the GIL."""
    monitor = None
    try:
        monitor = ResourceMonitor(csv_path, interval=interval, gpu_timeout=gpu_timeout, root_pid=root_pid)
        monitor.start()
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "phase":
                monitor.set_phase(*payload)
                connection.send(("ok", None))
            elif command == "stop":
                monitor.stop()
                result = monitor.report()
                result["sampler_execution"] = "separate_process"
                result["notes"].append("Process-tree metrics include the sampling subprocess; it does not load the model.")
                connection.send(("stopped", result))
                monitor = None
                return
            else:
                raise ValueError("invalid_monitor_command")
    except EOFError:
        pass
    except BaseException as error:
        try:
            connection.send(("error", type(error).__name__))
        except (OSError, EOFError):
            pass
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
        connection.close()


class ProcessResourceMonitor:
    """Parent-side controller; no resource reads depend on the inference thread."""

    def __init__(self, csv_path, interval=0.5, gpu_timeout=2.0):
        self.csv_path = Path(csv_path)
        self.interval = interval
        self.gpu_timeout = gpu_timeout
        self._timeout = max(15., 4 * gpu_timeout + 5.)
        self._process = None
        self._connection = None
        self._report = {"sampling_errors": {"MonitorNotCompleted": 1}}

    def _receive(self, expected):
        if not self._connection.poll(self._timeout):
            raise RuntimeError("resource_monitor_response_timeout")
        status, result = self._connection.recv()
        if status != expected:
            raise RuntimeError("resource_monitor_failed")
        return result

    def start(self):
        context = multiprocessing.get_context("spawn")
        self._connection, worker_connection = context.Pipe()
        self._process = context.Process(
            target=_monitor_worker,
            args=(worker_connection, self.csv_path, self.interval, self.gpu_timeout, os.getpid()),
            name="redaction-resource-monitor",
            daemon=True,
        )
        self._process.start()
        worker_connection.close()
        self._receive("ready")
        return self

    def set_phase(self, phase, round_number=None):
        self._connection.send(("phase", (phase, round_number)))
        self._receive("ok")

    def stop(self):
        if self._process is None:
            return
        try:
            if self._process.is_alive():
                self._connection.send(("stop", None))
                self._report = self._receive("stopped")
                self._process.join(timeout=self._timeout)
            else:
                raise RuntimeError("resource_monitor_exited_early")
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.)
            self._connection.close()

    def report(self):
        return self._report
