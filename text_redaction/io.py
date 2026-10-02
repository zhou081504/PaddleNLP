"""有界流式读取和原子输出，不在内存中积累全量文本。"""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
import time


def same_file(left, right):
    left, right = Path(left), Path(right)
    return left.resolve() == right.resolve() or (left.exists() and right.exists() and os.path.samefile(left, right))


def records(path, max_chars):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        line_number = 0
        while True:
            line = stream.readline(max_chars + 1)
            if not line:
                return
            line_number += 1
            if len(line) > max_chars:
                raise ValueError(f"第 {line_number} 行超过 max-record-chars（含换行）；请调整上限")
            yield line


def batches(path, args):
    if args.max_record_chars > args.batch_chars:
        raise ValueError("max-record-chars 不能大于 batch-chars")
    batch, size = [], 0
    for line in records(path, args.max_record_chars):
        if batch and (len(batch) >= args.batch_records or size + len(line) > args.batch_chars):
            yield batch
            batch, size = [], 0
        batch.append(line)
        size += len(line)
    if batch:
        yield batch


class Latencies:
    """固定空间 reservoir：小任务精确，大任务估算分位数。"""
    def __init__(self, capacity=4096):
        self.capacity, self.count, self.total, self.maximum = capacity, 0, 0., 0.
        self.samples = []
        self.random = random.Random(0)

    def add(self, seconds):
        self.count += 1
        self.total += seconds
        self.maximum = max(self.maximum, seconds)
        if len(self.samples) < self.capacity:
            self.samples.append(seconds)
        else:
            index = self.random.randrange(self.count)
            if index < self.capacity:
                self.samples[index] = seconds

    def summary(self):
        ordered = sorted(self.samples)
        def percentile(q):
            if not ordered:
                return None
            position = (len(ordered) - 1) * q
            low = int(position)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        return dict(count=self.count, mean=self.total / self.count if self.count else None,
                    p50=percentile(.5), p95=percentile(.95), max=self.maximum if self.count else None,
                    sample_count=len(ordered), approximate=self.count > self.capacity,
                    scope="pipeline.process per batch; excludes file I/O")


def run_file(engine, args, output=None):
    measured = not getattr(args, "no_benchmark", False)
    clock = time.perf_counter if measured else lambda: 0.
    if output is not None:
        output = Path(output)
        if same_file(args.input, output) or (args.config and same_file(args.config, output)):
            raise ValueError("输出不能覆盖输入或配置文件")
    stats = dict(records=0, blank_records=0, chars=0, batches=0, regex_seconds=0., model_seconds=0., replace_seconds=0.,
                 max_record_chars=0)
    latency = Latencies()
    temporary, stream = None, None
    started = clock()
    try:
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                                                 prefix=".redact-", delete=False)
            temporary = Path(stream.name)
        for batch in batches(args.input, args):
            tick = clock()
            result, timing = engine.process(batch)
            if measured:
                latency.add(clock() - tick)
            if len(result) != len(batch) or any(not isinstance(item, str) for item in result):
                raise ValueError("流水线输出条数或类型不正确")
            if stream is not None:
                stream.writelines(result)
            stats["records"] += len(batch)
            stats["blank_records"] += sum(not text.strip() for text in batch)
            stats["chars"] += sum(map(len, batch))
            stats["max_record_chars"] = max(stats["max_record_chars"], max(map(len, batch)))
            stats["batches"] += 1
            for key in ("regex_seconds", "model_seconds", "replace_seconds"):
                stats[key] += timing[key]
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            stream = None
            os.replace(temporary, output)
            temporary = None
    finally:
        try:
            if stream is not None:
                stream.close()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    stats["wall_seconds"] = clock() - started
    stats["processing_seconds"] = sum(stats[k] for k in ("regex_seconds", "model_seconds", "replace_seconds"))
    stats["io_and_overhead_seconds"] = max(0., stats["wall_seconds"] - stats["processing_seconds"])
    for metric in ("records", "chars"):
        stats[metric + "_per_second"] = stats[metric] / stats["wall_seconds"] if stats["wall_seconds"] else 0.
    stats["batch_latency_seconds"] = latency.summary()
    return stats
