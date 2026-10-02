"""与 speech_demo 一致的目录批处理：复用模型、跳过已有、失败继续。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import tempfile
import time

from .cli import build_parser as base_parser, validate_args
from .io import run_file
from .pipeline import Pipeline

PROJECT = Path(__file__).resolve().parents[1]


def discover_inputs(input_dir, output_dir, exclude_dirs=()):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError("输入必须是文件夹")
    if source == output or source.is_relative_to(output):
        raise ValueError("输出不能等于输入目录或位于输入目录的上层")
    excluded = [output, *(Path(p).resolve() for p in exclude_dirs)]
    return sorted(p for p in source.rglob('*') if p.is_file() and not p.is_symlink()
                  and p.suffix.lower() == '.txt' and not any(p.resolve().is_relative_to(d) for d in excluded))


def input_signature(input_dir, output_dir):
    root = Path(input_dir).resolve()
    return [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in discover_inputs(root, output_dir)]


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def resolve_device(args):
    requested = args.device
    if requested == 'auto':
        if args.mode == 'rules':
            args.device = 'cpu'
            return
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['MKL_NUM_THREADS'] = str(args.threads)
        import paddle
        try:
            available = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > args.device_id
        except Exception:
            available = False
        args.device = 'gpu' if available else 'cpu'
    elif requested == 'cpu':
        return
    elif re.fullmatch(r'(?:gpu|cuda)(?::\d+)?', requested):
        if ':' in requested:
            args.device_id = int(requested.split(':')[1])
        args.device = 'gpu'
    else:
        raise ValueError('设备应为 auto/cpu/gpu/cuda:0 等')


def build_parser():
    parser = base_parser(directory=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / '脱敏结果')
    parser.add_argument('--overwrite', action='store_true', help='重新处理已有非空结果')
    parser.add_argument('--no-benchmark', action='store_true', help='关闭应用计时、资源采样和性能报告')
    parser.add_argument('--exclude-dir', type=Path, action='append', default=[], help='排除目录，可多次指定')
    from benchmarks.run import _positive_float
    parser.add_argument('--sample-interval', type=_positive_float, default=.5)
    return parser


def _protect_outputs(files, args):
    protected = list(files) + ([args.config.resolve()] if args.config else [])
    identities = {(p.stat().st_dev, p.stat().st_ino) for p in protected}
    paths = {p.resolve() for p in protected}
    destinations = [args.output_dir / p.relative_to(args.input_dir) for p in files]
    destinations += [args.output_dir / n for n in ('处理明细.csv', '性能统计.json', '资源采样.csv')]
    resolved = set()
    for target in destinations:
        if target.resolve() in resolved:
            raise ValueError('多个输出路径通过链接指向同一目标')
        resolved.add(target.resolve())
        if not target.resolve().is_relative_to(args.output_dir):
            raise ValueError('输出路径通过链接指向结果目录之外')
        if target.resolve() in paths or (target.exists() and (target.stat().st_dev, target.stat().st_ino) in identities):
            raise ValueError('输出或报告会覆盖输入/配置文件')


def run_directory(args):
    args.input_dir, args.output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    files = discover_inputs(args.input_dir, args.output_dir, args.exclude_dir)
    if not files:
        raise ValueError('输入目录没有 TXT 文件')
    _protect_outputs(files, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enabled = not args.no_benchmark
    clock = time.perf_counter if enabled else lambda: 0.
    started = clock()
    requested = args.device
    counts = dict(files=len(files), success=0, failed=0, skipped=0)
    totals = dict(records=0, blank_records=0, chars=0, processing_seconds=0., wall_seconds=0.,
                  regex_seconds=0., model_seconds=0., replace_seconds=0.)
    monitor, engine, fatal, interrupted = None, None, None, False
    load_seconds = 0.
    resources = {}
    try:
        if enabled:
            from benchmarks.monitor import ProcessResourceMonitor
            # 每次普通运行更新本次采样文件，已完成脱敏文本不受影响。
            (args.output_dir / '资源采样.csv').unlink(missing_ok=True)
            monitor = ProcessResourceMonitor(args.output_dir / '资源采样.csv', interval=args.sample_interval)
            monitor.start()
        with (args.output_dir / '处理明细.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            fields = ['输入文件', '输出文件', '状态', '记录数', '空行数', '字符数', '耗时_秒', '条_每秒', '错误类型']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, source in enumerate(files, 1):
                relative = source.relative_to(args.input_dir)
                output = args.output_dir / relative
                row = {'输入文件': str(relative), '输出文件': str(output)}
                if output.is_file() and output.stat().st_size > 0 and not args.overwrite:
                    counts['skipped'] += 1
                    writer.writerow({**row, '状态': '跳过'})
                    stream.flush()
                    print(f'[{index}/{len(files)}] 跳过：{relative}', flush=True)
                    continue
                args.input = source
                if engine is None:
                    tick = clock()
                    resolve_device(args)
                    validate_args(args)
                    engine = Pipeline(args)
                    load_seconds = clock() - tick
                    if monitor:
                        monitor.set_phase('measure')
                    print(f'脱敏模块已就绪，设备：{engine.device}', flush=True)
                try:
                    stats = run_file(engine, args, output)
                except Exception as error:
                    counts['failed'] += 1
                    writer.writerow({**row, '状态': '失败', '错误类型': type(error).__name__})
                    print(f'[{index}/{len(files)}] 失败：{relative}（{type(error).__name__}），继续下一个', flush=True)
                else:
                    counts['success'] += 1
                    for key in totals:
                        totals[key] += stats[key]
                    writer.writerow({**row, '状态': '成功', '记录数': stats['records'],
                                     '空行数': stats['blank_records'], '字符数': stats['chars'],
                                     '耗时_秒': stats['wall_seconds'] if enabled else '',
                                     '条_每秒': stats['records_per_second'] if enabled else ''})
                    print(f'[{index}/{len(files)}] 完成：{relative}，{stats["records"]} 条', flush=True)
                stream.flush()
    except KeyboardInterrupt:
        fatal, interrupted = 'KeyboardInterrupt', True
    except Exception as error:
        fatal = type(error).__name__
        print(f'批处理终止（{fatal}）；请检查设备、模型、配置及输出权限。', flush=True)
    finally:
        if monitor:
            try:
                monitor.stop()
            except Exception as error:
                fatal = fatal or type(error).__name__
            resources = monitor.report()
    wall = clock() - started
    complete = not fatal and counts['success'] == counts['files'] and totals['records'] > totals['blank_records']
    reliable = not resources.get('sampling_errors')
    report = dict(schema_version=1, status='failed' if fatal or counts['failed'] else 'completed',
                  requested_device=requested, device=engine.device if engine else None,
                  model=args.model, mode=args.mode, model_batch_size=args.model_batch_size,
                  batch_records=args.batch_records, threads=args.threads, **counts,
                  unprocessed=len(files) - sum(counts[k] for k in ('success', 'failed', 'skipped')),
                  records=totals['records'], blank_records=totals['blank_records'], chars=totals['chars'],
                  nonblank_records=totals['records'] - totals['blank_records'],
                  wall_seconds=wall if enabled else None, model_load_seconds=load_seconds if enabled else None,
                  successful_process_seconds=totals['wall_seconds'] if enabled else None,
                  records_per_second=totals['records'] / wall if wall else None,
                  chars_per_second=totals['chars'] / wall if wall else None,
                  stage_seconds={k:totals[k] for k in ('regex_seconds', 'model_seconds', 'replace_seconds')} if enabled else None,
                  valid_for_comparison=bool(complete and reliable and enabled),
                  resource_measurement_status=('completed' if reliable else 'degraded') if enabled else 'disabled',
                  resources=resources, error_type=fatal,
                  parameters={k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
                              for k,v in vars(args).items()},
                  timing_scope='含模型加载、文件读取、脱敏、写盘、失败和采样开销；不含初始Python导入及最后汇总写入',
                  notes=['每行一条记录，空行计入records；所有文件共用一个模型。',
                         '默认跳过非空已有输出；输入或配置改变后需--overwrite。',
                         'GPU整卡指标包含其他进程；容量只适用于同等硬件与相似语料。'])
    if enabled:
        from benchmarks.run import _environment
        report['environment'] = _environment()
        atomic_json(args.output_dir / '性能统计.json', report)
    print(f'完成：成功 {counts["success"]}，失败 {counts["failed"]}，跳过 {counts["skipped"]}；输出：{args.output_dir}', flush=True)
    if enabled:
        print(f'性能统计：{args.output_dir / "性能统计.json"}', flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = run_directory(args)
        return 0 if report['status'] == 'completed' else 1
    except KeyboardInterrupt:
        print('已取消；已完成文件保留。')
        return 130
    except Exception as error:
        print(f'运行失败（{type(error).__name__}）；请检查输入目录、输出路径、设备和配置。')
        return 1
