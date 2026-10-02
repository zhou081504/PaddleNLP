"""按 speech_demo 的方式以独立进程比较目录脱敏的设备和 batch。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from text_redaction.batch import PROJECT, atomic_json, input_signature
from text_redaction.cli import build_parser as base_parser, nonnegative, positive
from .run import _positive_float


def estimate_capacity(speed, *, history_records=0, daily_records=0, deadline_days=30,
                      hours_per_day=20, utilization=.8):
    if any(not math.isfinite(v) or v <= 0 for v in (speed, deadline_days, hours_per_day, utilization)):
        raise ValueError('速度、期限、运行时间和利用率必须为有限正数')
    if hours_per_day > 24 or utilization > 1 or min(history_records, daily_records) < 0:
        raise ValueError('容量参数超出范围')
    capacity = speed * 3600 * hours_per_day * utilization
    workload = bool(history_records or daily_records)
    return dict(records_per_node_per_day=capacity, history_records=history_records, daily_records=daily_records,
                deadline_days=deadline_days, hours_per_day=hours_per_day, utilization=utilization,
                nodes_for_history=math.ceil(history_records / deadline_days / capacity) if workload else None,
                nodes_for_daily=math.ceil(daily_records / capacity) if workload else None,
                nodes_for_history_plus_daily=math.ceil((history_records / deadline_days + daily_records) / capacity) if workload else None,
                scope='仅同等硬件、同等语料的独立工作节点，未验证多节点线性扩展；不是本机并行进程数。')


def rank_results(results, expected_repeats):
    groups = {}
    for row in results:
        groups.setdefault((row['requested_device'], row['model_batch_size']), []).append(row)
    ranked = []
    for (device, batch), runs in groups.items():
        if len(runs) != expected_repeats or not all(r.get('valid_for_comparison') and r.get('records_per_second', 0) > 0 for r in runs):
            continue
        if len({(r['files'], r['records'], r['chars'], r['blank_records'], r['device']) for r in runs}) != 1:
            continue
        peak_rss = [r.get('resources', {}).get('overall', {}).get('cpu_memory', {}).get('process_tree_rss_bytes', {}).get('max') for r in runs]
        ranked.append(dict(requested_device=device, device=runs[0]['device'], model_batch_size=batch,
                           repeat_count=len(runs), median_records_per_second=statistics.median(r['records_per_second'] for r in runs),
                           median_chars_per_second=statistics.median(r['chars_per_second'] for r in runs),
                           peak_rss_bytes=max((v for v in peak_rss if v is not None), default=None)))
    return sorted(ranked, key=lambda r:r['median_records_per_second'], reverse=True)


def write_reports(root, results, args, aborted=None):
    ranked = rank_results(results, args.repeats)
    plans = []
    for device in dict.fromkeys(r['requested_device'] for r in ranked):
        best = next(r for r in ranked if r['requested_device'] == device)
        plan = estimate_capacity(best['median_records_per_second'], history_records=args.history_records,
                                 daily_records=args.daily_records, deadline_days=args.deadline_days,
                                 hours_per_day=args.hours_per_day, utilization=args.utilization)
        plans.append({**best, **plan, 'chars_per_node_per_day':best['median_chars_per_second'] * 3600 * args.hours_per_day * args.utilization})
    atomic_json(root / '评测汇总.json', dict(runs=results, ranked_configurations=ranked, capacity_plans=plans,
                                            aborted=aborted, expected_repeats=args.repeats))
    columns = ['设备请求', '实际设备', '模型批大小', '重复轮次', '返回码', '可参与排名', '成功文件', '失败文件', '跳过文件',
               '记录数', '空行数', '字符数', '端到端秒', '条每秒', '字符每秒', 'CPU平均百分比', 'CPU峰值百分比',
               'RSS峰值GiB', 'GPU可用', 'GPU指标JSON', '结果目录']
    with (root / '评测对比.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            resources = r.get('resources', {})
            metrics = resources.get('overall', {}).get('cpu_memory', {})
            cpu = metrics.get('process_tree_cpu_percent', {})
            rss = metrics.get('process_tree_rss_bytes', {}).get('max')
            writer.writerow(dict(zip(columns, [r['requested_device'], r.get('device'), r['model_batch_size'], r['repeat'],
                            r['returncode'], r['valid_for_comparison'], r.get('success'), r.get('failed'), r.get('skipped'),
                            r.get('records'), r.get('blank_records'), r.get('chars'), r.get('wall_seconds'), r.get('records_per_second'),
                            r.get('chars_per_second'), cpu.get('mean'), cpu.get('max'), rss / 1024**3 if rss is not None else None,
                            resources.get('gpu_available'), json.dumps(resources.get('overall', {}).get('gpus', {}), ensure_ascii=False), r['run_dir']])))
    lines = ['# 脱敏性能与算力评估', '',
             '每组使用全新进程完整重跑同一文件集；速度包括启动、依赖导入、模型加载、全部文件脱敏、写盘、报告和退出。没有额外预热。', '',
             '仅输入集未变化、全部成功且无跳过、无采样异常的完整重复组参与排名，按端到端条/秒中位数排序。', '',
             '|设备|模型批大小|重复次数|中位条/秒|中位字符/秒|RSS峰值GiB|', '|---|---:|---:|---:|---:|---:|']
    for r in ranked:
        ram = f"{r['peak_rss_bytes'] / 1024**3:.3f}" if r['peak_rss_bytes'] is not None else '不可用'
        lines.append(f"|{r['device']}|{r['model_batch_size']}|{r['repeat_count']}|{r['median_records_per_second']:.3f}|{r['median_chars_per_second']:.2f}|{ram}|")
    if not ranked:
        lines += ['', '暂无完整成功的配置可用于排名。失败原因类别和资源缺失见各 case 的性能统计及运行日志。']
    if aborted:
        lines += ['', f'评测提前结束：{aborted}。已完成结果保留。']
    for p in plans:
        lines += ['', f"## {p['device']} 最佳实测配置", '', f"模型批大小 {p['model_batch_size']}；单节点每日约 {p['records_per_node_per_day']:.0f} 条、{p['chars_per_node_per_day']:.0f} 字符（{args.hours_per_day:g} 小时/天，利用率 {args.utilization:g}）。"]
        if args.history_records or args.daily_records:
            lines += [f"历史 {args.history_records} 条需在 {args.deadline_days:g} 天完成，日增 {args.daily_records} 条。",
                      f"仅历史需 {p['nodes_for_history']} 个同等节点；仅日增需 {p['nodes_for_daily']} 个；同时处理需 {p['nodes_for_history_plus_daily']} 个。"]
        else:
            lines += ['未提供业务量，仅报告单节点日容量，不给出设备采购数量。']
    lines += ['', '公式：节点数 = ceil((历史条数 / 完成天数 + 日增条数) / (实测条/秒 × 3600 × 每日运行小时 × 利用率))。', '',
              'CPU 100% 代表一个逻辑核；RSS含采样子进程，共享页可能重复计算。GPU指标是整卡或已标明的进程树显存，缺失为null，采样可能遗漏尖峰。', '',
              '每行一条记录，空行计入条数；不同长度、实体密度、schema和模型不能直接套用同一速度。GPU节点指同型号单卡及其配套主机，CPU节点指同等CPU主机；未验证多节点线性扩展，不等同于在本机启动同样数量的进程。内存/显存需另留余量，吞吐评测不代表脱敏准确率。']
    (root / '算力评估.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_parser():
    p = base_parser(directory=True)
    p.description = __doc__
    p.add_argument('--devices', nargs='+', help='例如 cpu cuda:0；默认auto')
    p.add_argument('--batch-sizes', nargs='+', type=positive, default=[1, 4, 8, 16])
    p.add_argument('--repeats', type=positive, default=1)
    p.add_argument('--output-dir', type=Path, default=PROJECT / '评测结果')
    p.add_argument('--sample-interval', type=_positive_float, default=.5)
    p.add_argument('--history-records', type=nonnegative, default=0)
    p.add_argument('--daily-records', type=nonnegative, default=0)
    p.add_argument('--deadline-days', type=_positive_float, default=30)
    p.add_argument('--hours-per-day', type=_positive_float, default=20)
    p.add_argument('--utilization', type=_positive_float, default=.8)
    return p


def run_evaluation(args):
    if args.hours_per_day > 24 or args.utilization > 1:
        raise ValueError('每日运行小时不能大于24，利用率不能大于1')
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    initial = input_signature(source, destination)
    if not initial:
        raise ValueError('没有TXT输入文件')
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='run_', dir=destination))
    atomic_json(root / '输入清单.json', initial)
    print(f'评测目录：{root}', flush=True)
    results, aborted = [], None
    write_reports(root, results, args)
    try:
        for device in dict.fromkeys(args.devices or [args.device]):
            for batch in dict.fromkeys(args.batch_sizes):
                for repeat in range(1, args.repeats + 1):
                    if input_signature(source, destination) != initial:
                        raise RuntimeError('input_changed')
                    directory = root / f'case_{len(results) + 1:03d}'
                    directory.mkdir()
                    command = [sys.executable, str(PROJECT / 'desensitize.py'), str(source), '--output-dir', str(directory),
                               '--overwrite', '--device', device, '--model-batch-size', str(batch),
                               '--exclude-dir', str(destination)]
                    for key in ('mode', 'threads', 'batch_records', 'batch_chars', 'max_record_chars', 'device_id',
                                'model', 'max_seq_len', 'position_prob', 'name_mask', 'sample_interval'):
                        command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
                    command += ['--schema', *args.schema]
                    for key in ('config', 'model_path'):
                        if getattr(args, key) is not None:
                            command += ['--' + key.replace('_', '-'), str(getattr(args, key).resolve())]
                    print(f'评测 {device} / batch={batch} / 第{repeat}次；日志：{directory / "运行.log"}', flush=True)
                    started = time.perf_counter()
                    with (directory / '运行.log').open('w', encoding='utf-8') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
                        try:
                            code = child.wait()
                        except KeyboardInterrupt:
                            child.terminate()
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            raise
                    elapsed = time.perf_counter() - started
                    row = dict(requested_device=device, model_batch_size=batch, repeat=repeat, returncode=code,
                               run_dir=str(directory), valid_for_comparison=False, wall_seconds=elapsed)
                    summary = directory / '性能统计.json'
                    if summary.is_file():
                        try:
                            payload = json.loads(summary.read_text(encoding='utf-8'))
                            row.update(payload)
                            row.update(requested_device=device, model_batch_size=batch, repeat=repeat, returncode=code,
                                       wall_seconds=elapsed, batch_wall_seconds=payload.get('wall_seconds'))
                            row['records_per_second'] = payload['records'] / elapsed
                            row['chars_per_second'] = payload['chars'] / elapsed
                            row['valid_for_comparison'] = bool(code == 0 and payload.get('valid_for_comparison')
                                                               and payload['files'] == len(initial)
                                                               and input_signature(source, destination) == initial)
                            row['timing_scope'] = '完整独立子进程墙钟时间，含启动、导入、加载、读写、脱敏及报告'
                        except (ValueError, TypeError, KeyError):
                            row['valid_for_comparison'] = False
                            row['report_error'] = 'invalid_summary'
                    results.append(row)
                    write_reports(root, results, args)
                    if input_signature(source, destination) != initial:
                        raise RuntimeError('input_changed')
    except KeyboardInterrupt:
        aborted = '用户中断'
    except Exception as error:
        aborted = '输入集变化或评测执行异常（' + type(error).__name__ + '）'
    finally:
        write_reports(root, results, args, aborted)
    print(f'评测报告：{root / "算力评估.md"}', flush=True)
    return root, results, aborted


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        _, results, aborted = run_evaluation(args)
        if aborted == '用户中断':
            return 130
        return 0 if not aborted and results and all(r['valid_for_comparison'] for r in results) else 1
    except Exception as error:
        print(f'评测失败（{type(error).__name__}）；请检查输入目录和评测参数。')
        return 1
