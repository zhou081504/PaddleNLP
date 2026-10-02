"""目录运行、独立进程矩阵及容量报告的回归。"""
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from text_redaction import batch
from benchmarks import evaluate


class FakeMonitor:
    def __init__(self, path, interval):
        self.path = path
    def start(self):
        self.path.write_text('phase\ninitialize\n', encoding='utf-8')
    def set_phase(self, *args):
        pass
    def stop(self):
        pass
    def report(self):
        return {'sampling_errors': {}, 'gpu_available': False}


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input'
        self.output = self.root / 'out'
        (self.source / 'nested').mkdir(parents=True)
        (self.source / 'a.txt').write_text('电话13800138000\n', encoding='utf-8')
        (self.source / 'nested/a.TXT').write_text('车牌沪A12345\n', encoding='utf-8')
        (self.source / 'ignored.csv').write_text('untouched', encoding='utf-8')

    def args(self, *extra):
        return batch.build_parser().parse_args([str(self.source), '--mode', 'rules', '--output-dir', str(self.output), *extra])

    def run_batch(self, *extra):
        with redirect_stdout(io.StringIO()), patch('benchmarks.monitor.ProcessResourceMonitor', FakeMonitor):
            return batch.run_directory(self.args(*extra))

    def test_recursive_processing_reuses_model_and_preserves_relative_paths(self):
        with patch.object(batch, 'Pipeline', wraps=batch.Pipeline) as factory:
            result = self.run_batch()
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(result['success'], 2)
        self.assertTrue(result['valid_for_comparison'])
        self.assertEqual((self.output / 'a.txt').read_text(), '电话***********\n')
        self.assertEqual((self.output / 'nested/a.TXT').read_text(), '车牌*******\n')
        self.assertTrue((self.output / '性能统计.json').is_file())
        self.assertTrue((self.output / '处理明细.csv').is_file())

    def test_existing_results_skip_without_loading_model_and_overwrite_is_explicit(self):
        self.run_batch()
        with patch.object(batch, 'Pipeline', side_effect=AssertionError('must not load')):
            result = self.run_batch()
        self.assertEqual(result['skipped'], 2)
        self.assertFalse(result['valid_for_comparison'])
        self.assertEqual(self.run_batch('--overwrite')['success'], 2)

    def test_failed_file_continues_and_old_output_is_preserved(self):
        (self.source / 'a.txt').write_text('x' * 50)
        (self.source / 'nested/a.TXT').write_text('ok\n')
        self.output.mkdir()
        (self.output / 'a.txt').write_text('old')
        result = self.run_batch('--max-record-chars', '10', '--batch-chars', '10', '--overwrite')
        self.assertEqual((result['failed'], result['success']), (1, 1))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['valid_for_comparison'])
        self.assertEqual((self.output / 'a.txt').read_text(), 'old')
        self.assertEqual((self.output / 'nested/a.TXT').read_text(), 'ok\n')

    def test_no_benchmark_disables_timers_monitor_and_performance_files(self):
        with patch('time.perf_counter', side_effect=AssertionError('timer should be off')), patch('benchmarks.monitor.ProcessResourceMonitor', side_effect=AssertionError('monitor should be off')):
            with redirect_stdout(io.StringIO()):
                result = batch.run_directory(self.args('--no-benchmark'))
        self.assertEqual(result['success'], 2)
        self.assertIsNone(result['wall_seconds'])
        self.assertFalse((self.output / '性能统计.json').exists())
        self.assertFalse((self.output / '资源采样.csv').exists())
        with (self.output / '处理明细.csv').open(encoding='utf-8-sig') as f:
            self.assertTrue(all(row['耗时_秒'] == '' for row in csv.DictReader(f)))

    def test_output_subdirectory_and_explicit_exclusions_not_reprocessed(self):
        generated = self.source / 'generated'
        generated.mkdir()
        (generated / 'stale.txt').write_text('ignored')
        self.output = self.source / 'results'
        self.run_batch('--exclude-dir', str(generated))
        result = self.run_batch('--exclude-dir', str(generated))
        self.assertEqual(result['files'], 2)
        self.assertEqual(result['skipped'], 2)

    def test_output_must_not_alias_any_input_or_parent(self):
        self.output.mkdir()
        os.link(self.source / 'nested/a.TXT', self.output / 'a.txt')
        with self.assertRaises(ValueError):
            self.run_batch('--overwrite')
        for output in (self.source, self.root):
            with self.assertRaises(ValueError):
                batch.discover_inputs(self.source, output)

    def test_model_failure_is_sanitized_and_reported(self):
        buffer = io.StringIO()
        with patch.object(batch, 'Pipeline', side_effect=RuntimeError('SECRET RAW CONTENT')):
            with redirect_stdout(buffer), patch('benchmarks.monitor.ProcessResourceMonitor', FakeMonitor):
                result = batch.run_directory(self.args())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['unprocessed'], 2)
        self.assertNotIn('SECRET', buffer.getvalue())
        self.assertNotIn('SECRET', (self.output / '性能统计.json').read_text())

    def test_output_directory_links_cannot_merge_two_results(self):
        (self.source / 'nested/a.TXT').rename(self.source / 'nested/a.txt')
        self.output.mkdir()
        (self.output / 'nested').symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_batch('--overwrite')

    def test_device_aliases_and_batch_alias(self):
        for value, device, number in [('cuda:2', 'gpu', 2), ('gpu:1', 'gpu', 1), ('auto', 'cpu', 0)]:
            args = self.args('--device', value, '--batch-size', '4')
            batch.resolve_device(args)
            self.assertEqual((args.device, args.device_id, args.model_batch_size), (device, number, 4))
        with self.assertRaises(ValueError):
            batch.resolve_device(self.args('--device', 'cuda:bad'))


class EvaluationTests(unittest.TestCase):
    def test_capacity_no_business_volume_does_not_invent_node_count(self):
        result = evaluate.estimate_capacity(10)
        self.assertIsNone(result['nodes_for_history_plus_daily'])
        self.assertEqual(result['records_per_node_per_day'], 576000)
        result = evaluate.estimate_capacity(10, history_records=1152000, deadline_days=2, daily_records=1000)
        self.assertEqual(result['nodes_for_history_plus_daily'], 2)
        with self.assertRaises(ValueError):
            evaluate.estimate_capacity(10, utilization=2)

    def test_ranking_requires_all_repeats_same_data_and_device(self):
        row = dict(requested_device='cpu', device='cpu', model_batch_size=8, files=2, records=10,
                   blank_records=0, chars=100, valid_for_comparison=True, records_per_second=2, chars_per_second=20)
        self.assertEqual(evaluate.rank_results([row], 2), [])
        self.assertEqual(evaluate.rank_results([row, dict(row, valid_for_comparison=False)], 2), [])
        self.assertEqual(evaluate.rank_results([row, dict(row, chars=101)], 2), [])
        ranked = evaluate.rank_results([row, dict(row, records_per_second=4)], 2)
        self.assertEqual(ranked[0]['median_records_per_second'], 3)

    def test_real_subprocess_matrix_excludes_prior_case_outputs_and_continues_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input'
            source.mkdir()
            (source / 'test.txt').write_text('电话13800138000\n', encoding='utf-8')
            args = evaluate.build_parser().parse_args([str(source), '--output-dir', str(source / 'evaluations'),
                                                      '--mode', 'rules', '--devices', 'invalid', 'cpu',
                                                      '--batch-sizes', '1', '--repeats', '2'])
            with redirect_stdout(io.StringIO()):
                root, results, aborted = evaluate.run_evaluation(args)
            self.assertIsNone(aborted)
            self.assertEqual(len(results), 4)
            self.assertTrue(all(not r['valid_for_comparison'] for r in results[:2]))
            self.assertTrue(all(r['valid_for_comparison'] and r['files'] == 1 for r in results[2:]))
            summary = json.loads((root / '评测汇总.json').read_text())
            self.assertEqual(len(summary['ranked_configurations']), 1)
            self.assertTrue((root / '算力评估.md').is_file())
            self.assertTrue((root / '评测对比.csv').is_file())
            self.assertNotIn('13800138000', (root / '评测汇总.json').read_text())


if __name__ == '__main__':
    unittest.main()
