"""统一流水线的契约测试；用假模型验证边界，运行时不加载 Paddle。"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from text_redaction.cli import build_parser, main, validate_args
from text_redaction.io import Latencies, batches, records, run_file, same_file
from text_redaction.pipeline import Pipeline


class FakeDetector:
    device = "cpu"

    def __init__(self, response=None):
        self.calls = []
        self.response = response

    def __call__(self, texts):
        self.calls.append(list(texts))
        if callable(self.response):
            return self.response(texts)
        return self.response if self.response is not None else [{} for _ in texts]


def arguments(**updates):
    args = build_parser().parse_args([])
    for key, value in updates.items():
        setattr(args, key, value)
    return args


def known_entities(texts):
    values = {"姓名": "陈嘉宁", "地址": "南京市栖霞区", "工作单位": "云杉科技公司"}
    results = []
    for text in texts:
        result = {}
        for kind, value in values.items():
            start = text.find(value)
            if start >= 0:
                result[kind] = [{"start": start, "end": start + len(value)}]
        results.append(result)
    return results


class PipelineTests(unittest.TestCase):
    def test_landline_and_known_values_are_masked_before_numeric_rules_can_interfere(self):
        engine = Pipeline(arguments(mode="rules"))
        self.assertEqual(engine.process(["联系010-12345678。"])[0], ["联系************。"])
        from text_redaction.rules import RuleRedactor
        engine.rules = RuleRedactor(known_values={"业务编号": ["TEST-010-12345678-END"]})
        self.assertEqual(engine.process(["编号TEST-010-12345678-END。"])[0], ["编号" + "*" * 21 + "。"])

    def test_rules_run_before_model_with_original_character_positions(self):
        detector = FakeDetector(known_entities)
        engine = Pipeline(arguments(), detector)
        source = "电话１３７４８２６０９１５，陈嘉宁住南京市栖霞区。\r\n"
        result, timings = engine.process([source])
        self.assertEqual(detector.calls, [["电话***********，陈嘉宁住南京市栖霞区。\r\n"]])
        self.assertEqual(result, ["电话***********，***住[地址已脱敏]。\r\n"])
        self.assertEqual(set(timings), {"regex_seconds", "model_seconds", "replace_seconds"})
        self.assertTrue(all(seconds >= 0 for seconds in timings.values()))

    def test_blank_records_are_preserved_and_excluded_from_model(self):
        detector = FakeDetector(known_entities)
        source = ["\n", " \t\r\n", "陈嘉宁\n", "", "\r", "陈嘉宁"]
        result, _ = Pipeline(arguments(), detector).process(source)
        self.assertEqual(detector.calls, [["陈嘉宁\n", "陈嘉宁"]])
        self.assertEqual(result, ["\n", " \t\r\n", "***\n", "", "\r", "***"])

    def test_entirely_blank_batch_does_not_call_detector(self):
        detector = FakeDetector(lambda _: self.fail("空白记录不应送入模型"))
        self.assertEqual(Pipeline(arguments(), detector).process(["", " \r\n"])[0], ["", " \r\n"])
        self.assertEqual(detector.calls, [])

    def test_name_default_full_and_optional_keep_first(self):
        self.assertEqual(arguments().name_mask, "full")
        for option, expected in [("full", "***"), ("keep-first", "陈**")]:
            with self.subTest(option=option):
                engine = Pipeline(arguments(name_mask=option), FakeDetector(known_entities))
                self.assertEqual(engine.process(["陈嘉宁"])[0], [expected])
        engine = Pipeline(arguments(name_mask="keep-first"), FakeDetector())
        self.assertEqual(engine.replace("陈", [{"type": "姓名", "start": 0, "end": 1}]), "*")

    def test_address_and_company_markers_preserve_context(self):
        source = "陈嘉宁住南京市栖霞区，在云杉科技公司工作。"
        engine = Pipeline(arguments(), FakeDetector(known_entities))
        self.assertEqual(engine.process([source])[0], ["***住[地址已脱敏]，在[单位已脱敏]工作。"])

    def test_overlap_merges_entire_sensitive_union(self):
        engine = Pipeline(arguments(name_mask="keep-first"), FakeDetector())
        entities = [
            {"type": "姓名", "start": 1, "end": 4},
            {"type": "地址", "start": 3, "end": 6},
            {"type": "工作单位", "start": 5, "end": 8},
        ]
        self.assertEqual(engine.replace("ABCDEFGHIJ", entities), "A*******IJ")

    def test_duplicate_and_adjacent_entities_are_not_over_merged(self):
        engine = Pipeline(arguments(), FakeDetector())
        address = {"type": "地址", "start": 0, "end": 3}
        entities = [address, dict(address), {"type": "工作单位", "start": 3, "end": 5}]
        self.assertEqual(engine.replace("ABCDE!", entities), "[地址已脱敏][单位已脱敏]!")

    def test_invalid_entity_coordinates_fail_closed(self):
        engine = Pipeline(arguments(), FakeDetector())
        for start, end in [(-1, 1), (0, 0), (2, 1), (0, 4), (True, 2), (0, False), (0., 1), ("0", 1), (None, 1)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                engine.replace("ABC", [{"type": "姓名", "start": start, "end": end}])
        with self.assertRaises(ValueError):
            engine.replace("ABC", [{"type": None, "start": 0, "end": 1}])

    def test_detector_output_count_and_shape_must_match(self):
        for response in [[], [{}, {}], {}, ({},), "invalid", [None]]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                Pipeline(arguments(), FakeDetector(response)).process(["普通文本"])

    def test_rules_mode_never_loads_model(self):
        with patch("text_redaction.model.UIEDetector", side_effect=AssertionError("不得加载模型")):
            engine = Pipeline(arguments(mode="rules"))
            self.assertEqual(engine.process(["电话13748260915\n"])[0], ["电话***********\n"])
            self.assertIsNone(engine.detector)


class FilePipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "input.txt"
        self.output = self.root / "output.txt"
        self.args = arguments(input=self.source, mode="rules", batch_records=2, batch_chars=100, max_record_chars=100)
        self.engine = Pipeline(self.args)

    def test_bom_accepted_and_mixed_newlines_preserved(self):
        self.source.write_bytes("\ufeff电话13748260915\r\n\n \t\r南京市\r最后一行".encode("utf-8"))
        stats = run_file(self.engine, self.args, self.output)
        self.assertEqual(self.output.read_bytes(), "电话***********\r\n\n \t\r南京市\r最后一行".encode("utf-8"))
        self.assertEqual(stats["records"], 5)
        self.assertEqual(stats["batches"], 3)
        self.assertEqual(stats["chars"], len(self.source.read_bytes().decode("utf-8-sig")))

    def test_exact_record_limit_includes_newline(self):
        self.source.write_bytes(b"abc\nxyz")
        self.assertEqual(list(records(self.source, 4)), ["abc\n", "xyz"])
        with self.assertRaises(ValueError):
            list(records(self.source, 3))

    def test_record_and_character_budgets(self):
        lines = ["ab\n", "cd\n", "wxyz\n", "z\n", "q"]
        self.source.write_text("".join(lines), encoding="utf-8")
        self.args.batch_records = 2
        self.args.batch_chars = self.args.max_record_chars = 7
        result = list(batches(self.source, self.args))
        self.assertEqual(result, [["ab\n", "cd\n"], ["wxyz\n", "z\n"], ["q"]])
        self.assertEqual([line for batch in result for line in batch], lines)
        self.assertTrue(all(len(batch) <= 2 and sum(map(len, batch)) <= 7 for batch in result))

    def test_invalid_budget_fails_before_processing(self):
        self.source.write_text("普通文本", encoding="utf-8")
        self.args.max_record_chars = self.args.batch_chars + 1
        with self.assertRaises(ValueError):
            validate_args(self.args)
        with self.assertRaises(ValueError):
            list(batches(self.source, self.args))

    def test_oversized_record_preserves_existing_output(self):
        self.source.write_text("secret-original-record", encoding="utf-8")
        self.output.write_text("previous-success", encoding="utf-8")
        self.args.max_record_chars = 5
        with self.assertRaises(ValueError) as error:
            run_file(self.engine, self.args, self.output)
        self.assertNotIn("secret-original-record", str(error.exception))
        self.assertEqual(self.output.read_text(), "previous-success")
        self.assertEqual(list(self.root.glob(".redact-*")), [])

    def test_partial_failure_preserves_existing_output_and_removes_temporary(self):
        self.source.write_text("第一条\n第二条\n", encoding="utf-8")
        self.output.write_text("previous-success", encoding="utf-8")
        self.args.batch_records = 1
        process = self.engine.process
        count = 0
        def fail_second(batch):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("sensitive-original-text")
            return process(batch)
        with patch.object(self.engine, "process", side_effect=fail_second), self.assertRaises(RuntimeError):
            run_file(self.engine, self.args, self.output)
        self.assertEqual(count, 2)
        self.assertEqual(self.output.read_text(), "previous-success")
        self.assertEqual(list(self.root.glob(".redact-*")), [])

    def test_wrong_pipeline_output_fails_without_publishing(self):
        self.source.write_text("第一条\n", encoding="utf-8")
        for result in [[], [123]]:
            with self.subTest(result=result):
                with patch.object(self.engine, "process", return_value=(result, {})), self.assertRaises(ValueError):
                    run_file(self.engine, self.args, self.output)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".redact-*")), [])

    def test_input_cannot_be_overwritten_via_path_hardlink_or_symlink(self):
        self.source.write_text("phone13748260915", encoding="utf-8")
        hardlink, symlink = self.root / "hardlink.txt", self.root / "symlink.txt"
        os.link(self.source, hardlink)
        symlink.symlink_to(self.source)
        for target in [self.source, hardlink, symlink]:
            with self.subTest(target=target.name):
                self.assertTrue(same_file(self.source, target))
                with self.assertRaises(ValueError):
                    run_file(self.engine, self.args, target)
        self.assertEqual(self.source.read_text(), "phone13748260915")
        self.assertTrue(symlink.is_symlink())
        self.assertEqual(list(self.root.glob(".redact-*")), [])

    def test_rule_configuration_cannot_be_overwritten(self):
        self.source.write_text("普通文本", encoding="utf-8")
        config = self.root / "rules.json"
        config.write_text("{}", encoding="utf-8")
        self.args.config = config
        with self.assertRaises(ValueError):
            run_file(self.engine, self.args, config)
        self.assertEqual(config.read_text(), "{}")

    def test_empty_input_publishes_empty_output_and_zero_stats(self):
        self.source.write_bytes(b"")
        self.output.write_text("previous", encoding="utf-8")
        with patch.object(self.engine, "process", side_effect=AssertionError("空文件不应处理")):
            stats = run_file(self.engine, self.args, self.output)
        self.assertEqual(self.output.read_bytes(), b"")
        for name in ["records", "chars", "batches", "max_record_chars", "records_per_second", "chars_per_second"]:
            self.assertEqual(stats[name], 0)
        self.assertEqual(stats["batch_latency_seconds"]["count"], 0)
        self.assertIsNone(stats["batch_latency_seconds"]["p95"])

    def test_cli_rules_mode_runs_from_another_working_directory(self):
        self.source.write_text("姓名陈嘉宁，电话13748260915\r\n", encoding="utf-8", newline="")
        project_root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=project_root)
        result = subprocess.run(
            [sys.executable, "-m", "text_redaction", "--mode", "rules", "-i", str(self.source), "-o", str(self.output)],
            cwd=self.root, env=env, text=True, capture_output=True, check=True,
        )
        self.assertEqual(self.output.read_bytes(), "姓名陈嘉宁，电话***********\r\n".encode("utf-8"))
        stats = json.loads(result.stdout)
        self.assertEqual(stats["records"], 1)
        self.assertEqual(stats["mode"], "rules")
        self.assertEqual(stats["device"], "cpu")
        self.assertNotIn("陈嘉宁", result.stdout + result.stderr)
        self.assertNotIn("13748260915", result.stdout + result.stderr)

    def test_cli_exception_logs_no_source_or_model_exception_detail(self):
        secret = "陈嘉宁的私密地址及电话13748260915"
        self.source.write_text(secret, encoding="utf-8")
        self.output.write_text("previous-success", encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["text_redaction", "--mode", "rules", "-i", str(self.source), "-o", str(self.output)]
        with patch("sys.argv", argv), patch("text_redaction.pipeline.Pipeline", return_value=self.engine):
            with patch.object(self.engine, "process", side_effect=RuntimeError(secret)):
                with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    main()
        self.assertEqual(error.exception.code, 1)
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn(secret, stdout.getvalue() + stderr.getvalue())
        self.assertNotIn("13748260915", stdout.getvalue() + stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(self.output.read_text(), "previous-success")
        self.assertEqual(list(self.root.glob(".redact-*")), [])


class LatencyTests(unittest.TestCase):
    def test_empty_and_small_workloads_have_exact_statistics(self):
        sample = Latencies(capacity=8)
        self.assertIsNone(sample.summary()["mean"])
        for value in [1., 2., 3., 4.]:
            sample.add(value)
        summary = sample.summary()
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertFalse(summary["approximate"])

    def test_large_workload_reservoir_is_bounded_and_reported_as_approximate(self):
        sample = Latencies(capacity=8)
        for value in range(10000):
            sample.add(float(value))
        self.assertEqual(len(sample.samples), 8)
        summary = sample.summary()
        self.assertEqual(summary["count"], 10000)
        self.assertEqual(summary["sample_count"], 8)
        self.assertEqual(summary["mean"], 4999.5)
        self.assertEqual(summary["max"], 9999.)
        self.assertTrue(summary["approximate"])
        self.assertLessEqual(summary["p50"], summary["p95"])
        self.assertLessEqual(summary["p95"], summary["max"])


if __name__ == "__main__":
    unittest.main()
