import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from text_redaction.rules import RuleRedactor as Desensitizer


class RedactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Desensitizer()

    def test_address_and_names_unchanged(self):
        text = '我叫陈嘉宁，住南京市栖霞区云栖路86号6栋1203室。姓名：陈嘉宁；地址：北京市朝阳区。'
        self.assertEqual(self.engine.redact(text), text)
        obj = {'name': '陈嘉宁', 'address': '南京市栖霞区', '住址': {'city': '北京'}}
        self.assertEqual(self.engine.redact_json(obj), obj)

    def test_digits_inside_address_still_masked(self):
        self.assertEqual(self.engine.redact('地址：南京市云栖路1234567号'), '地址：南京市云栖路*******号')

    def test_unlabeled_numbers(self):
        self.assertEqual(self.engine.redact('拨打+86 137-4826-0915，证件320106199204183527。'), '拨打' + '*' * 17 + '，证件' + '*' * 18 + '。')

    def test_unicode_mapping(self):
        self.assertEqual(self.engine.redact('Ａ：① 联系１３７\u200b４８２６０９１５！'), 'Ａ：① 联系' + '*' * 12 + '！')

    def test_other_field_boundary(self):
        self.assertEqual(self.engine.redact('电话：13748260915，地址：南京市，姓名：陈嘉宁'),
                         '电话：***********，地址：南京市，姓名：陈嘉宁')

    def test_questions_unchanged(self):
        text = '请问手机号码是多少？地址是什么？电话是您本人的吗？'
        self.assertEqual(self.engine.redact(text), text)

    def test_json_nested(self):
        obj = {'phone': 13748260915, 'rows': [{'姓名': '陈嘉宁', '地址': {'city': '南京'}}], 'count': 2}
        self.assertEqual(self.engine.redact_json(obj),
                         {'phone': '***********', 'rows': [{'姓名': '陈嘉宁', '地址': {'city': '南京'}}], 'count': 2})

    def test_custom_and_known(self):
        engine = Desensitizer({'客户编号': ['客户编码']}, {'业务编号': ['TEST-ABC']})
        self.assertEqual(engine.redact('TEST-ABC；客户编码：ABC123'), '********；客户编码：******')

    def test_empty_json_values(self):
        self.assertEqual(self.engine.redact_json({'phone': '', 'bank_card': None}), {'phone': '', 'bank_card': None})

    def test_plain_text_idempotence(self):
        result = self.engine.redact('电话：13748260915；车牌：皖A12345')
        self.assertEqual(result, '电话：***********；车牌：*******')
        self.assertEqual(self.engine.redact(result), result)

    def test_old_id_and_unicode_digits(self):
        self.assertEqual(self.engine.redact('号码110105491231002 ١٣٨٠٠١٣٨٠٠٠'), '号码' + '*' * 15 + ' ' + '*' * 11)

    def test_no_cross_line_numbers(self):
        self.assertEqual(self.engine.redact('13748\n260915'), '13748\n260915')

    def test_find_keeps_categories(self):
        hits = self.engine.find('皖A12345，6222021234567890，1234567')
        self.assertEqual([hit.category for hit in hits], ['车牌号', '银行卡号', '连续数字'])

    def test_packaged_resources_from_another_directory(self):
        project_root = str(Path(__file__).resolve().parents[1])
        code = (
            "import json, sys; "
            f"sys.path.insert(0, {project_root!r}); "
            "from text_redaction.rules import RuleRedactor; "
            "print(json.dumps(RuleRedactor().redact_json(json.load(sys.stdin))))"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', code],
                                    input=json.dumps({'address': '南京市栖霞区', 'phone': '13748260915', 'plate': '皖A12345'}),
                                    cwd=directory, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout),
                         {'address': '南京市栖霞区', 'phone': '***********', 'plate': '*******'})


class NewRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Desensitizer()

    def test_plate_normal_and_new_energy_length(self):
        self.assertEqual(self.engine.redact('车是皖A12345，另一辆浙BD12345，停在楼下。'),
                         '车是*******，另一辆********，停在楼下。')

    def test_plate_lowercase_fullwidth_and_separators(self):
        self.assertEqual(self.engine.redact('皖ａ·１２３ｂ５、浙 B-D12345、粤Z1234港、沪A1234学'),
                         '********、**********、*******、*******')

    def test_plate_masks_entire_suffix(self):
        self.assertEqual(self.engine.redact('皖A12ABC34567890，稍后到。'), '*' * 15 + '，稍后到。')

    def test_plate_prefix_alone_and_unknown_prefix(self):
        self.assertEqual(self.engine.redact('皖A和浙B只是前缀，甲A12345不是库内前缀。'),
                         '皖A和浙B只是前缀，甲A12345不是库内前缀。')

    def test_custom_plate_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prefixes.json'
            path.write_text('["皖A"]', encoding='utf-8')
            engine = Desensitizer(plate_prefixes_file=path)
            self.assertEqual(engine.redact('皖A12345，浙B12345'), '*******，浙B12345')

    def test_bank_card_contiguous(self):
        self.assertEqual(self.engine.redact('用6222021234567890，还有6222021234567890123。'),
                         '用' + '*' * 16 + '，还有' + '*' * 19 + '。')

    def test_bank_card_grouped(self):
        self.assertEqual(self.engine.redact('用6222 0212 3456 7890或6222-0212-3456-7890-123。'),
                         '用' + '*' * 19 + '或' + '*' * 23 + '。')

    def test_bank_card_fullwidth(self):
        self.assertEqual(self.engine.redact('６２２２　０２１２　３４５６　７８９０'), '*' * 19)

    def test_bank_field_resolves_18_digit_ambiguity(self):
        self.assertEqual(self.engine.redact('银行卡号：622202123456789012；车牌号：皖A12345'),
                         '银行卡号：' + '*' * 18 + '；车牌号：*******')

    def test_bank_and_phone_keep_specific_labels(self):
        self.assertEqual(self.engine.redact('13748260915，6222021234567890，皖A1234567，320106199204183527'),
                         '*' * 11 + '，' + '*' * 16 + '，' + '*' * 9 + '，' + '*' * 18)

    def test_digit_threshold_boundaries(self):
        self.assertEqual(self.engine.redact('编号123456，1234567，12345678，0000000。'),
                         '编号123456，*******，********，*******。')

    def test_complete_digit_run_and_alphanumeric_id(self):
        self.assertEqual(self.engine.redact('订单ORD202609210001，流水123456789012345678901234567890。'),
                         '订单ORD' + '*' * 12 + '，流水' + '*' * 30 + '。')

    def test_continuous_does_not_join_separators(self):
        text = '1234 5678、1234-5678、1234\n5678、123456.7'
        self.assertEqual(self.engine.redact(text), text)

    def test_normalized_numeric_run(self):
        self.assertEqual(self.engine.redact('１２３\u200b４５６７；١٢٣٤٥٦٧'),
                         '********；*******')

    def test_custom_threshold(self):
        engine = Desensitizer(min_continuous_digits=8)
        self.assertEqual(engine.redact('1234567，12345678'), '1234567，********')

    def test_disable_continuous_keeps_other_rules(self):
        engine = Desensitizer(enable_continuous_digits=False)
        self.assertEqual(engine.redact('1234567，13748260915，皖A12345，6222021234567890'),
                         '1234567，' + '*' * 11 + '，' + '*' * 7 + '，' + '*' * 16)

    def test_invalid_thresholds(self):
        for threshold in [0, -1, True, 7.5, '7']:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                Desensitizer(min_continuous_digits=threshold)

    def test_json_integer_rules(self):
        obj = {'name': '陈嘉宁', 'serial': 1234567, 'card_number': 6222021234567890,
               'other': 6222021234567890, 'plate': '皖A12345', 'count': 6, 'ok': True}
        expected = {'name': '陈嘉宁', 'serial': '*******', 'card_number': '*' * 16,
                    'other': '*' * 16, 'plate': '*******', 'count': 6, 'ok': True}
        self.assertEqual(self.engine.redact_json(obj), expected)

    def test_all_rules_idempotent(self):
        text = '陈嘉宁住在南京市栖霞区云栖路86号。皖A1234567、6222 0212 3456 7890、12345678。'
        result = self.engine.redact(text)
        self.assertEqual(self.engine.redact(result), result)

    def test_configured_threshold_and_disable(self):
        for options, expected in [({'min_continuous_digits': 8}, '1234567，********'),
                                   ({'enable_continuous_digits': False}, '1234567，12345678')]:
            with self.subTest(options=options):
                self.assertEqual(Desensitizer(**options).redact('1234567，12345678'), expected)


class EqualLengthTests(unittest.TestCase):
    def test_chinese_custom_field(self):
        engine = Desensitizer(fields={"备注": ["敏感备注"]})
        self.assertEqual(engine.redact("敏感备注：测试汉字"), "敏感备注：****")

    def test_original_not_normalized_length(self):
        engine = Desensitizer(known_values={"自定义": ["ffi"]})
        self.assertEqual(engine.redact("原文ﬃ结束"), "原文*结束")

    def test_original_length_and_untouched_context(self):
        text = "我住南京，车牌皖A12345，电话１３７４８２６０９１５。"
        result = Desensitizer().redact(text)
        self.assertEqual(result, "我住南京，车牌*******，电话***********。")
        self.assertEqual(len(result), len(text))

    def test_sensitive_json_lengths(self):
        engine = Desensitizer()
        obj = {"phone": "137-4826-0915", "card_number": 6222021234567890, "plate": "皖A12345"}
        expected = {"phone": "*" * 13, "card_number": "*" * 16, "plate": "*" * 7}
        self.assertEqual(engine.redact_json(obj), expected)
        self.assertEqual(engine.redact_json(expected), expected)

    def test_sensitive_json_container(self):
        self.assertEqual(Desensitizer().redact_json({"phone": ["12", "345"]}), {"phone": "*" * 12})


if __name__ == '__main__':
    unittest.main()
