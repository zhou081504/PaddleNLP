"""规则预处理 → 模型识别 → 合并重叠范围并遮盖。"""
from __future__ import annotations

import json
import re
import time

from .rules import RuleRedactor


def load_rule_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path else {}
    allowed = {"fields", "known_values", "min_continuous_digits", "enable_continuous_digits", "plate_prefixes_file"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("规则配置包含不支持的字段")
    for name in ("fields", "known_values"):
        values = config.get(name)
        if values is not None and (
            not isinstance(values, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, list)
                or any(not isinstance(item, str) or not item for item in v)
                for k, v in values.items()
            )
        ):
            raise ValueError("规则字段和已知值必须是字符串列表映射")
    if "enable_continuous_digits" in config and not isinstance(config["enable_continuous_digits"], bool):
        raise ValueError("enable_continuous_digits 必须为布尔值")
    if config.get("plate_prefixes_file") is not None:
        from pathlib import Path
        candidate = Path(config["plate_prefixes_file"])
        config["plate_prefixes_file"] = candidate if candidate.is_absolute() else path.parent / candidate
    return config


class Pipeline:
    def __init__(self, args, detector=None):
        self.measure = not getattr(args, "no_benchmark", False)
        self.rules = RuleRedactor(**load_rule_config(args.config))
        self.name_mask = args.name_mask
        # 兼容原 PIIDetector 的补充号码规则；在模型前全部遮盖。
        self.patterns = [re.compile(pattern) for pattern in (
            r"(?<!\d)1[3-9]\d{9}(?!\d)",
            r"(?<!\d)\d{17}[\dXx](?!\d)",
            r"(?<!\d)\d{16,19}(?!\d)",
            r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)",
        )]
        self.detector = None
        self.device = "cpu"
        if args.mode == "hybrid":
            if detector is None:
                from .model import UIEDetector
                detector = UIEDetector(args)
            self.detector = detector
            self.device = getattr(detector, "device", args.device)

    def replace(self, text, entities):
        spans = []
        for entity in entities:
            start, end, kind = entity["start"], entity["end"], entity["type"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("模型返回了无效实体坐标")
            if not isinstance(kind, str):
                raise ValueError("模型返回了无效实体类型")
            spans.append((start, end, kind))
        groups = []
        for start, end, kind in sorted(set(spans)):
            if groups and start < groups[-1][1]:
                old_start, old_end, _ = groups.pop()
                groups.append((old_start, max(old_end, end), None))
            else:
                groups.append((start, end, kind))
        parts, previous = [], 0
        for start, end, kind in groups:
            parts.append(text[previous:start])
            if kind == "地址":
                masked = "[地址已脱敏]"
            elif kind == "工作单位":
                masked = "[单位已脱敏]"
            elif kind == "姓名" and self.name_mask == "keep-first" and end - start > 1:
                masked = text[start] + "*" * (end - start - 1)
            else:
                masked = "*" * (end - start)
            parts.append(masked)
            previous = end
        parts.append(text[previous:])
        return "".join(parts)

    def process(self, lines):
        clock = time.perf_counter if self.measure else lambda: 0.
        started = clock()
        prepared = []
        for line in lines:
            # 所有规则在同一份原文上识别，避免连续数字遮盖打断座机等更长命中。
            spans = [(hit.start, hit.end) for hit in self.rules.find(line)]
            for pattern in self.patterns:
                spans.extend(match.span() for match in pattern.finditer(line))
            merged = []
            for start, end in sorted(spans):
                if merged and start < merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            parts, previous = [], 0
            for start, end in merged:
                parts.extend((line[previous:start], "*" * (end - start)))
                previous = end
            parts.append(line[previous:])
            prepared.append("".join(parts))
        timings = {"regex_seconds": clock() - started, "model_seconds": 0., "replace_seconds": 0.}
        active = [i for i, text in enumerate(prepared) if text.strip()]
        if self.detector is None or not active:
            return prepared, timings
        started = clock()
        results = self.detector([prepared[i] for i in active])
        timings["model_seconds"] = clock() - started
        if not isinstance(results, list) or len(results) != len(active):
            raise ValueError("模型输出条数与输入不一致")
        started = clock()
        for index, result in zip(active, results):
            if not isinstance(result, dict):
                raise ValueError("模型输出格式错误")
            entities = [dict(type=kind, start=item["start"], end=item["end"])
                        for kind, items in result.items() for item in items]
            prepared[index] = self.replace(prepared[index], entities)
        timings["replace_seconds"] = clock() - started
        return prepared, timings
