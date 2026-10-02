#!/usr/bin/env python3
"""标准库文本脱敏：规范化识别、原文坐标替换、字段上下文和已知值匹配。"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .numeric import NumericMatcher
from .vehicle import plate_pattern

# 【修改位置 5：统一替换内容】命中原文每个字符替换成一个星号，保持字符数一致。
REPLACEMENT = "*"

DEFAULT_FIELDS = {
    "手机号": ["手机号", "手机号码", "手机", "联系电话", "联系方式", "电话", "mobile", "phone"],
    "身份证号": ["身份证号", "身份证号码", "身份证", "证件号码", "证件号", "id_card", "idcard"],
    "车牌号": ["车牌", "车牌号", "车牌号码", "车辆号牌", "plate", "license_plate"],
    "银行卡号": ["银行卡号", "银行卡", "银行账号", "卡号", "bank_card", "bank_account", "card_number"],
}
# 仅允许有限的常见分隔符，避免跨行拼接不同号码。
SEP = r"[ \t\-·]{0,3}"
PHONE_RE = re.compile(r"(?<![A-Za-z0-9_])(?:\+?86[ \t\-]{0,3})?1[3-9](?:" + SEP + r"[0-9]){9}(?![A-Za-z0-9_])")
ID_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[1-9](?:" + SEP + r"[0-9]){16}" + SEP + r"[0-9Xx]|[1-9](?:" + SEP + r"[0-9]){14})(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class Hit:
    start: int
    end: int
    category: str
    source: str


def normalize(text: str) -> tuple[str, list[tuple[int, int]]]:
    """逐字符 NFKC + 十进制数字规范化；保留每个规范化字符的原文范围。"""
    chars, offsets = [], []
    for i, original in enumerate(text):
        for ch in unicodedata.normalize("NFKC", original):
            if unicodedata.category(ch) == "Cf":  # 零宽字符、方向控制符等
                continue
            if ch.isdecimal():
                ch = str(unicodedata.decimal(ch))
            if ch in "‐‑‒–—−":
                ch = "-"
            if ch in "\u2028\u2029":
                ch = "\n"
            chars.append(ch)
            offsets.append((i, i + 1))
    return "".join(chars), offsets


class RuleRedactor:
    def __init__(self, fields: dict[str, list[str]] | None = None,
                 known_values: dict[str, list[str]] | None = None,
                 min_continuous_digits: int | None = None,
                 enable_continuous_digits: bool | None = None,
                 plate_prefixes_file: str | Path | None = None):
        """fields 添加或扩展字段别名；known_values 用于无字段标签的精确识别。"""
        self.numeric_matcher = NumericMatcher(min_continuous_digits, enable_continuous_digits)
        self.plate_re = plate_pattern(plate_prefixes_file)
        merged = {k: list(v) for k, v in DEFAULT_FIELDS.items()}
        for category, aliases in (fields or {}).items():
            merged.setdefault(category, []).extend(aliases)
        self.aliases: dict[str, str] = {}
        for category, aliases in merged.items():
            for alias in aliases:
                key = normalize(alias)[0].casefold()
                if not key:
                    raise ValueError("字段别名不能为空")
                self.aliases[key] = category
        keys = "|".join(re.escape(k) for k in sorted(self.aliases, key=len, reverse=True))
        self.field_re = re.compile(
            r"(?<![A-Za-z0-9_])(?P<quote>[\"']?)(?P<key>" + keys +
            r")(?P=quote)[ \t]*(?::|=)[ \t]*", re.I)
        # 字段值止于下一个有分隔符的键值字段，含未配置的普通字段。
        self.next_key_re = re.compile(
            r"[,，;；|\t ]+[\"']?[\w\u4e00-\u9fff]{1,32}[\"']?[ \t]*[:=]")
        self.known: list[tuple[str, re.Pattern[str]]] = []
        for category, values in (known_values or {}).items():
            for value in values:
                normalized = normalize(value)[0]
                if not normalized:
                    raise ValueError("已知值不能为空")
                self.known.append((category, re.compile(re.escape(normalized))))

    def find(self, text: str) -> list[Hit]:
        normalized, offsets = normalize(text)
        hits: list[Hit] = []

        def add(start: int, end: int, category: str, source: str) -> None:
            if start < end:
                hits.append(Hit(offsets[start][0], offsets[end - 1][1], category, source))

        fields = list(self.field_re.finditer(normalized))
        for index, match in enumerate(fields):
            start = match.end()
            end = fields[index + 1].start() if index + 1 < len(fields) else len(normalized)
            line = re.search(r"[\r\n;；|]", normalized[start:end])
            if line:
                end = start + line.start()
            next_key = self.next_key_re.search(normalized, start, end)
            if next_key:
                end = next_key.start()
            # 引号包裹的值只替换引号内部，支持反斜杠转义。
            if start < end and normalized[start] in "\"'":
                quote = normalized[start]
                quoted = re.match(r"(?:\\.|[^\\" + quote + r"])*" + quote,
                                  normalized[start + 1:end])
                if quoted:
                    start += 1
                    end = start + quoted.end() - 1
            while end > start and normalized[end - 1] in " \t,，。":
                end -= 1
            add(start, end, self.aliases[match.group("key").casefold()], "field")

        for category, pattern in [("身份证号", ID_RE), ("手机号", PHONE_RE)]:
            for match in pattern.finditer(normalized):
                add(*match.span(), category, "format")
        if self.plate_re is not None:
            for match in self.plate_re.finditer(normalized):
                add(*match.span(), "车牌号", "vehicle_plate")
        for start, end, category, source in self.numeric_matcher.find(normalized):
            add(start, end, category, source)
        for category, pattern in self.known:
            for match in pattern.finditer(normalized):
                add(*match.span(), category, "known")

        # 合并重叠范围，避免手机号和字段等规则重复替换、错位或留下片段。
        result: list[Hit] = []
        # 【修改位置 4：重叠规则的标签优先级】数值越小，标签优先级越高。
        # 合并后的范围覆盖所有重叠命中；连续数字兜底不会盖掉更明确的分类。
        priority = {"field": 0, "known": 1, "vehicle_plate": 2, "format": 3,
                    "bank_card": 4, "continuous_digits": 5}
        for hit in sorted(hits, key=lambda h: (h.start, priority[h.source], -h.end)):
            if result and hit.start < result[-1].end:
                prev = result.pop()
                winner = min((prev, hit), key=lambda h: priority[h.source])
                result.append(Hit(prev.start, max(prev.end, hit.end), winner.category, winner.source))
            else:
                result.append(hit)
        return result

    def redact(self, text: str) -> str:
        hits = self.find(text)
        if not hits:
            return text
        # 一次拼接，避免每次命中都复制整篇文本；区间已经排序并去重。
        parts: list[str] = []
        cursor = 0
        for hit in hits:
            parts.append(text[cursor:hit.start])
            parts.append(REPLACEMENT * (hit.end - hit.start))
            cursor = hit.end
        parts.append(text[cursor:])
        return "".join(parts)

    @staticmethod
    def mask_value(value: Any) -> Any:
        """字符串按原字符数；其他非空值按紧凑 JSON 表示的字符数遮盖。"""
        if value is None or value == "":
            return value
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return REPLACEMENT * len(text)

    def redact_json(self, value: Any) -> Any:
        """JSON 中的敏感键直接覆盖整个值，含数字、数组和对象；保留空值。"""
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                category = self.aliases.get(normalize(key)[0].casefold())
                result[key] = self.mask_value(item) \
                    if category else self.redact_json(item)
            return result
        if isinstance(value, list):
            return [self.redact_json(item) for item in value]
        if isinstance(value, str):
            return self.redact(value)
        # 未知 JSON 键下的整数也应用号码及连续数字规则；小整数和布尔值保留。
        if isinstance(value, int) and not isinstance(value, bool):
            text = str(value)
            redacted = self.redact(text)
            return redacted if redacted != text else value
        return value
