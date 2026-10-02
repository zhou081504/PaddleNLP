"""银行卡号候选和连续数字兜底规则。传入文本应已完成 normalize()。"""
from __future__ import annotations

import re
from typing import Iterator

# =====================================================================
# 【修改位置 1：连续数字脱敏开关和阈值】
# 7 表示连续 7 位及以上全部替换；改为 8 就是连续 8 位及以上。
# 规则没有最大长度上限，会完整遮盖整段数字，不只遮盖前 7 位。
# 空格、横线、换行会中断连续数字；规范化阶段去除的零宽字符不会中断。
# 设为 False 可关闭本模块的连续数字兜底，其他规则仍然生效。
# =====================================================================
ENABLE_CONTINUOUS_DIGITS = True
MIN_CONTINUOUS_DIGITS = 7

# =====================================================================
# 【修改位置 2：银行卡号候选规则】
# 这是可调整的候选长度范围，不是银行卡真实性验证。
# 默认不使用 Luhn 校验，避免错误记录的卡号漏脱敏。
# 支持完整连续数字，以及每组 4 位、末组 1～4 位的空格/横线分组。
# 更短的卡号可以通过“银行卡号”等明确字段处理，或修改下方最小长度。
# =====================================================================
ENABLE_BANK_CARD = True
BANK_CARD_MIN_DIGITS = 16
BANK_CARD_MAX_DIGITS = 19


class NumericMatcher:
    def __init__(self, min_continuous_digits: int | None = None,
                 enable_continuous_digits: bool | None = None):
        threshold = MIN_CONTINUOUS_DIGITS if min_continuous_digits is None else min_continuous_digits
        enabled = ENABLE_CONTINUOUS_DIGITS if enable_continuous_digits is None else enable_continuous_digits
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ValueError("连续数字阈值必须是正整数")
        if not isinstance(enabled, bool):
            raise ValueError("连续数字开关必须是布尔值")
        if not 1 <= BANK_CARD_MIN_DIGITS <= BANK_CARD_MAX_DIGITS:
            raise ValueError("银行卡号长度配置不正确")
        self.continuous_re = re.compile(r"[0-9]{" + str(threshold) + r",}") if enabled else None
        # 匹配整段数字后再判断长度，避免从过长号码中截取一段当银行卡。
        self.bank_re = re.compile(
            r"(?<![A-Za-z0-9_])(?:[0-9]{4}(?:[ \t-][0-9]{4}){2,}"
            r"[ \t-][0-9]{1,4}|[0-9]+)(?![A-Za-z0-9_])"
        ) if ENABLE_BANK_CARD else None

    def find(self, text: str) -> Iterator[tuple[int, int, str, str]]:
        if self.bank_re is not None:
            for match in self.bank_re.finditer(text):
                digits = re.sub(r"[^0-9]", "", match.group())
                if BANK_CARD_MIN_DIGITS <= len(digits) <= BANK_CARD_MAX_DIGITS:
                    yield *match.span(), "银行卡号", "bank_card"
        if self.continuous_re is not None:
            for match in self.continuous_re.finditer(text):
                yield *match.span(), "连续数字", "continuous_digits"
