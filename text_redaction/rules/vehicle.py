"""车牌前缀库 + 后续字母数字匹配；采用宽松候选规则，不验证实际发牌。"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_PLATE_PREFIXES = Path(__file__).resolve().parent / "resources" / "plate_prefixes.json"

# =====================================================================
# 【修改位置 3：车牌规则】
# 前缀库位于 resources/plate_prefixes.json，可直接增加/删除“皖A”“浙B”等项。
# 默认库由 31 个省级车牌简称 × A-Z 生成，宽松覆盖，并非合法发牌前缀表。
# 后续字母数字至少 1 位，且会完整匹配连续串；不限制为标准牌照长度。
# 支持前缀后的间隔点/空格/横线、全角字符（由主模块规范化）和小写字母。
# 如需关闭，设置 ENABLE_VEHICLE_PLATE = False。
# =====================================================================
ENABLE_VEHICLE_PLATE = True
PLATE_BODY_MIN_LENGTH = 1
PLATE_SUFFIXES = "挂学警港澳使领"


@lru_cache(maxsize=8)
def load_plate_pattern(path: str, min_length: int, suffixes: str) -> re.Pattern[str]:
    prefixes = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("车牌前缀库必须是非空列表")
    if isinstance(min_length, bool) or not isinstance(min_length, int) or min_length < 1:
        raise ValueError("车牌后缀最小长度必须是正整数")
    grouped: dict[str, set[str]] = {}
    for prefix in prefixes:
        if not isinstance(prefix, str) or not re.fullmatch(r"[\u4e00-\u9fff][A-Z]", prefix):
            raise ValueError("车牌前缀需为一个汉字和一个大写字母")
        grouped.setdefault(prefix[0], set()).add(prefix[1])
    alternatives = [re.escape(province) + r"[ \t]{0,2}[" + "".join(sorted(letters)) + "]"
                    for province, letters in sorted(grouped.items())]
    suffix = "[" + re.escape(suffixes) + "]?" if suffixes else ""
    return re.compile(r"(?:" + "|".join(alternatives) + r")[ \t·.\-]{0,3}"
                      + r"[A-Z0-9]{" + str(min_length) + r",}" + suffix, re.I | re.ASCII)


def plate_pattern(path: str | Path | None = None) -> re.Pattern[str] | None:
    if not ENABLE_VEHICLE_PLATE:
        return None
    return load_plate_pattern(str(Path(path or DEFAULT_PLATE_PREFIXES).resolve()),
                              PLATE_BODY_MIN_LENGTH, PLATE_SUFFIXES)
