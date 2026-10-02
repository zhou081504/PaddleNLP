"""结构化敏感信息规则：号码、车牌、字段上下文与已知值。"""
from .core import DEFAULT_FIELDS, REPLACEMENT, Hit, RuleRedactor, normalize

# 保留原项目的类名，便于现有调用代码迁移。
Desensitizer = RuleRedactor

__all__ = ["RuleRedactor", "Desensitizer", "Hit", "normalize", "DEFAULT_FIELDS", "REPLACEMENT"]
