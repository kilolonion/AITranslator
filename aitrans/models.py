"""数据模型定义。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .constants import LANG_CODE_MAP


@dataclass
class AITranslated:
    """表示翻译结果的简单数据结构。"""

    src: str
    dest: str
    origin: str
    text: str
    pronunciation: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"<AITranslated src={self.src!r} dest={self.dest!r} "
            f"text={self.text!r} pronunciation={self.pronunciation!r} "
            f"metadata={self.metadata!r}>"
        )


@dataclass
class AIDetected:
    """语言检测结果。"""

    lang: str
    confidence: float
    details: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lang = LANG_CODE_MAP.get(self.lang.lower(), self.lang.lower())

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        details = f" details={self.details!r}" if self.details else ""
        return f"<AIDetected lang={self.lang!r} confidence={self.confidence:.3f}{details}>"


@dataclass
class TranslationRecoveryState:
    """跟踪翻译重试过程。"""

    attempts: int = 0
    last_error: Optional[Exception] = None
    partial_results: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_attempt(self, error: Exception) -> None:
        self.attempts += 1
        self.last_error = error

    @property
    def duration(self) -> float:
        return time.time() - self.start_time


__all__ = [
    "AITranslated",
    "AIDetected",
    "TranslationRecoveryState",
]
