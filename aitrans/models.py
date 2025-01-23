"""数据模型模块"""

from dataclasses import dataclass
from typing import Dict, Optional
from .constants import LANG_CODE_MAP


@dataclass
class AITranslated:
    """表示翻译结果的类"""
    src: str
    dest: str
    origin: str
    text: str
    pronunciation: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        self.metadata = self.metadata or {}

    def __repr__(self):
        return f'<AITranslated src={self.src} dest={self.dest} text={self.text} pronunciation={self.pronunciation} metadata={self.metadata}>'


@dataclass
class AIDetected:
    """表示语言检测结果的类"""
    lang: str
    confidence: float
    details: Dict = None

    def __init__(self, lang: str, confidence: float, details: Dict = None):
        self.lang = self._normalize_lang_code(lang)
        self.confidence = confidence
        self.details = details or {}

    def _normalize_lang_code(self, lang: str) -> str:
        """标准化语言代码"""
        return LANG_CODE_MAP.get(lang.lower(), lang.lower())

    def __repr__(self):
        if self.details:
            return f'<AIDetected lang={self.lang} confidence={self.confidence:.3f} details={self.details}>'
        return f'<AIDetected lang={self.lang} confidence={self.confidence:.3f}>'


@dataclass
class TranslationProgress:
    """翻译进度信息"""
    completed: int
    total: int
    current_text: str
    status: str = 'translating'
    error: str = None

    @property
    def percent(self) -> float:
        """计算完成百分比"""
        return (self.completed / self.total) * 100 if self.total > 0 else 0


@dataclass
class TranslationRecoveryState:
    """管理翻译恢复状态"""
    attempts: int = 0
    last_error: Exception = None
    recovery_strategy: str = None
    partial_results: list = None
    start_time: float = None

    def __post_init__(self):
        import time
        self.start_time = time.time()
        self.partial_results = []

    def record_attempt(self, error: Exception):
        self.attempts += 1
        self.last_error = error

    def add_partial_result(self, result: AITranslated):
        self.partial_results.append(result)

    @property
    def duration(self) -> float:
        import time
        return time.time() - self.start_time
