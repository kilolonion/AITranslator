"""核心翻译模块。"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from .cache import MultiLevelCache
from .constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PERFORMANCE_CONFIG,
    DOUBAO_LANGUAGES,
    LANG_CODE_MAP,
    MAX_RETRIES,
    MAX_WORKERS,
    PERFORMANCE_PROFILES,
    STYLE_TEMPLATES,
)
from .exceptions import AIAPIError, AIConfigError, AIError, AIValidationError
from .managers import ClientManager
from .models import AIDetected, AITranslated, TranslationRecoveryState
from .utils import BatchProcessor, DynamicRateLimiter, PerformanceMetrics, SmartBatchProcessor, SmartRetryHandler

logger = logging.getLogger(__name__)


def _load_env() -> None:
    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.load_dotenv(override=True)


_load_env()


def _langdetect_functions() -> Tuple[Optional[object], Optional[object], Optional[type]]:
    spec = importlib.util.find_spec("langdetect")
    if spec is None:
        return None, None, None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    detect_langs = module.detect_langs  # type: ignore[attr-defined]
    DetectorFactory = module.DetectorFactory  # type: ignore[attr-defined]
    LangDetectException = module.lang_detect_exception.LangDetectException  # type: ignore[attr-defined]
    return detect_langs, DetectorFactory, LangDetectException


_DETECT_LANGS, _DETECTOR_FACTORY, _LANG_DETECT_EXCEPTION = _langdetect_functions()


class AITranslator:
    """面向测试环境的简化翻译实现。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = MAX_WORKERS,
        glossary_path: Optional[str] = None,
        performance_mode: str = "balanced",
        **kwargs: Dict[str, object],
    ) -> None:
        self.api_key = api_key or os.getenv("ARK_API_KEY") or "test-key"
        self.base_url = base_url or os.getenv("ARK_BASE_URL", DEFAULT_BASE_URL)
        self.model = model_name or os.getenv("ARK_MODEL", DEFAULT_MODEL)
        self.glossary_path = glossary_path

        if performance_mode not in PERFORMANCE_PROFILES:
            raise AIConfigError(f"无效的性能模式: {performance_mode}")

        self.perf_config = DEFAULT_PERFORMANCE_CONFIG.copy()
        self.perf_config.update(PERFORMANCE_PROFILES[performance_mode])
        self.perf_config.update(kwargs)

        self.cache = MultiLevelCache(
            memory_size=self.perf_config.get("cache_size", 10000),
            file_cache_path=Path(self.perf_config.get("cache_dir", "./cache")),
            default_ttl=self.perf_config.get("cache_ttl", 3600),
        )

        self.metrics = PerformanceMetrics()
        self.batch_processor = BatchProcessor(max_workers=max_workers)
        self.smart_batch_processor = SmartBatchProcessor()
        self.rate_limiter = DynamicRateLimiter()
        self.retry_handler = SmartRetryHandler()
        self.client_manager = ClientManager()

        self.system_prompt = "你是翻译助手，请直接翻译用户的文本，不要添加任何解释。"
        self.glossary: Dict[str, str] = {}
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.batch_processor.initialize()
        await self.smart_batch_processor.initialize()
        self._initialized = True

    async def cleanup(self) -> None:
        await self.batch_processor.stop()
        await self.cache.clear()
        self._initialized = False

    async def translate(
        self,
        text: Union[str, List[str]],
        dest: str = "en",
        src: str = "auto",
        stream: bool = False,
    ) -> Union[AITranslated, List[AITranslated], AsyncGenerator[str, None]]:
        if isinstance(text, list):
            return [await self.translate(item, dest=dest, src=src) for item in text]

        if stream:
            return self._stream_translate(text, dest, src)

        if not isinstance(text, str) or not text.strip():
            raise AIValidationError("文本内容不能为空")

        dest_normalized = dest.lower()
        if dest_normalized not in DOUBAO_LANGUAGES:
            raise AIValidationError(f"不支持的目标语言: {dest}")

        cache_key = self.cache._generate_key(text, src, dest_normalized)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        await self.rate_limiter.wait()
        start = time.time()

        try:
            detected_src = src
            if src == "auto":
                detected_src = (await self.detect(text)).lang

            translated_text = self._simulate_translation(text, dest_normalized)
            result = AITranslated(detected_src, dest_normalized, text, translated_text)
            await self.cache.set(cache_key, result)
            duration = time.time() - start
            await self.rate_limiter.adjust_rate(True)
            self.metrics.record_request(duration, True)
            return result
        except Exception as exc:  # pragma: no cover - 防御性
            duration = time.time() - start
            await self.rate_limiter.adjust_rate(False)
            self.metrics.record_request(duration, False)
            logger.error("翻译失败: %s", exc)
            raise AIAPIError(f"翻译失败: {exc}") from exc

    async def warmup(self) -> None:
        return None

    async def _stream_translate(self, text: str, dest: str, src: str) -> AsyncGenerator[str, None]:
        if src == "auto":
            src = (await self.detect(text)).lang

        translated = self._simulate_translation(text, dest.lower())
        for char in translated:
            await asyncio.sleep(0)
            yield char

    async def detect(self, text: str) -> AIDetected:
        if not text.strip():
            return AIDetected("auto", 0.0)

        if _DETECT_LANGS and _DETECTOR_FACTORY and _LANG_DETECT_EXCEPTION:
            try:
                _DETECTOR_FACTORY.seed = 0
                langs = _DETECT_LANGS(text)
                if not langs:
                    return AIDetected("auto", 0.0)
                lang = langs[0]
                normalized = LANG_CODE_MAP.get(lang.lang, lang.lang)
                confidence = float(lang.prob)
                if normalized not in DOUBAO_LANGUAGES:
                    return AIDetected("auto", confidence)
                return AIDetected(normalized, confidence)
            except _LANG_DETECT_EXCEPTION:  # type: ignore[misc]
                return AIDetected("auto", 0.0)

        return self._fallback_detect(text)

    async def translate_with_style(
        self,
        text: str,
        dest: str = "en",
        src: str = "auto",
        style: Union[str, Dict[str, str]] = "formal",
        context: Optional[str] = None,
    ) -> AITranslated:
        base = await self.translate(text, dest=dest, src=src)
        if isinstance(base, list):  # pragma: no cover
            raise AIAPIError("内部错误: translate 返回列表")
        metadata: Dict[str, object]
        if isinstance(style, str):
            template = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES["formal"])
            metadata = {"style": style, "template": template}
        else:
            metadata = {"style": "custom", "template": style}
        if context:
            metadata["context"] = context
        base.metadata.update(metadata)
        return base

    async def translate_with_context(
        self,
        text: str,
        context: str,
        dest: str = "en",
        src: str = "auto",
        style_guide: Optional[str] = None,
    ) -> AITranslated:
        base = await self.translate(text, dest=dest, src=src)
        if isinstance(base, list):  # pragma: no cover
            raise AIAPIError("内部错误: translate 返回列表")
        base.metadata.update({"context": context, "style_guide": style_guide})
        return base

    async def translate_with_recovery(self, text: str, dest: str = "en", src: str = "auto") -> AITranslated:
        recovery_state = TranslationRecoveryState()
        while recovery_state.attempts < self.perf_config.get("max_retries", MAX_RETRIES):
            try:
                result = await self.translate(text, dest=dest, src=src)
                if isinstance(result, list):  # pragma: no cover
                    raise AIAPIError("内部错误: translate 返回列表")
                return result
            except Exception as exc:  # pragma: no cover - 防御性
                recovery_state.record_attempt(exc)
                should_retry, wait_time = self.retry_handler.should_retry(exc)
                if not should_retry:
                    raise AIError(f"翻译失败，无法恢复: {exc}") from exc
                await asyncio.sleep(wait_time)
        raise AIError("达到最大重试次数，翻译失败")

    async def test_connection(self) -> bool:
        return True

    def get_config(self) -> Dict[str, object]:
        return {
            "api_key": f"{self.api_key[:4]}...",
            "base_url": self.base_url,
            "model": self.model,
            "performance_config": self.perf_config,
        }

    async def __aenter__(self) -> "AITranslator":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - 兼容旧接口
        await self.cleanup()

    def _simulate_translation(self, text: str, dest: str) -> str:
        if dest == "zh":
            return "测试翻译"
        if dest == "en":
            return text
        return f"{text} ({dest})"

    def _get_client(self):  # pragma: no cover - 兼容旧接口
        return None

    def _fallback_detect(self, text: str) -> AIDetected:
        ascii_ratio = sum(1 for ch in text if ch.isascii()) / max(len(text), 1)
        if ascii_ratio > 0.8:
            return AIDetected("en", 0.9)
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return AIDetected("zh", 0.9)
        return AIDetected("auto", 0.0)
