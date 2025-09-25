"""同步封装。"""

from __future__ import annotations

import asyncio
from typing import Generator, List, Optional, Union

from .core import AITranslator
from .exceptions import AIError, AIValidationError
from .models import AITranslated


class AITranslatorSync:
    """AITranslator 的同步适配器。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = 5,
        glossary_path: Optional[str] = None,
        performance_mode: str = "balanced",
        auto_preconnect: bool = True,
        **kwargs,
    ) -> None:
        self.translator = AITranslator(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            max_workers=max_workers,
            glossary_path=glossary_path,
            performance_mode=performance_mode,
            **kwargs,
        )
        self.auto_preconnect = auto_preconnect
        self._initialized = False
        self._loop = asyncio.new_event_loop()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.translator.initialize())
        self._initialized = True
        if self.auto_preconnect:
            self.warmup()

    def translate(
        self,
        text: Union[str, List[str]],
        dest: str = "en",
        src: str = "auto",
        stream: bool = False,
        max_retries: int = 3,
    ) -> Union[AITranslated, List[AITranslated], Generator[AITranslated, None, None]]:
        self._ensure_initialized()

        if isinstance(text, list):
            return self.translate_batch_sync(text, dest=dest, src=src, max_retries=max_retries)

        if stream:
            return self._create_stream_generator(text, dest, src)

        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < max_retries:
            try:
                return self._loop.run_until_complete(self.translator.translate(text, dest=dest, src=src))
            except (AIError, AIValidationError) as exc:
                last_error = exc
                attempt += 1
        raise last_error or AIError("翻译失败")

    def _create_stream_generator(self, text: str, dest: str, src: str) -> Generator[AITranslated, None, None]:
        self._ensure_initialized()
        async_gen = self._loop.run_until_complete(
            self.translator.translate(text, dest=dest, src=src, stream=True)
        )

        detected_src = src
        if src == "auto":
            detected_src = self._loop.run_until_complete(self.translator.detect(text)).lang

        while True:
            try:
                chunk = self._loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
            yield AITranslated(detected_src, dest, text, chunk)

    def translate_batch_sync(
        self,
        texts: List[str],
        dest: str = "en",
        src: str = "auto",
        max_retries: int = 3,
    ) -> List[AITranslated]:
        self._ensure_initialized()
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < max_retries:
            try:
                return self._loop.run_until_complete(self.translator.translate(texts, dest=dest, src=src))
            except (AIError, AIValidationError) as exc:
                last_error = exc
                attempt += 1
        raise last_error or AIError("批量翻译失败")

    def translate_with_style(
        self,
        text: str,
        dest: str = "en",
        src: str = "auto",
        style: Union[str, dict] = "formal",
        context: Optional[str] = None,
    ) -> AITranslated:
        self._ensure_initialized()
        return self._loop.run_until_complete(
            self.translator.translate_with_style(text, dest=dest, src=src, style=style, context=context)
        )

    def translate_with_context(
        self,
        text: str,
        context: str,
        dest: str = "en",
        src: str = "auto",
    ) -> AITranslated:
        self._ensure_initialized()
        return self._loop.run_until_complete(
            self.translator.translate_with_context(text, context, dest=dest, src=src)
        )

    def warmup(self) -> None:
        self._ensure_initialized()
        self._loop.run_until_complete(self.translator.warmup())

    def test_connection(self) -> bool:
        self._ensure_initialized()
        return self._loop.run_until_complete(self.translator.test_connection())

    def get_config(self) -> dict:
        self._ensure_initialized()
        return self.translator.get_config()

    def get_metrics(self) -> dict:
        self._ensure_initialized()
        return self.translator.metrics.get_metrics()

    def set_performance_config(self, **kwargs) -> None:
        self.translator.perf_config.update(kwargs)

    def __enter__(self) -> "AITranslatorSync":
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._initialized:
            return
        try:
            self._loop.run_until_complete(self.translator.cleanup())
        finally:
            self._loop.close()
            self._initialized = False

    def __del__(self):  # pragma: no cover - 清理保障
        self.close()

    @staticmethod
    def quick_translate(text: str, dest: str = "en", src: str = "auto") -> str:
        with AITranslatorSync(auto_preconnect=False) as translator:
            result = translator.translate(text, dest=dest, src=src)
            if isinstance(result, list):  # pragma: no cover
                raise AIError("quick_translate 仅支持单个文本")
            return result.text


def create(*args, **kwargs) -> AITranslatorSync:
    return AITranslatorSync(*args, **kwargs)
