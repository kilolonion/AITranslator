"""文档翻译模块"""

import asyncio
import logging
import time
from typing import List, Optional
from .models import AITranslated
from .utils import BatchProcessor

logger = logging.getLogger(__name__)


class DocumentTranslator:
    """文档翻译器"""

    def __init__(self, translator: 'AITranslator', context_window: int = 2, batch_size: int = 5):
        self.translator = translator
        self.context_window = context_window
        self.batch_size = batch_size
        self.processor = BatchProcessor(
            max_workers=translator.perf_config['max_workers'],
            batch_size=batch_size
        )
        self._paragraphs = []
        self._context_cache = {}

    def _get_context(self, index: int) -> str:
        """获取指定段落的上下文"""
        if index in self._context_cache:
            return self._context_cache[index]

        start_idx = max(0, index - self.context_window)
        end_idx = min(len(self._paragraphs), index + self.context_window + 1)
        context_paras = self._paragraphs[start_idx:index] + \
            self._paragraphs[index+1:end_idx]
        context = "\n".join(context_paras)

        self._context_cache[index] = context
        return context

    async def translate_document(self,
                                 paragraphs: List[str],
                                 dest: str = 'en',
                                 src: str = 'auto',
                                 style_guide: str = None) -> List[AITranslated]:
        """翻译整个文档

        Args:
            paragraphs: 段落列表
            dest: 目标语言
            src: 源语言
            style_guide: 风格指南
        """
        if not paragraphs:
            return []

        self._paragraphs = paragraphs
        self._context_cache.clear()
        start_time = time.time()

        async def translate_paragraph(text: str, index: int) -> AITranslated:
            try:
                context = self._get_context(index)
                return await self.translator.translate_with_context(
                    text=text,
                    context=context,
                    dest=dest,
                    src=src,
                    style_guide=style_guide
                )
            except Exception as e:
                logger.error(
                    f"Failed to translate paragraph {index}: {str(e)}")
                return AITranslated(
                    src=src,
                    dest=dest,
                    origin=text,
                    text=f"Translation failed: {str(e)}"
                )

        try:
            # 创建任务列表
            tasks = []
            for i, text in enumerate(paragraphs):
                tasks.append(translate_paragraph(text, i))

            # 分批执行任务
            results = []
            for i in range(0, len(tasks), self.batch_size):
                batch = tasks[i:i + self.batch_size]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)

            duration = time.time() - start_time
            success_count = sum(
                1 for r in results if not r.text.startswith("Translation failed"))
            logger.info(
                f"Document translation completed in {duration:.2f}s - "
                f"{len(paragraphs)} paragraphs, {success_count} successful"
            )

            return results
        finally:
            self._context_cache.clear()
            self._paragraphs = []
