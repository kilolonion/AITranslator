"""文档翻译模块"""

import logging
from itertools import chain
from time import perf_counter
from typing import List, Optional, Tuple

from .models import AITranslated
from .utils import BatchProcessor

logger = logging.getLogger(__name__)


class DocumentTranslator:
    """文档翻译器"""

    def __init__(
        self,
        translator: 'AITranslator',
        context_window: int = 2,
        batch_size: int = 5,
        max_workers: Optional[int] = None,
    ):
        self.translator = translator
        self.context_window = context_window
        self.batch_size = batch_size
        perf_workers = max_workers or translator.perf_config.get(
            'max_workers', translator.batch_processor.max_workers
        )
        self.processor = BatchProcessor(
            max_workers=int(perf_workers),
            batch_size=batch_size
        )

    def _build_contexts(self, paragraphs: List[str]) -> List[str]:
        total = len(paragraphs)
        contexts: List[str] = [""] * total
        window = self.context_window

        for index in range(total):
            start_idx = max(0, index - window)
            end_idx = min(total, index + window + 1)
            context_iter = chain(
                paragraphs[start_idx:index],
                paragraphs[index + 1 : end_idx],
            )
            contexts[index] = "\n".join(context_iter)

        return contexts

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

        contexts = self._build_contexts(paragraphs)
        start_time = perf_counter()

        async def translate_paragraph(text: str, index: int) -> AITranslated:
            try:
                context = contexts[index]
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

        async def runner(item: Tuple[int, str]) -> AITranslated:
            index, text = item
            return await translate_paragraph(text, index)

        indexed_paragraphs = list(enumerate(paragraphs))
        results = await self.processor.process(
            indexed_paragraphs, runner, batch_size=self.batch_size
        )

        duration = perf_counter() - start_time
        success_count = sum(1 for r in results if not r.text.startswith("Translation failed"))
        logger.info(
            f"Document translation completed in {duration:.2f}s - "
            f"{len(paragraphs)} paragraphs, {success_count} successful"
        )

        return results
