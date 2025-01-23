"""流式翻译模块"""

import logging
from typing import AsyncGenerator
from .models import AITranslated

logger = logging.getLogger(__name__)


class StreamTranslator:
    """流式翻译的异步迭代器实现"""

    def __init__(self, translator: 'AITranslator', text: str, dest: str, src: str):
        self.translator = translator
        self.text = text
        self.dest = dest
        self.src = src
        self._response_stream = None
        self._accumulated_text = ""
        self._is_first_chunk = True
        self._translation_complete = False
        self._request_sent = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> AITranslated:
        """实现异步迭代器的下一个值获取"""
        try:
            if self._translation_complete:
                raise StopAsyncIteration

            # 初始化流式响应（只在第一次调用时执行）
            if not self._request_sent:
                self._request_sent = True
                client = await self.translator.client_manager.get_client()
                messages = [
                    {"role": "system", "content": self.translator.system_prompt},
                    {"role": "user",
                        "content": f"将以下{self.src}文本翻译成{self.dest}：\n{self.text}"}
                ]

                self._response_stream = await client.chat.completions.create(
                    model=self.translator.model,
                    messages=messages,
                    stream=True,
                    temperature=self.translator.perf_config['temperature'],
                    max_tokens=self.translator.perf_config['max_tokens']
                )

            try:
                chunk = await anext(self._response_stream)
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        self._accumulated_text += content
                        return AITranslated(
                            self.src,
                            self.dest,
                            self.text,
                            self._accumulated_text,
                            metadata={
                                'status': 'streaming',
                                'is_partial': True,
                                'is_first_chunk': self._is_first_chunk,
                                'progress': len(self._accumulated_text) / len(self.text) * 100
                            }
                        )
                return await self.__anext__()

            except StopAsyncIteration:
                if not self._translation_complete:
                    self._translation_complete = True
                    if self._accumulated_text:
                        return AITranslated(
                            self.src,
                            self.dest,
                            self.text,
                            self._accumulated_text,
                            metadata={
                                'status': 'completed',
                                'is_partial': False,
                                'progress': 100
                            }
                        )
                raise

        except Exception as e:
            logger.error(f"流式翻译错误: {str(e)}")
            self._cleanup()
            raise

    def _cleanup(self):
        """清理资源"""
        self._response_stream = None
        self._accumulated_text = ""
        self._is_first_chunk = True
        self._translation_complete = True
        self._request_sent = False
