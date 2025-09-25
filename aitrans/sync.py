"""同步封装模块"""

import asyncio
import logging
import time
import nest_asyncio
from typing import Optional, Union, List
from .core import AITranslator
from .models import AITranslated
from .exceptions import AIError
from .managers import SyncClientManager
from .cache import SyncMultiLevelCache

logger = logging.getLogger(__name__)

# 允许嵌套使用事件循环
nest_asyncio.apply()


class AITranslatorSync:
    """AITranslator的同步封装类，使异步API更易使用"""

    def __init__(self, api_key=None, model_name=None, base_url=None,
                 max_workers=5, glossary_path=None,
                 performance_mode='balanced', auto_preconnect=True, **kwargs):
        """初始化同步翻译器"""
        self.translator = None
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_workers = max_workers
        self.glossary_path = glossary_path
        self.performance_mode = performance_mode
        self.auto_preconnect = auto_preconnect
        self.kwargs = kwargs
        self._initialized = False
        self.client_manager = None
        self.cache = None

        # 初始化事件循环
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def _ensure_loop(self):
        """确保事件循环可用"""
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _ensure_translator(self):
        """确保翻译器已初始化"""
        if not self._initialized:
            # 创建异步翻译器实例
            self.translator = AITranslator(
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                max_workers=self.max_workers,
                glossary_path=self.glossary_path,
                performance_mode=self.performance_mode,
                **self.kwargs
            )

            # 初始化同步包装器
            self.client_manager = SyncClientManager(
                self.translator.client_manager)
            self.cache = SyncMultiLevelCache(self.translator.cache)

            # 初始化异步翻译器
            self._loop.run_until_complete(self.translator.initialize())
            self._initialized = True

            # 自动预热
            if self.auto_preconnect:
                self.warmup()

    def translate(self, text: Union[str, List[str]], dest='en', src='auto', stream=False, max_retries=3) -> Union[AITranslated, List[AITranslated]]:
        """翻译方法（带重试机制）"""
        self._ensure_translator()

        if isinstance(text, list):
            return self.translate_batch_sync(text, dest=dest, src=src)

        if stream:
            return self._create_stream_generator(text, dest, src)

        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                return self._loop.run_until_complete(
                    self.translator.translate(text, dest=dest, src=src)
                )
            except AIError as e:
                attempt += 1
                last_error = e
                if attempt >= max_retries:
                    break
                wait_time = min(2 ** attempt, 10)  # 指数退避
                time.sleep(wait_time)

        raise last_error or AIError("翻译失败")

    def _create_stream_generator(self, text: str, dest: str, src: str):
        """创建流式翻译生成器"""
        self._ensure_translator()

        async def stream_generator():
            try:
                stream = await self.translator.translate(text, dest=dest, src=src, stream=True)
                async for partial in stream:
                    yield partial
            except Exception as e:
                logger.error(f"流式翻译出错: {str(e)}")
                raise

        gen = stream_generator()
        while True:
            try:
                yield self._loop.run_until_complete(gen.__anext__())
            except StopAsyncIteration:
                break
            except Exception as e:
                logger.error(f"流式翻译生成器出错: {str(e)}")
                raise

    def __enter__(self):
        """上下文管理器入口"""
        self._ensure_translator()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self._initialized:
            try:
                if self.client_manager:
                    self.client_manager.cleanup()
                if self.cache:
                    self.cache.clear()
                if hasattr(self, 'translator') and self.translator:
                    self._loop.run_until_complete(self.translator.cleanup())
                self._initialized = False
            except Exception as e:
                logger.error(f"清理资源失败: {str(e)}")

    def translate_batch_sync(self, texts: List[str], dest='en', src='auto', max_retries=3) -> List[AITranslated]:
        """批量翻译方法（带重试机制）"""
        self._ensure_translator()

        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                return self._loop.run_until_complete(
                    self.translator.translate_batch(texts, dest=dest, src=src)
                )
            except AIError as e:
                attempt += 1
                last_error = e
                if attempt >= max_retries:
                    break
                wait_time = min(2 ** attempt, 10)
                time.sleep(wait_time)

        raise last_error or AIError("批量翻译失败")

    def translate_with_style(self, text: str, dest='en', src='auto', style='formal', context=None, max_retries=3) -> AITranslated:
        """带风格的翻译方法（带重试机制）"""
        self._ensure_translator()

        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                return self._loop.run_until_complete(
                    self.translator.translate_with_style(
                        text, dest=dest, src=src, style=style, context=context
                    )
                )
            except AIError as e:
                attempt += 1
                last_error = e
                if attempt >= max_retries:
                    break
                wait_time = min(2 ** attempt, 10)
                time.sleep(wait_time)

        raise last_error or AIError("风格化翻译失败")

    def translate_with_context(self, text: str, context: str, dest='en', src='auto', max_retries=3) -> AITranslated:
        """带上下文的翻译方法（带重试机制）"""
        self._ensure_translator()

        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                return self._loop.run_until_complete(
                    self.translator.translate_with_context(
                        text, context, dest=dest, src=src
                    )
                )
            except AIError as e:
                attempt += 1
                last_error = e
                if attempt >= max_retries:
                    break
                wait_time = min(2 ** attempt, 10)
                time.sleep(wait_time)

        raise last_error or AIError("上下文翻译失败")

    def warmup(self):
        """预热连接"""
        self._ensure_translator()
        self._loop.run_until_complete(self.translator.warmup())

    def test_connection(self) -> bool:
        """测试连接"""
        self._ensure_translator()
        return self._loop.run_until_complete(self.translator.test_connection())

    def get_config(self) -> dict:
        """获取配置信息"""
        self._ensure_translator()
        return self.translator.get_config()

    def get_metrics(self):
        """获取性能指标"""
        self._ensure_translator()
        return self.translator.metrics.get_metrics()

    def __del__(self):
        """析构时确保资源被清理"""
        if self._initialized:
            try:
                if self.client_manager:
                    self.client_manager.cleanup()
                if self.cache:
                    self.cache.clear()
                self._initialized = False
            except Exception as e:
                logger.error(f"析构时清理资源失败: {str(e)}")


def create(api_key=None, model_name=None, base_url=None, performance_mode='balanced', **kwargs) -> AITranslatorSync:
    """创建翻译器实例的工厂方法"""
    translator = AITranslatorSync(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        performance_mode=performance_mode,
        auto_preconnect=True,
        **kwargs
    ).__enter__()

    return translator
