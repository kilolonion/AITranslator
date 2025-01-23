import os
import json
import time
import asyncio
import logging
import threading
from typing import Dict, Any, List, AsyncGenerator, Union, Optional

import openai
from langdetect import detect_langs, DetectorFactory, LangDetectException
import aiohttp

# 配置日志
logger = logging.getLogger(__name__)

# 常量定义
MAX_WORKERS = 5

# 语言代码映射
LANG_CODE_MAP = {
    'zh-cn': 'zh',
    'zh-tw': 'zh',
    'en-us': 'en',
    'en-gb': 'en',
    'ja-jp': 'ja'
}

# 支持的语言列表
DOUBAO_LANGUAGES = {
    'zh', 'en', 'ja', 'ko', 'fr', 'de', 'es', 'pt', 'it', 'ru',
    'ar', 'hi', 'th', 'vi', 'id', 'ms', 'tr', 'pl', 'uk'
}


class AIError(Exception):
    """AI 翻译错误基类"""
    pass


class AIAPIError(AIError):
    """API 调用错误"""
    pass


class AITranslated:
    """翻译结果类"""

    def __init__(self, src: str, dest: str, origin: str, text: str):
        self.src = src
        self.dest = dest
        self.origin = origin
        self.text = text


class AIDetected:
    """语言检测结果类"""

    def __init__(self, lang: str, confidence: float):
        self.lang = lang
        self.confidence = confidence


class SessionManager:
    """管理异步HTTP会话"""

    def __init__(self):
        self._session = None

    async def initialize(self):
        """初始化会话"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def get_session(self) -> aiohttp.ClientSession:
        """获取会话实例"""
        if not self._session or self._session.closed:
            await self.initialize()
        return self._session

    async def cleanup(self):
        """清理会话资源"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


class ClientManager:
    """管理OpenAI客户端"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    async def initialize(self):
        """初始化客户端"""
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    async def get_client(self):
        """获取客户端实例"""
        if not self._client:
            await self.initialize()
        return self._client

    async def cleanup(self):
        """清理客户端资源"""
        if self._client:
            await self._client.close()
            self._client = None


class BatchProcessor:
    """处理批量任务"""

    def __init__(self, max_workers: int = 5, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process(self, items: list, processor_func) -> list:
        """处理批量任务"""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_tasks = []
            for item in batch:
                async with self.semaphore:
                    task = asyncio.create_task(processor_func(item))
                    batch_tasks.append(task)
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        return results

    async def stop(self):
        """停止处理器"""
        pass


class PerformanceMetrics:
    """跟踪性能指标"""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.total_duration = 0.0
        self.min_duration = float('inf')
        self.max_duration = 0.0
        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_request(self, duration: float, success: bool):
        """记录请求性能指标"""
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
                self.total_duration += duration
                self.min_duration = min(self.min_duration, duration)
                self.max_duration = max(self.max_duration, duration)

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            uptime = time.time() - self._start_time
            if self.total_requests == 0:
                return {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "success_rate": 0.0,
                    "average_duration": 0.0,
                    "min_duration": 0.0,
                    "max_duration": 0.0,
                    "requests_per_second": 0.0,
                    "uptime_seconds": uptime
                }

            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "success_rate": self.successful_requests / self.total_requests,
                "average_duration": self.total_duration / self.successful_requests if self.successful_requests > 0 else 0.0,
                "min_duration": self.min_duration if self.min_duration != float('inf') else 0.0,
                "max_duration": self.max_duration,
                "requests_per_second": self.total_requests / uptime,
                "uptime_seconds": uptime
            }

    def reset(self):
        """重置性能指标"""
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.total_duration = 0.0
            self.min_duration = float('inf')
            self.max_duration = 0.0
            self._start_time = time.time()


class AITranslator:
    """异步翻译器类"""

    def __init__(self, api_key=None, model_name=None, base_url=None,
                 max_workers=MAX_WORKERS, glossary_path=None,
                 performance_mode='balanced', **kwargs):
        """初始化翻译器"""
        self.api_key = api_key or os.getenv('ARK_API_KEY')
        self.model_name = model_name or os.getenv('ARK_MODEL', 'deepseek-chat')
        self.base_url = base_url or os.getenv('ARK_BASE_URL')
        self.max_workers = max_workers
        self.glossary_path = glossary_path
        self.performance_mode = performance_mode
        self.kwargs = kwargs

        # 初始化组件
        self.session_manager = SessionManager()
        self.client_manager = ClientManager(self.api_key, self.base_url)
        self.batch_processor = BatchProcessor(max_workers=max_workers)
        self.metrics = PerformanceMetrics()

        # 设置性能配置
        self.perf_config = self.get_performance_profile(performance_mode)

        # 初始化缓存
        self._response_cache = {}
        self._context_cache = {}

        # 设置系统提示词
        self.system_prompt = "你是一个专业的翻译助手，请准确翻译用户的文本，保持原文的语气和风格。"

        logger.info("AITranslator initialized with model: %s and performance_mode: %s",
                    self.model_name, self.performance_mode)

    async def initialize(self):
        """初始化异步资源"""
        await self.session_manager.initialize()
        await self.client_manager.initialize()

        # 加载术语表（如果有）
        if self.glossary_path:
            self.load_glossary(self.glossary_path)

    async def cleanup(self):
        """清理资源"""
        await self.session_manager.cleanup()
        await self.client_manager.cleanup()
        await self.batch_processor.stop()
        self.metrics.reset()

    def get_performance_profile(self, mode='balanced'):
        """获取性能配置"""
        profiles = {
            'fast': {
                'max_workers': 10,
                'cache_ttl': 3600,
                'min_request_interval': 0.1,
                'max_retries': 2,
                'timeout': 10,
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'balanced': {
                'max_workers': 5,
                'cache_ttl': 7200,
                'min_request_interval': 0.2,
                'max_retries': 3,
                'timeout': 20,
                'temperature': 0.5,
                'max_tokens': 2000
            },
            'accurate': {
                'max_workers': 3,
                'cache_ttl': 14400,
                'min_request_interval': 0.5,
                'max_retries': 5,
                'timeout': 30,
                'temperature': 0.3,
                'max_tokens': 4000
            }
        }
        return profiles.get(mode, profiles['balanced'])

    async def translate(self, text: str, dest: str = 'en', src: str = 'auto',
                        stream: bool = False) -> Union[AITranslated, AsyncGenerator[AITranslated, None]]:
        """翻译文本"""
        if stream:
            async for result in self._translate_stream(text, dest, src):
                yield result
        else:
            result = await self._translate_single(text, dest, src)
            return result

    async def translate_batch(self, texts: List[str], dest: str = 'en',
                              src: str = 'auto') -> List[AITranslated]:
        """批量翻译文本"""
        async def translate_one(text):
            return await self.translate(text, dest, src)

        return await self.batch_processor.process(texts, translate_one)

    async def translate_with_style(self, text: str, dest: str = 'en',
                                   src: str = 'auto', style: str = 'formal',
                                   context: str = None) -> AITranslated:
        """带风格的翻译"""
        style_prompts = {
            'formal': "请使用正式的语气翻译",
            'casual': "请使用日常口语翻译",
            'creative': "请使用富有创意的方式翻译"
        }

        prompt = f"{style_prompts.get(style, style_prompts['formal'])}。"
        if context:
            prompt += f"\n上下文：{context}"

        result = await self._translate_with_prompt(text, dest, src, prompt)
        return result

    async def translate_with_context(self, text: str, context: str,
                                     dest: str = 'en', src: str = 'auto') -> AITranslated:
        """带上下文的翻译"""
        prompt = f"请根据以下上下文翻译文本：\n上下文：{context}\n要翻译的文本：{text}"
        return await self._translate_with_prompt(text, dest, src, prompt)

    async def preconnect(self):
        """预热连接"""
        await self.client_manager.initialize()
        try:
            await self.translate("test", "en", "en")
        except Exception as e:
            logger.warning(f"预热失败，但不影响使用: {str(e)}")

    def load_glossary(self, path: str):
        """加载术语表"""
        pass  # 待实现

    async def _translate_single(self, text: str, dest: str, src: str) -> AITranslated:
        """单个文本翻译实现"""
        try:
            text = text.strip()
            if not text:
                return AITranslated(src, dest, text, text)

            # 检查缓存
            cache_key = f"{text}:{src}:{dest}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            # 语言检测
            if src == 'auto':
                try:
                    detected = await self.detect(text)
                    src = detected.lang
                except Exception as e:
                    logger.warning(f"语言检测失败，使用 'auto': {str(e)}")

            # 构建提示词
            prompt = f"将以下{src}文本翻译成{dest}：\n{text}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            # 发送请求
            start_time = time.time()
            client = await self.client_manager.get_client()
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.perf_config['temperature'],
                max_tokens=self.perf_config['max_tokens']
            )
            duration = time.time() - start_time

            # 记录性能指标
            self.metrics.record_request(duration, True)

            # 处理响应
            translated_text = response.choices[0].message.content.strip()
            result = AITranslated(src, dest, text, translated_text)

            # 缓存结果
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"翻译失败: {text[:50]}... 错误: {str(e)}")
            raise AIAPIError(f"翻译失败: {str(e)}")

    async def _translate_stream(self, text: str, dest: str, src: str) -> AsyncGenerator[AITranslated, None]:
        """流式翻译实现"""
        try:
            text = text.strip()
            if not text:
                yield AITranslated(src, dest, text, text)
                return

            # 语言检测
            if src == 'auto':
                try:
                    detected = await self.detect(text)
                    src = detected.lang
                except Exception as e:
                    logger.warning(f"语言检测失败，使用 'auto': {str(e)}")

            # 构建提示词
            prompt = f"将以下{src}文本翻译成{dest}：\n{text}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            # 发送流式请求
            start_time = time.time()
            client = await self.client_manager.get_client()
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.perf_config['temperature'],
                max_tokens=self.perf_config['max_tokens'],
                stream=True
            )

            translated_text = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    translated_text += chunk.choices[0].delta.content
                    yield AITranslated(src, dest, text, translated_text)

            # 记录性能指标
            duration = time.time() - start_time
            self.metrics.record_request(duration, True)

        except Exception as e:
            logger.error(f"流式翻译失败: {text[:50]}... 错误: {str(e)}")
            raise AIAPIError(f"流式翻译失败: {str(e)}")

    async def _translate_with_prompt(self, text: str, dest: str, src: str, prompt: str) -> AITranslated:
        """带提示词的翻译实现"""
        try:
            text = text.strip()
            if not text:
                return AITranslated(src, dest, text, text)

            # 语言检测
            if src == 'auto':
                try:
                    detected = await self.detect(text)
                    src = detected.lang
                except Exception as e:
                    logger.warning(f"语言检测失败，使用 'auto': {str(e)}")

            # 构建消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{prompt}\n{text}"}
            ]

            # 发送请求
            start_time = time.time()
            client = await self.client_manager.get_client()
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.perf_config['temperature'],
                max_tokens=self.perf_config['max_tokens']
            )
            duration = time.time() - start_time

            # 记录性能指标
            self.metrics.record_request(duration, True)

            # 处理响应
            translated_text = response.choices[0].message.content.strip()
            return AITranslated(src, dest, text, translated_text)

        except Exception as e:
            logger.error(f"带提示词翻译失败: {text[:50]}... 错误: {str(e)}")
            raise AIAPIError(f"带提示词翻译失败: {str(e)}")

    def _get_from_cache(self, key: str) -> Optional[AITranslated]:
        """从缓存获取结果"""
        if key in self._response_cache:
            cached_time, result = self._response_cache[key]
            if time.time() - cached_time < self.perf_config['cache_ttl']:
                return result
            else:
                del self._response_cache[key]
        return None

    def _add_to_cache(self, key: str, result: AITranslated):
        """添加结果到缓存"""
        self._response_cache[key] = (time.time(), result)

    async def detect(self, text: str) -> AIDetected:
        """检测文本的语言"""
        if not text or not text.strip():
            logger.warning("检测空文本，返回 auto")
            return AIDetected('auto', 0.0)

        try:
            # 设置随机种子确保结果一致性
            DetectorFactory.seed = 0

            # 使用 langdetect 进行检测
            detected_langs = detect_langs(text)

            if not detected_langs:
                logger.warning("无法检测语言，返回 auto")
                return AIDetected('auto', 0.0)

            # 获取最可能的语言
            most_probable = detected_langs[0]
            lang_code = most_probable.lang
            confidence = most_probable.prob

            # 标准化语言代码
            normalized_lang = LANG_CODE_MAP.get(lang_code, lang_code)

            # 验证语言是否支持
            if normalized_lang not in DOUBAO_LANGUAGES:
                logger.warning(f"检测到不支持的语言: {normalized_lang}，返回 auto")
                return AIDetected('auto', 0.0)

            logger.info(f"语言检测结果: {normalized_lang}, 置信度: {confidence:.2f}")
            return AIDetected(normalized_lang, confidence)

        except LangDetectException as e:
            logger.error(f"语言检测失败: {str(e)}")
            return AIDetected('auto', 0.0)
        except Exception as e:
            logger.error(f"语言检测过程出错: {str(e)}")
            return AIDetected('auto', 0.0)


class AITranslatorSync:
    """同步翻译器类"""

    @classmethod
    def quick_translate(cls, text: str, dest='en', src='auto') -> str:
        """
        快速翻译方法，无需创建实例

        Args:
            text: 要翻译的文本
            dest: 目标语言
            src: 源语言

        Returns:
            str: 翻译结果
        """
        with create() as translator:
            result = translator.translate(text, dest=dest, src=src)
            return result.text

    def __init__(self, api_key=None, model_name=None, base_url=None,
                 max_workers=MAX_WORKERS, glossary_path=None,
                 performance_mode='balanced', auto_preconnect=True, **kwargs):
        """初始化翻译器"""
        self.translator = None
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_workers = max_workers
        self.glossary_path = glossary_path
        self.performance_mode = performance_mode
        self.auto_preconnect = auto_preconnect
        self.kwargs = kwargs
        self._loop = None
        self._initialized = False

    def _ensure_translator(self):
        """确保翻译器已初始化"""
        if not self._initialized:
            if not self._loop:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

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

            # 初始化异步翻译器
            self._loop.run_until_complete(self.translator.initialize())
            self._initialized = True

    def translate(self, text, dest='en', src='auto', stream=False):
        """翻译方法（自动初始化）"""
        self._ensure_translator()
        if not stream:
            result = self._loop.run_until_complete(
                self.translator.translate(text, dest=dest, src=src)
            )
            return result
        else:
            async def stream_generator():
                async for partial in self.translator.translate(text, dest=dest, src=src, stream=True):
                    yield partial
            return self._create_sync_generator(stream_generator())

    def _create_sync_generator(self, async_gen):
        """将异步生成器转换为同步生成器"""
        while True:
            try:
                yield self._loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break

    def __enter__(self):
        """上下文管理器入口（自动初始化）"""
        self._ensure_translator()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动清理资源"""
        try:
            if self._loop and not self._loop.is_closed():
                self._loop.run_until_complete(self._cleanup_resources())
                self._loop.close()
            self._initialized = False
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")

    async def _cleanup_resources(self):
        """清理所有资源"""
        if self.translator:
            await self.translator.cleanup()

    def get_metrics(self):
        """获取性能指标"""
        self._ensure_translator()
        return self.translator.metrics.get_metrics()

    def translate_batch_sync(self, texts, dest='en', src='auto'):
        """批量翻译方法"""
        self._ensure_translator()
        return self._loop.run_until_complete(
            self.translator.translate_batch(texts, dest=dest, src=src)
        )

    def translate_with_style(self, text, dest='en', src='auto', style='formal', context=None):
        """带风格的翻译方法"""
        self._ensure_translator()
        return self._loop.run_until_complete(
            self.translator.translate_with_style(
                text, dest=dest, src=src, style=style, context=context
            )
        )

    def translate_with_context(self, text, context, dest='en', src='auto'):
        """带上下文的翻译方法"""
        self._ensure_translator()
        return self._loop.run_until_complete(
            self.translator.translate_with_context(
                text, context, dest=dest, src=src)
        )

    def preconnect(self):
        """预热连接"""
        self._ensure_translator()
        self._loop.run_until_complete(self.translator.preconnect())

    def __del__(self):
        """析构时确保资源被清理"""
        if self._initialized:
            self.__exit__(None, None, None)


# 工厂函数
def create(api_key=None, model_name=None, base_url=None, performance_mode='balanced', **kwargs) -> 'AITranslatorSync':
    """
    创建翻译器实例的工厂方法

    Args:
        api_key: API密钥，如未提供则从环境变量获取
        model_name: 模型名称，如未提供则使用默认模型
        base_url: API的基础URL，如未提供则使用默认URL
        performance_mode: 性能模式('fast', 'balanced', 'accurate')
        **kwargs: 其他配置参数

    Returns:
        AITranslatorSync: 已初始化的翻译器实例
    """
    translator = AITranslatorSync(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        performance_mode=performance_mode,
        **kwargs
    ).__enter__()

    # 自动预热
    try:
        translator.preconnect()
    except Exception as e:
        logger.warning(f"预热失败，但不影响使用: {str(e)}")

    return translator
