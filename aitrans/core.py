"""核心翻译模块"""

import os
import logging
import time
import asyncio
import json
from typing import List, Union, Dict, Optional, AsyncGenerator
from dotenv import load_dotenv
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pathlib import Path

from .models import AITranslated, AIDetected, TranslationProgress, TranslationRecoveryState
from .exceptions import (
    AIError, AIAuthenticationError, AIConnectionError,
    AIAPIError, AIConfigError, AIValidationError
)
from .constants import (
    DEFAULT_BASE_URL, DEFAULT_MODEL, MAX_RETRIES, MAX_WORKERS,
    DEFAULT_PERFORMANCE_CONFIG, PERFORMANCE_PROFILES,
    DOUBAO_LANGUAGES, LANG_CODE_MAP, STYLE_TEMPLATES
)
from .utils import (
    BatchProcessor, SmartBatchProcessor, DynamicRateLimiter,
    SmartRetryHandler, EnhancedCache, PerformanceMetrics
)
from .managers import ClientManager
from .cache import MultiLevelCache

logger = logging.getLogger(__name__)


class AITranslator:
    """AI翻译器类"""

    def __init__(self, api_key=None, model_name=None, base_url=None,
                 max_workers=MAX_WORKERS, glossary_path=None,
                 performance_mode='balanced', **kwargs):
        """初始化AITranslator对象"""
        # 加载环境变量
        load_dotenv(override=True)

        # 设置API配置
        self.api_key = api_key or os.getenv('ARK_API_KEY')
        self.base_url = base_url or os.getenv('ARK_BASE_URL', DEFAULT_BASE_URL)
        self.model = model_name or os.getenv('ARK_MODEL', DEFAULT_MODEL)

        if not self.api_key:
            raise AIAuthenticationError("API密钥未提供")

        # 验证性能模式
        if performance_mode not in PERFORMANCE_PROFILES:
            raise AIConfigError(f"无效的性能模式: {performance_mode}")

        # 加载性能配置
        self.perf_config = PERFORMANCE_PROFILES[performance_mode].copy()
        self.perf_config.update(kwargs)

        # 初始化新的连接池管理器
        self.client_manager = ClientManager(
            pool_size=self.perf_config.get('pool_size', 100),
            keepalive_timeout=self.perf_config.get('keepalive_timeout', 30),
            ttl_dns_cache=self.perf_config.get('ttl_dns_cache', 300),
            api_key=self.api_key
        )

        # 初始化新的缓存系统
        self.cache = MultiLevelCache(
            memory_size=self.perf_config.get('cache_size', 10000),
            file_cache_path=Path(self.perf_config.get('cache_dir', './cache')),
            default_ttl=self.perf_config.get('cache_ttl', 3600)
        )

        # 初始化其他组件
        self.metrics = PerformanceMetrics()
        self.batch_processor = BatchProcessor(max_workers=max_workers)
        self.smart_batch_processor = SmartBatchProcessor()
        self.rate_limiter = DynamicRateLimiter()
        self.retry_handler = SmartRetryHandler()

        # 其他设置
        self.system_prompt = "你是翻译助手，请直接翻译用户的文本，不要添加任何解释。"
        self.glossary = {}

    async def initialize(self):
        """初始化异步资源"""
        await self.client_manager.warmup([
            f"{self.base_url}/chat/completions",
            f"{self.base_url}/models"
        ])
        await self.batch_processor.initialize()
        await self.smart_batch_processor.initialize()

    async def cleanup(self):
        """清理资源"""
        await self.client_manager.cleanup()
        await self.cache.clear()

    async def translate(self, text: Union[str, List[str]], dest: str = 'en',
                        src: str = 'auto', stream: bool = False) -> Union[AITranslated, List[AITranslated], AsyncGenerator[str, None]]:
        """翻译方法"""
        if isinstance(text, list):
            result = await self.translate_batch(text, dest=dest, src=src)
            return result

        if stream:
            return self._stream_translate(text, dest, src)

        try:
            # 检查缓存
            cache_key = self.cache._generate_key(text, src, dest)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result

            # 构建提示
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"将以下{src}文本翻译成{dest}：\n{text}"}
            ]

            # 发送请求
            async with self.client_manager.get_session("translate") as session:
                await self.rate_limiter.wait()

                start_time = time.time()
                try:
                    response = await session.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.perf_config['temperature'],
                            "max_tokens": self.perf_config['max_tokens']
                        }
                    )
                    response_data = await response.json()
                    translated_text = response_data['choices'][0]['message']['content'].strip(
                    )

                    # 记录成功
                    duration = time.time() - start_time
                    await self.rate_limiter.adjust_rate(True)
                    self.metrics.record_request(duration, True)

                    # 创建结果对象
                    result = AITranslated(src, dest, text, translated_text)

                    # 缓存结果
                    await self.cache.set(cache_key, result)

                    return result

                except Exception as e:
                    # 记录失败
                    duration = time.time() - start_time
                    await self.rate_limiter.adjust_rate(False)
                    self.metrics.record_request(duration, False)
                    raise

        except Exception as e:
            logger.error(f"翻译失败: {str(e)}")
            raise AIAPIError(f"翻译失败: {str(e)}")

    async def warmup(self):
        """预热系统"""
        # 预热连接
        await self.client_manager.warmup([
            f"{self.base_url}/chat/completions",
            f"{self.base_url}/models"
        ])

    async def _stream_translate(self, text: str, dest: str, src: str) -> AsyncGenerator[str, None]:
        """流式翻译实现"""
        try:
            if src == 'auto':
                detected = await self.detect(text)
                src = detected.lang

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"将以下{src}文本翻译成{dest}：\n{text}"}
            ]

            async with self.client_manager.get_session("translate") as session:
                await self.rate_limiter.wait()
                start_time = time.time()

                try:
                    response = await session.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.perf_config['temperature'],
                            "max_tokens": self.perf_config['max_tokens'],
                            "stream": True
                        }
                    )

                    accumulated_text = ""
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode(
                                    'utf-8').strip('data: '))
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        accumulated_text += content
                                        yield accumulated_text
                            except json.JSONDecodeError:
                                continue

                    # 记录成功
                    duration = time.time() - start_time
                    await self.rate_limiter.adjust_rate(True)
                    self.metrics.record_request(duration, True)

                except Exception as e:
                    # 记录失败
                    duration = time.time() - start_time
                    await self.rate_limiter.adjust_rate(False)
                    self.metrics.record_request(duration, False)
                    raise

        except Exception as e:
            logger.error(f"流式翻译失败: {str(e)}")
            raise AIAPIError(f"流式翻译失败: {str(e)}")

    async def translate_batch(self, texts: List[str], dest: str = 'en', src: str = 'auto') -> List[AITranslated]:
        """批量翻译实现"""
        if not texts:
            return []

        results = []
        for text in texts:
            if self.smart_batch_processor.add_to_batch(text):
                continue

            # 处理当前批次
            batch_results = await self.smart_batch_processor.process_batch(
                lambda batch: asyncio.gather(*[
                    self.translate(t, dest=dest, src=src) for t in batch
                ])
            )
            results.extend(batch_results)

            # 添加新文本到新批次
            self.smart_batch_processor.add_to_batch(text)

        # 处理最后的批次
        if self.smart_batch_processor.current_batch:
            batch_results = await self.smart_batch_processor.process_batch(
                lambda batch: asyncio.gather(*[
                    self.translate(t, dest=dest, src=src) for t in batch
                ])
            )
            results.extend(batch_results)

        return results

    async def detect(self, text: str) -> AIDetected:
        """检测文本语言"""
        if not text or not text.strip():
            return AIDetected('auto', 0.0)

        try:
            DetectorFactory.seed = 0
            detected_langs = detect_langs(text)

            if not detected_langs:
                return AIDetected('auto', 0.0)

            most_probable = detected_langs[0]
            lang_code = most_probable.lang
            confidence = most_probable.prob

            normalized_lang = LANG_CODE_MAP.get(lang_code, lang_code)
            if normalized_lang not in DOUBAO_LANGUAGES:
                return AIDetected('auto', 0.0)

            return AIDetected(normalized_lang, confidence)

        except LangDetectException:
            return AIDetected('auto', 0.0)

    async def translate_with_style(self, text: str, dest: str = 'en', src: str = 'auto',
                                   style: Union[str, Dict] = 'formal', context: str = None) -> AITranslated:
        """带风格的翻译"""
        if isinstance(style, str):
            if style not in STYLE_TEMPLATES:
                raise AIValidationError(f"未知的预定义风格: {style}")
            style_prompt = STYLE_TEMPLATES[style]
        else:
            style_prompt = "翻译要求：\n" + \
                "\n".join([f"{k}: {v}" for k, v in style.items()])

        context_part = f"\n相关上下文：\n{context}\n" if context else ""

        messages = [
            {"role": "system", "content": "你是一个专业的翻译手，请按照指定的风格要求进行翻译。"},
            {"role": "user", "content": f"""
{style_prompt}

{context_part}
需要翻译的文本：
{text}

请直接提供翻译结果，不要添加任何解释。
"""}
        ]

        async with self.client_manager.get_session("translate") as session:
            response = await session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.perf_config['temperature'],
                    "max_tokens": self.perf_config['max_tokens']
                }
            )
            response_data = await response.json()
            translated_text = response_data['choices'][0]['message']['content'].strip(
            )

            return AITranslated(src, dest, text, translated_text)

    async def translate_with_context(self, text: str, context: str, dest: str = 'en',
                                     src: str = 'auto', style_guide: str = None) -> AITranslated:
        """带上下文的翻译"""
        if src == 'auto':
            detected = await self.detect(text)
            src = detected.lang

        style_instructions = f"\n\n风格要求：\n{style_guide}" if style_guide else ""
        prompt = f"""请在理解以下上下文的基础上，将文本从{src}翻译成{dest}：

上下文背景：
{context}

需要翻译的文本：
{text}

翻译要求：
1. 保持与上下文的连贯性和一致性
2. 保留专业术语准确性
3. 保持原文的语气和风格
4. 保持代词指代的正确性
5. 注意上下文中的特定含义{style_instructions}

请直接返回翻译结果，不要添加任何解释。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        async with self.client_manager.get_session("translate") as session:
            response = await session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.perf_config['temperature'],
                    "max_tokens": self.perf_config['max_tokens']
                }
            )
            response_data = await response.json()
            translated_text = response_data['choices'][0]['message']['content'].strip(
            )

            return AITranslated(src, dest, text, translated_text)

    async def translate_with_recovery(self, text: str, dest: str = 'en',
                                      src: str = 'auto') -> AITranslated:
        """带恢复机制的翻译"""
        recovery_state = TranslationRecoveryState()

        while recovery_state.attempts < self.perf_config['max_retries']:
            try:
                result = await self.translate(text, dest, src)
                return result

            except Exception as e:
                recovery_state.record_attempt(e)
                should_retry, wait_time = self.retry_handler.should_retry(e)

                if not should_retry:
                    raise AIError(f"翻译失败，无法恢复: {str(e)}")

                await asyncio.sleep(wait_time)

        raise AIError(f"达到最大重试次数 ({self.perf_config['max_retries']})，翻译失败")

    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            async with self.client_manager.get_session("translate") as session:
                await session.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": self.model, "messages": [
                        {"role": "user", "content": "test"}], "max_tokens": 5}
                )
            return True
        except Exception as e:
            logger.error(f"连接测试失败: {str(e)}")
            return False

    def get_config(self) -> dict:
        """获取当前配置"""
        return {
            'api_key': f"{self.api_key[:8]}...",
            'base_url': self.base_url,
            'model': self.model,
            'performance_config': self.perf_config
        }

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
