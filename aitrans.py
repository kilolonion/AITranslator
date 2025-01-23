import os
from dotenv import load_dotenv
import openai
from typing import List, Union, Dict, Optional, Tuple, Any, AsyncGenerator, Generator, Callable
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from pathlib import Path
import re
import uuid
import asyncio
import aiohttp
import httpx
import threading
from dataclasses import dataclass
from contextlib import asynccontextmanager

try:
    from collections.abc import MutableSet
except ImportError:
    from collections import MutableSet

# 确保在最开始就加载环境变量，并打印调试信息
load_dotenv(override=True)  # 添加 override=True 确保覆盖已存在的环境变量

# 添加调试日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 打印环境变量加载情况
logger.info("Environment variables loaded from .env file:")
logger.info(
    f"ARK_API_KEY: {'*' * 8 + os.getenv('ARK_API_KEY', '')[-4:] if os.getenv('ARK_API_KEY') else 'Not found'}")
logger.info(f"ARK_BASE_URL: {os.getenv('ARK_BASE_URL', 'Not found')}")
logger.info(f"ARK_MODEL: {os.getenv('ARK_MODEL', 'Not found')}")

# 首先定义常量
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
MAX_RETRIES = 3
MAX_WORKERS = 5  # 并发线程数

# 语言名称映射
LANGUAGE_NAMES = {
    'auto': '自动检测',
    'zh': '中文',
    'en': '英语',
    'ja': '日语',
    'ko': '韩语',
    'fr': '法语',
    'es': '西班牙语',
    'it': '意大利语',
    'de': '德语',
    'ru': '俄语',
    'pt': '葡萄牙语',
    'vi': '越南语',
    'th': '泰语',
    'ar': '阿拉伯语'
}

# 自定义异常类


class AIError(Exception):
    """AI翻译器基础异常类"""
    pass


class AIAuthenticationError(AIError):
    """认证错误"""
    pass


class AIConnectionError(AIError):
    """连接错误"""
    pass


class AIAPIError(AIError):
    """API调用错误"""
    pass


class AIConfigError(AIError):
    """配置错误"""
    pass


class AIValidationError(AIError):
    """输入验证错误"""
    pass


class AITranslationError(Exception):
    """AI翻译错误基类"""
    pass


# 性能配置常量
DEFAULT_PERFORMANCE_CONFIG = {
    'max_workers': 5,
    'cache_ttl': 3600,
    'min_request_interval': 0.1,
    'max_retries': 3,
    'timeout': 30,
    'temperature': 0.3,
    'max_tokens': 1024,
}

PERFORMANCE_PROFILES = {
    'fast': {
        'max_workers': 10,
        'cache_ttl': 1800,
        'min_request_interval': 0.05,
        'max_retries': 2,
        'timeout': 15,
        'temperature': 0.5,
        'max_tokens': 512,
    },
    'balanced': DEFAULT_PERFORMANCE_CONFIG,
    'accurate': {
        'max_workers': 3,
        'cache_ttl': 7200,
        'min_request_interval': 0.2,
        'max_retries': 5,
        'timeout': 60,
        'temperature': 0.1,
        'max_tokens': 2048,
    }
}

# 支持的语言代码常量
DOUBAO_LANGUAGES = {
    'zh': 'chinese',
    'en': 'english',
    'ja': 'japanese',
    'ko': 'korean',
    'fr': 'french',
    'es': 'spanish',
    'ru': 'russian',
    'de': 'german',
    'it': 'italian',
    'tr': 'turkish',
    'pt': 'portuguese',
    'vi': 'vietnamese',
    'id': 'indonesian',
    'th': 'thai',
    'ms': 'malay',
    'ar': 'arabic',
    'hi': 'hindi'
}

# 语言代码映射表
LANG_CODE_MAP = {
    # ISO 639-1 到语言代码的映射
    'zh-cn': 'zh', 'zh-tw': 'zh', 'zh': 'zh',
    'en': 'en',
    'ja': 'ja',
    'ko': 'ko',
    'fr': 'fr',
    'es': 'es',
    'ru': 'ru',
    'de': 'de',
    'it': 'it',
    'tr': 'tr',
    'pt': 'pt',
    'vi': 'vi',
    'id': 'id',
    'th': 'th',
    'ms': 'ms',
    'ar': 'ar',
    'hi': 'hi'
}


class AITranslated:
    """表示翻译结果的类"""

    def __init__(self, src, dest, origin, text, pronunciation=None, metadata=None):
        self.src = src
        self.dest = dest
        self.origin = origin
        self.text = text
        self.pronunciation = pronunciation
        self.metadata = metadata or {}

    def __repr__(self):
        return f'<AITranslated src={self.src} dest={self.dest} text={self.text} pronunciation={self.pronunciation} metadata={self.metadata}>'


@dataclass
class AIDetected:
    """表示语言检测结果的类"""
    lang: str
    confidence: float

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


class TranslationRecoveryState:
    """管理翻译恢复状态"""

    def __init__(self):
        self.attempts = 0
        self.last_error = None
        self.recovery_strategy = None
        self.partial_results = []
        self.start_time = time.time()

    def record_attempt(self, error: Exception):
        self.attempts += 1
        self.last_error = error

    def add_partial_result(self, result: AITranslated):
        self.partial_results.append(result)

    @property
    def duration(self) -> float:
        return time.time() - self.start_time


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
        """初始化批处理器

        Args:
            max_workers: 最大并发工作数
            batch_size: 每批处理的任务数
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_workers)
        self._is_initialized = False

    async def initialize(self):
        """初始化处理器"""
        if not self._is_initialized:
            # 这里可以添加额外的初始化逻辑
            self._is_initialized = True
            logger.debug("BatchProcessor initialized")

    async def process(self, items: list, processor_func) -> list:
        """处理批量任务

        Args:
            items: 要处理的项目列表
            processor_func: 处理单个项目的异步函数

        Returns:
            处理结果列表
        """
        if not self._is_initialized:
            await self.initialize()

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
        self._is_initialized = False
        logger.debug("BatchProcessor stopped")


class SmartBatchProcessor:
    """智能批处理处理器，支持基于token的动态批处理"""

    def __init__(self, max_batch_size=10, max_tokens=4000):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.current_batch = []
        self.current_tokens = 0
        self._is_initialized = False
        self.semaphore = None

    async def initialize(self):
        """初始化处理器"""
        if not self._is_initialized:
            self.semaphore = asyncio.Semaphore(self.max_batch_size)
            self._is_initialized = True

    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        return len(text.split()) * 1.3  # 简单估算

    def can_add_to_batch(self, text: str) -> bool:
        """检查是否可以添加到当前批次"""
        estimated_tokens = self.estimate_tokens(text)
        return (len(self.current_batch) < self.max_batch_size and
                self.current_tokens + estimated_tokens <= self.max_tokens)

    def add_to_batch(self, text: str) -> bool:
        """添加文本到批次"""
        if not self.can_add_to_batch(text):
            return False
        self.current_batch.append(text)
        self.current_tokens += self.estimate_tokens(text)
        return True

    def get_current_batch(self) -> List[str]:
        """获取当前批次并清空"""
        batch = self.current_batch[:]
        self.current_batch = []
        self.current_tokens = 0
        return batch

    async def process_batch(self, processor_func: Callable) -> List[Any]:
        """处理当前批次"""
        if not self.current_batch:
            return []
        batch = self.get_current_batch()
        async with self.semaphore:
            return await processor_func(batch)


class DynamicRateLimiter:
    """动态请求限流控制器"""

    def __init__(self, initial_rate=10, max_rate=20):
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.success_count = 0
        self.failure_count = 0
        self.window_size = 60  # 60秒窗口
        self.last_adjustment = time.time()
        self._lock = asyncio.Lock()

    async def adjust_rate(self, success: bool):
        """根据请求成功/失败动态调整速率"""
        async with self._lock:
            current_time = time.time()
            if current_time - self.last_adjustment >= self.window_size:
                # 重置计数器
                self.success_count = 0
                self.failure_count = 0
                self.last_adjustment = current_time

            if success:
                self.success_count += 1
                if self.success_count > 10 and self.current_rate < self.max_rate:
                    self.current_rate = min(
                        self.current_rate * 1.2, self.max_rate)
            else:
                self.failure_count += 1
                if self.failure_count > 2:
                    self.current_rate = max(self.current_rate * 0.8, 1)

    async def wait(self):
        """等待下一个请求的时间间隔"""
        wait_time = 1 / self.current_rate
        await asyncio.sleep(wait_time)


class SmartRetryHandler:
    """智能重试处理器"""

    def __init__(self):
        self.error_counts = defaultdict(int)
        self.last_errors = defaultdict(list)
        self.max_retries = 5

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """判断是否应该重试并返回等待时间"""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1

        if self.error_counts[error_type] > self.max_retries:
            return False, 0

        if isinstance(error, (openai.APIError, httpx.ConnectError)):
            wait_time = min(2 ** self.error_counts[error_type], 32)
            return True, wait_time

        if isinstance(error, openai.RateLimitError):
            wait_time = 60  # 固定等待时间
            return True, wait_time

        return False, 0

    def reset_error_count(self, error_type: str):
        """重置错误计数"""
        self.error_counts[error_type] = 0


class EnhancedCache:
    """增强的缓存实现"""

    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        """获取缓存值"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    self.hits += 1
                    return entry['value']
            self.misses += 1
            return None

    async def put(self, key: str, value: str):
        """存入缓存值"""
        async with self._lock:
            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }

    async def cleanup(self):
        """清理过期缓存"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                k for k, v in self.cache.items()
                if current_time - v['timestamp'] >= self.ttl
            ]
            for k in expired_keys:
                del self.cache[k]

    def get_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class AITranslator:
    """AI翻译器类"""

    # 类方法：语言相关的工具方法
    @classmethod
    def get_language_name(cls, code):
        """获取语言代码对应的语言名称"""
        return LANGUAGE_NAMES.get(code)

    @classmethod
    def get_language_code(cls, name):
        """根据语言名称获取语言代码"""
        for code, lang_name in LANGUAGE_NAMES.items():
            if lang_name == name:
                return code
        return None

    @classmethod
    def get_supported_languages(cls):
        """获取所有支持的语言及其代码"""
        return LANGUAGE_NAMES.copy()

    @classmethod
    def get_supported_languages_code(cls):
        """获取所有支持的语言代码"""
        return LANGUAGE_NAMES.keys()

    @classmethod
    def get_supported_languages_name(cls):
        """获取所有支持的语言名称"""
        return LANGUAGE_NAMES.values()

    @classmethod
    def is_language_supported(cls, code):
        """检查语言代码是否被支持"""
        return code in LANGUAGE_NAMES

    def __init__(self, api_key=None, model_name=None, base_url=None,
                 max_workers=MAX_WORKERS, glossary_path=None,
                 performance_mode='balanced', **kwargs):
        """
        初始化AITranslator对象。

        Args:
            api_key: API密钥，如未提供则从环境变量获取
            model_name: 模型名称，如未提供则使用默认模型
            base_url: API的基础URL，如未提供则使用默认URL
            max_workers: 最大并发线程数
            glossary_path: 术语表文件路径
            performance_mode: 性能模式('fast', 'balanced', 'accurate')
            **kwargs: 自定义性能参数，可覆盖预设配置

        Raises:
            AIConfigError: 配置参数无效
            AIAuthenticationError: API密钥无效
            AIValidationError: 参数验证失败
        """
        # 添加预连接状态标志
        self._is_preconnected = False
        self._preconnected_session = None
        self._preconnected_client = None

        try:
            # 打印环境变量调试信息
            logger.info(f"Environment variables:")
            logger.info(
                f"ARK_API_KEY: {'*' * 8 + os.getenv('ARK_API_KEY')[-4:] if os.getenv('ARK_API_KEY') else 'Not found'}")
            logger.info(f"ARK_BASE_URL: {os.getenv('ARK_BASE_URL')}")
            logger.info(f"ARK_MODEL: {os.getenv('ARK_MODEL')}")

            # 设置配置，优先使用传入的参数，其次使用环境变量
            self.api_key = api_key if api_key is not None else os.getenv(
                'ARK_API_KEY')
            self.base_url = base_url if base_url is not None else os.getenv(
                'ARK_BASE_URL', DEFAULT_BASE_URL)
            self.model = model_name if model_name is not None else os.getenv(
                'ARK_MODEL', DEFAULT_MODEL)

            # 打印最终使用的配置
            logger.info(f"Using configuration:")
            logger.info(
                f"API Key: {'*' * 8 + self.api_key[-4:] if self.api_key else 'Not found'}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Model: {self.model}")

            if not self.api_key:
                raise AIAuthenticationError(
                    "API密钥未提供。请通过参数传入api_key或设置环境变量 ARK_API_KEY")

            if not isinstance(self.api_key, str) or len(self.api_key) < 32:
                raise AIAuthenticationError("API密钥格式无效")

            # 验证性能模式
            if performance_mode not in PERFORMANCE_PROFILES:
                raise AIConfigError(
                    f"无效的性能模式。必须是: {', '.join(PERFORMANCE_PROFILES.keys())}")

            # 加载性能配置
            self.perf_config = PERFORMANCE_PROFILES[performance_mode].copy()
            self.perf_config.update(kwargs)  # 允许通过kwargs覆盖特定配置

            # 验证配置参数
            self._validate_config(self.perf_config)

            # 应用性能配置
            self._cache_ttl = self.perf_config['cache_ttl']
            self._min_request_interval = self.perf_config['min_request_interval']
            self.max_retries = self.perf_config['max_retries']

            # 初始化资源管理器
            self.session_manager = SessionManager()
            self.client_manager = ClientManager(self.api_key, self.base_url)
            self.semaphore = asyncio.Semaphore(self.perf_config['max_workers'])

            # 初始化异步锁
            self._cache_lock = asyncio.Lock()
            self._request_lock = asyncio.Lock()
            self._metrics_lock = asyncio.Lock()

            # 初始化其他组件
            self._init_components(max_workers, glossary_path)

            logger.info(
                f"AITranslator initialized with model: {self.model} and performance_mode: {performance_mode}")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _init_components(self, max_workers: int, glossary_path: Optional[str]):
        """初始化组件

        Args:
            max_workers: 最大并发数
            glossary_path: 术语表路径
        """
        # 基础组件
        self._last_request_time = 0
        self._response_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.system_prompt = "你是翻译助手，请直接翻译用户的文本，不要添加任何解释。"

        # 术语表初始化
        self.glossary: Dict[str, Dict[str, str]] = {}
        self.glossary_path = glossary_path  # 保存术语表路径
        if glossary_path:
            self.load_glossary(glossary_path)

        # 风格模板初始化
        self._init_style_templates()

        # 性能监控 - 确保在这里初始化
        self.metrics = PerformanceMetrics()

        # 批处理器初始化
        self.batch_processor = BatchProcessor(
            max_workers=self.perf_config['max_workers'],
            batch_size=10  # 可以从 perf_config 中获取或使用默认值
        )

        # 异步组件
        self.session = None
        self.semaphore = asyncio.Semaphore(max_workers)

        # 优化组件
        self.smart_batch_processor = SmartBatchProcessor(
            max_batch_size=self.perf_config.get('max_batch_size', 10),
            max_tokens=self.perf_config.get('max_tokens', 4000)
        )
        self.rate_limiter = DynamicRateLimiter(
            initial_rate=self.perf_config.get('initial_rate', 10),
            max_rate=self.perf_config.get('max_rate', 20)
        )
        self.retry_handler = SmartRetryHandler()
        self.enhanced_cache = EnhancedCache(ttl=self._cache_ttl)

    def _init_style_templates(self):
        """初始化翻译风格模板"""
        self.style_templates = {
            'formal': """
                翻译要求：
                1. 使用正式的学术用语
                2. 保持严谨的句式结构
                3. 使用标准的专业术语
                4. 避免口语化和简化表达
            """,
            'casual': """
                翻译要求：
                1. 使用日常口语表达
                2. 保持语言自然流畅
                3. 使用简短句式
                4. 可以使用常见缩写
            """,
            'technical': """
                翻译要求：
                1. 严格使用技术术语
                2. 保持专业准确性
                3. 使用规范的技术表达
                4. 保持术语一致性
            """,
            'creative': """
                翻译要求：
                1. 提供2-3个不同的翻译版本
                2. 每个版本使用不同的表达方式
                3. 保持原文的核心含义
                4. 限制在3个版本以内
            """
        }

    def _should_retry(self, exception: Exception) -> bool:
        """判断是否应该重试"""
        if isinstance(exception, (openai.APIError, openai.APIConnectionError)):
            return True
        if isinstance(exception, httpx.ConnectError):
            return True
        return False

    def _get_retry_config(self):
        """获取重试配置"""
        return {
            'multiplier': self.perf_config.get('retry_multiplier', 0.5),
            'min': self.perf_config.get('retry_min_wait', 1),
            'max': self.perf_config.get('retry_max_wait', 4),
            'max_retries': self.perf_config.get('max_retries', 3)
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(
            (openai.APIError, openai.APIConnectionError, httpx.ConnectError))
    )
    async def initialize(self):
        """初始化异步资源"""
        try:
            # 初始化会话管理器
            await self.session_manager.initialize()

            # 初始化客户端管理器
            await self.client_manager.initialize()

            # 初始化批处理器
            await self.batch_processor.initialize()

            # 加载术语表（如果有）
            if self.glossary_path:
                self.load_glossary(self.glossary_path)

            logger.info("AITranslator initialized successfully")

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise AIError(f"初始化失败: {str(e)}")

    async def _make_request(self, messages, stream=False):
        """改进的异步请求处理，支持预连接模式"""
        async with self._request_lock:
            # 使用动态限流
            await self.rate_limiter.wait()

            start_time = time.time()

            try:
                # 检查缓存
                cache_key = self._get_cache_key(messages)
                cached_response = await self.enhanced_cache.get(cache_key)
                if cached_response:
                    return cached_response

                # 验证API密钥
                if not self.api_key or len(self.api_key) < 32:
                    raise AIAuthenticationError("无效的API密钥")

                # 使用预连接的客户端或创建新的客户端
                client = self._preconnected_client if self._is_preconnected else await self.client_manager.get_client()

                async with self.semaphore:
                    try:
                        completion = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            stream=stream,
                            temperature=self.perf_config['temperature'],
                            max_tokens=self.perf_config['max_tokens'],
                            timeout=self.perf_config['timeout']
                        )

                        # 记录成功并调整速率
                        await self.rate_limiter.adjust_rate(True)

                        # 缓存响应
                        if not stream:
                            await self.enhanced_cache.put(cache_key, completion)

                        return completion

                    except Exception as e:
                        # 记录失败并调整速率
                        await self.rate_limiter.adjust_rate(False)

                        # 智能重试处理
                        should_retry, wait_time = self.retry_handler.should_retry(
                            e)
                        if should_retry:
                            await asyncio.sleep(wait_time)
                            return await self._make_request(messages, stream)

                        raise

            finally:
                duration = time.time() - start_time
                await self._record_metrics(duration, True)

    async def translate_batch(self, texts: List[str], dest='en', src='auto',
                              progress_callback: Callable[[TranslationProgress], None] = None):
        """改进的批量翻译实现"""
        results = []
        total_texts = len(texts)
        completed = 0

        async def process_batch(batch: List[str]) -> List[AITranslated]:
            nonlocal completed
            batch_results = []

            for text in batch:
                try:
                    result = await self._translate_single(text, dest, src)
                    batch_results.append(result)
                    completed += 1

                    if progress_callback:
                        progress = TranslationProgress(
                            completed=completed,
                            total=total_texts,
                            current_text=text
                        )
                        progress_callback(progress)

                except Exception as e:
                    logger.error(f"批量翻译错误: {str(e)}")
                    batch_results.append(None)
                    completed += 1

            return batch_results

        # 使用智能批处理
        for text in texts:
            if self.smart_batch_processor.add_to_batch(text):
                continue

            # 当前批次已满，处理它
            batch_results = await self.smart_batch_processor.process_batch(process_batch)
            results.extend(batch_results)

            # 添加新文本到新批次
            self.smart_batch_processor.add_to_batch(text)

        # 处理最后的批次
        if self.smart_batch_processor.current_batch:
            batch_results = await self.smart_batch_processor.process_batch(process_batch)
            results.extend(batch_results)

        return results

    def load_glossary(self, path: Union[str, Path]) -> None:
        """
        加载术语表

        :param path: 术语表文件路径
        :raises: FileNotFoundError 如果文件不存在
        :raises: json.JSONDecodeError 如果JSON格式无效
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"术语表文件不存在: {path}")

            with path.open('r', encoding='utf-8') as f:
                self.glossary = json.load(f)

            logger.info(
                f"Loaded glossary with {len(self.glossary)} terms from {path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in glossary file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading glossary: {e}")
            raise

    def save_glossary(self, path: Union[str, Path]) -> None:
        """
        保存术语表

        :param path: 保存路径
        """
        try:
            path = Path(path)
            with path.open('w', encoding='utf-8') as f:
                json.dump(self.glossary, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved glossary to {path}")
        except Exception as e:
            logger.error(f"Error saving glossary: {e}")
            raise

    def add_term(self, term_id: str, translations: Dict[str, str]) -> None:
        """
        加或更新术语

        :param term_id: 术语ID
        :param translations: 各语言的翻译
        """
        self.glossary[term_id] = translations
        logger.debug(
            f"Added/updated term: {term_id} with translations: {translations}")

    async def apply_glossary(self, text: str, src: str, dest: str) -> AITranslated:
        """应用术语表进行翻译"""
        if not self.glossary:
            return await self.translate(text, dest=dest, src=src)

        try:
            # 创建术语替换映射
            replacements = {}
            placeholder_format = "[[TERM_{}_]]"

            # 第一步：替换术语为占位符
            modified_text = text
            for term_id, translations in self.glossary.items():
                if src in translations and dest in translations:
                    source_term = translations[src]
                    target_term = translations[dest]

                    # 使用正则表达式进行完整词匹配
                    pattern = r'\b' + re.escape(source_term) + r'\b'
                    if re.search(pattern, modified_text, re.IGNORECASE):
                        placeholder = placeholder_format.format(term_id)
                        modified_text = re.sub(
                            pattern,
                            placeholder,
                            modified_text,
                            flags=re.IGNORECASE
                        )
                        replacements[placeholder] = target_term

            # 如果没有找到任何术语匹配，直接翻译原文
            if not replacements:
                return await self.translate(text, dest=dest, src=src)

            # 第二步：翻译修改后的文本
            translated = await self.translate(modified_text, dest=dest, src=src)
            result = translated.text

            # 第三步：还原术语
            for placeholder, term in replacements.items():
                result = result.replace(placeholder, term)

            logger.debug(
                f"Applied glossary translation with {len(replacements)} terms")
            return AITranslated(src, dest, text, result)

        except Exception as e:
            logger.error(f"Error applying glossary: {e}")
            # 如果术语表应用失败，回退到普通翻译
            return await self.translate(text, dest=dest, src=src)

    def get_term(self, term_id: str) -> Optional[Dict[str, str]]:
        """
        获取术语的翻译

        :param term_id: 术语ID
        :return: 术语的翻译字典，如果不存在返回None
        """
        return self.glossary.get(term_id)

    async def _translate_single(self, text: str, dest: str, src: str, stream: bool = False) -> AITranslated:
        """单个文本翻译实现"""
        try:
            text = text.strip()
            if not text:
                return AITranslated(src, dest, text, text)

            # 对于流式翻译，不使用缓存
            if not stream:
                cache_key = f"{text}:{src}:{dest}"
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result

            if src == 'auto':
                try:
                    detected = await self.detect(text)
                    src = detected.lang
                except Exception as e:
                    logger.warning(
                        f"Language detection failed, using 'auto': {str(e)}")

            prompt = f"将以下{src}文本翻译成{dest}：\n{text}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            start_time = time.time()
            translated_text = await self._make_request(messages, stream=stream)
            duration = time.time() - start_time
            self.metrics.record_request(duration, True)

            result = AITranslated(
                src=src,
                dest=dest,
                origin=text,
                text=translated_text.strip()
            )

            # 只缓存非流式翻译结果
            if not stream:
                self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(
                f"Translation failed for text: {text[:50]}... Error: {str(e)}")
            raise AIAPIError(f"翻译失败: {str(e)}")

    async def _create_stream_generator(self, messages: List[Dict[str, str]], src: str, dest: str, text: str) -> AsyncGenerator[AITranslated, None]:
        """创建流式翻译生成器

        Args:
            messages: 翻译提示消息列表
            src: 源语言
            dest: 目标语言
            text: 原文本

        Yields:
            AITranslated 对象
        """
        retry_count = 0
        last_error = None
        translated_text = ""

        while retry_count <= self.max_retries:
            try:
                start_time = time.time()
                response_stream = await self._make_request(messages, stream=True)

                async for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        translated_text += chunk.choices[0].delta.content
                        yield AITranslated(src, dest, text, translated_text)

                # 记录成功的请求
                duration = time.time() - start_time
                self.metrics.record_request(duration, True)
                break

            except Exception as e:
                duration = time.time() - start_time
                self.metrics.record_request(duration, False)
                last_error = e
                retry_count += 1

                if retry_count > self.max_retries:
                    logger.error(f"流式翻译失败，已重试 {self.max_retries} 次: {str(e)}")
                    raise AIAPIError(f"流式翻译失败: {str(e)}")

                logger.warning(
                    f"第 {retry_count}/{self.max_retries} 次重试，原因: {str(e)}")
                await asyncio.sleep(min(2 ** retry_count, 10))  # 指数退避，最大等待10秒

        if last_error:
            logger.info(f"在第 {retry_count} 次重试后成功完成翻译")

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

    async def detect_enhanced(self, text: str) -> str:
        """
        Enhanced language detection with more accuracy.

        Args:
            text: The text to detect language for

        Returns:
            Detected language code
        """
        # 增强的语言检测实现
        return "auto"

    async def _detect_language_with_fallback(self, text: str, fallback_lang: str = "auto") -> str:
        """
        Detect the language of the text with fallback options.

        Args:
            text: The text to detect language for
            fallback_lang: Fallback language code if detection fails

        Returns:
            Detected language code or fallback
        """
        if not text:
            return fallback_lang

        try:
            detected_lang = await self.detect(text)
            return detected_lang
        except Exception as e:
            logging.warning(f"Language detection failed: {str(e)}")
            try:
                detected_lang = await self.detect_enhanced(text)
                return detected_lang
            except Exception as e:
                logging.error(f"Enhanced detection also failed: {str(e)}")
                return fallback_lang

    async def translate(self, text: Union[str, List[str]], dest='en', src='auto', stream=False) -> Union[AITranslated, List[AITranslated], AsyncGenerator[AITranslated, None]]:
        """翻译文本，支持批量处理和流式翻译

        Args:
            text: 要翻译的源文本（字符串或字符串列表）
            dest: 目标语言
            src: 源语言
            stream: 是否使用流式翻译

        Returns:
            翻译结果或生成器
        """
        # 验证语言代码
        if src != 'auto' and src not in DOUBAO_LANGUAGES:
            raise AIError(f"不支持的源语言: {src}")
        if dest not in DOUBAO_LANGUAGES:
            raise AIError(f"不支持的目标语言: {dest}")

        if isinstance(text, list):
            if stream:
                raise ValueError("流式翻译不支持批量处理")
            result = await self.translate_batch(text, dest, src)
            return result

        text = text.strip()
        if not text:
            return AITranslated(src, dest, text, text)

        if stream:
            current_src = src if src != 'auto' else await self._detect_language_with_fallback(text)
            return StreamTranslator(self, text, dest, current_src)
        else:
            result = await self._translate_single(text, dest, src, False)
            return result

    async def translate_with_context(self, text: str, context: str, dest='en', src='auto', style_guide=None, stream=False) -> Union[AITranslated, AsyncGenerator[AITranslated, None]]:
        """
        带上下文的翻译，支持风格指南和一致性控制

        Args:
            text: 要翻译的文本
            context: 上下文信息
            dest: 目标语言
            src: 源语言（auto为自动检测）
            style_guide: 风格指南（可选）
            stream: 是否使用流式翻译

        Returns:
            AITranslated对象或生成器（流式翻译时）
        """
        try:
            text = text.strip()
            if not text:
                return AITranslated(src, dest, text, text)

            # 1. 如果语言auto，先进行语言检测
            current_src = src if src != 'auto' else await self._detect_language_with_fallback(text)

            # 2. 构建提示词
            style_instructions = f"\n\n风格要求：\n{style_guide}" if style_guide else ""
            prompt = f"""请在理解以下上下文的基础上，将文本从{current_src}翻译成{dest}：

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

            # 3. 应用术语表（如果有）
            if self.glossary:
                try:
                    logger.debug(
                        "Applying glossary before context translation")
                    glossary_result = await self.apply_glossary(text, current_src, dest)
                    text = glossary_result.text
                except Exception as e:
                    logger.warning(f"Glossary application failed: {str(e)}")

            # 4. 发送翻译请求
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            if not stream:
                translated_text = await self._make_request(messages, stream=False)
                logger.info(
                    f"Context-aware translation completed for text length: {len(text)}")
                logger.debug(f"Translation context length: {len(context)}")
                return AITranslated(
                    src=current_src,
                    dest=dest,
                    origin=text,
                    text=translated_text
                )

            return await self._create_stream_generator(messages, current_src, dest, text)

        except Exception as e:
            logger.error(f"Context-aware translation failed: {str(e)}")
            raise Exception(f"上下文感知翻译失败: {str(e)}")

    async def translate_with_style(self, text: str, dest: str = 'en', src: str = 'auto',
                                   style: Union[str, Dict] = 'formal', context: str = None,
                                   max_versions: int = 3, stream: bool = False) -> Union[AITranslated, AsyncGenerator[AITranslated, None]]:
        """
        带风格的翻译

        Args:
            text: 要翻译的文本
            dest: 目标语言
            src: 源语言
            style: 翻译风格，可以是预定义风格的名称或自定义风格字典
            context: 上下文信息
            max_versions: 创意风格时的最大版本数量，默认3个
            stream: 是否使用流式翻译

        Returns:
            AITranslated对象或生成器（流式翻译时）
        """
        start_time = time.time()
        try:
            text = text.strip()
            if not text:
                return AITranslated(src, dest, text, text)

            # 1. 如果语言auto，先进行语言检测
            current_src = src if src != 'auto' else await self._detect_language_with_fallback(text)

            # 2. 构建提示信息
            if isinstance(style, str):
                if style not in self.style_templates:
                    raise ValueError(f"未知的预定义风格: {style}")
                style_prompt = self.style_templates[style]
                if style == 'creative':
                    style_prompt = f"""
                    翻译要求：
                    1. 提供{max_versions}个不同的翻译版本
                    2. 每个版本使用不同的表达方式
                    3. 保持原文的核心含义
                    4. 限制在{max_versions}个版本以内
                    """
            else:
                style_prompt = "翻译要求：\n" + \
                    "\n".join([f"{k}: {v}" for k, v in style.items()])

            # 添加上下文信息（如果有）
            context_part = f"\n相关上下文：\n{context}\n" if context else ""

            messages = [
                {"role": "system", "content": "你是一个专业的翻译手，请按照指定的风格要求进行翻译。"},
                {"role": "user",
                    "content": f"""
{style_prompt}

{context_part}
需要翻译的文本：
{text}

请直接提供翻译结果，不要添加任何解释。如果是创意风格，最多提供{max_versions}个不同的版本，用分号分隔。
"""}
            ]

            if not stream:
                translated_text = await self._make_request(messages, stream=False)
                duration = time.time() - start_time
                logger.info(
                    f"Styled translation completed in {duration:.2f}s - Style: {style}, Length: {len(text)}, Languages: {current_src}->{dest}")
                return AITranslated(
                    src=current_src,
                    dest=dest,
                    origin=text,
                    text=translated_text
                )

            return await self._create_stream_generator(messages, current_src, dest, text)

        except Exception as e:
            logger.error(f"Style translation failed: {str(e)}")
            raise Exception(f"风格化翻译失败: {str(e)}")

    async def evaluate_translation(self, original: str, translated: str, src: str, dest: str) -> Dict[str, float]:
        """
        评估翻译质量

        :return: 包含各项指标的字典
        """
        prompt = f"""请评估以下翻译的质量，给出0-1的分数：

原文 ({src}): {original}
译文 ({dest}): {translated}

请从以下几个方面评分：
1. 准确性：内容是否准确传达
2. 流畅性：是否自流畅
3. 专业性：专业术语使用是否恰当
4. 风格：是否保持原文风格

只返回JSON式的评分结果。"""

        try:
            messages = [
                {"role": "system", "content": "你是翻译质量评估专家"},
                {"role": "user", "content": prompt}
            ]

            response = await self._make_request(messages)
            scores = json.loads(response.choices[0].message.content)
            return scores

        except Exception as e:
            logger.error(f"Translation evaluation failed: {str(e)}")
            return {
                "accuracy": 0.0,
                "fluency": 0.0,
                "professionalism": 0.0,
                "style": 0.0
            }

    @lru_cache(maxsize=1000)
    def _get_cached_translation(self, text: str, dest: str, src: str, style: str = None) -> str:
        """获取缓存的翻译结果"""
        cache_key = f"{text}:{src}:{dest}:{style}"
        return cache_key

    def set_performance_config(self, **kwargs):
        """
        动态更新性能配置

        :param kwargs: 要更新的性能参数
        """
        self.perf_config.update(kwargs)
        # 更新相关实例变量
        if 'cache_ttl' in kwargs:
            self._cache_ttl = kwargs['cache_ttl']
        if 'min_request_interval' in kwargs:
            self._min_request_interval = kwargs['min_request_interval']
        if 'max_retries' in kwargs:
            self.max_retries = kwargs['max_retries']

    async def test_connection(self) -> bool:
        """测试API连接和认证"""
        try:
            response = await self._make_request([
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"}
            ])
            logger.info("API连接测试成功")
            return True
        except Exception as e:
            logger.error(f"API连接测试失败: {str(e)}")
            return False

    def get_config(self) -> dict:
        """
        获取当前配置信息

        :return: 包含所有配置信息的字典
        """
        return {
            'api_key': f"{self.api_key[:8]}...",  # 只显示前8位
            'base_url': self.base_url,
            'model': self.model,
            'performance_config': self.perf_config,
            'cache_ttl': self._cache_ttl,
            'min_request_interval': self._min_request_interval,
            'max_retries': self.max_retries
        }

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置参数"""
        required_fields = {
            'max_retries': (int, lambda x: x > 0),
            'cache_ttl': (int, lambda x: x > 0),
            'max_tokens': (int, lambda x: 0 < x <= 2048),
            'temperature': (float, lambda x: 0 <= x <= 1),
        }

        for field, (field_type, validator) in required_fields.items():
            value = config.get(field)
            if not isinstance(value, field_type):
                raise ValueError(
                    f"{field} must be of type {field_type.__name__}")
            if not validator(value):
                raise ValueError(f"Invalid value for {field}: {value}")

    def _get_retry_config(self):
        """获取更智能的重试配置"""
        return {
            'multiplier': self.perf_config.get('retry_multiplier', 0.5),
            'min': self.perf_config.get('retry_min_wait', 1),
            'max': self.perf_config.get('retry_max_wait', 4),
            'max_retries': self.perf_config.get('max_retries', 3),
            'retry_on_errors': {
                'network': True,      # 网络错误
                'rate_limit': True,   # 速率限制
                'server': True,       # 服务器错误
                'timeout': True,      # 超时
                'auth': False         # 认证错误不重试
            }
        }

    async def _should_retry(self, error: Exception, attempt: int) -> Tuple[bool, float]:
        """判断是否应该重试并返回等待时间"""
        retry_config = self._get_retry_config()

        if attempt >= retry_config['max_retries']:
            return False, 0

        if isinstance(error, openai.RateLimitError):
            if not retry_config['retry_on_errors']['rate_limit']:
                return False, 0
            wait_time = getattr(error, 'retry_after', 30)

        elif isinstance(error, (httpx.ConnectError, httpx.ReadTimeout)):
            if not retry_config['retry_on_errors']['network']:
                return False, 0
            wait_time = min(retry_config['multiplier'] * (2 ** attempt),
                            retry_config['max'])

        elif isinstance(error, openai.APIError):
            if not retry_config['retry_on_errors']['server']:
                return False, 0
            wait_time = min(retry_config['multiplier'] * (2 ** attempt),
                            retry_config['max'])

        else:
            return False, 0

        return True, max(wait_time, retry_config['min'])

    async def __aenter__(self):
        """异步上下文管理器入口"""
        try:
            await self.session_manager.initialize()
            await self.client_manager.initialize()
            return self
        except Exception as e:
            logger.error(f"Failed to initialize resources: {e}")
            await self.__aexit__(type(e), e, e.__traceback__)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口，确保清理预连接资源"""
        try:
            await self._cleanup_preconnection()
            await self.session_manager.cleanup()
            await self.client_manager.cleanup()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            if exc_type is None:
                raise

    async def _ensure_session(self):
        """确保会话已初始化"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def _cleanup_resources(self):
        """清理所有资源"""
        try:
            if hasattr(self, 'client'):
                await self.client.close()

            if hasattr(self, 'session') and self.session:
                if not self.session.closed:
                    await self.session.close()
                self.session = None

            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)

            # 清理缓存
            self._response_cache.clear()

            logger.debug("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")

    async def preconnect(self):
        """
        预连接方法，提前初始化所有必要的资源。
        如果预连接成功，后续的翻译请求将直接使用已建立的连接。

        Returns:
            bool: 预连接是否成功
        """
        try:
            if self._is_preconnected:
                return True

            # 初始化会话
            self._preconnected_session = aiohttp.ClientSession()

            # 初始化客户端
            self._preconnected_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            # 发送测试请求验证连接
            test_messages = [
                {"role": "system", "content": "test connection"},
                {"role": "user", "content": "test"}
            ]

            async with self.semaphore:
                try:
                    await self._preconnected_client.chat.completions.create(
                        model=self.model,
                        messages=test_messages,
                        max_tokens=10
                    )
                except Exception as e:
                    await self._cleanup_preconnection()
                    raise AIConnectionError(f"预连接测试失败: {str(e)}")

            self._is_preconnected = True
            logger.info("预连接成功建立")
            return True

        except Exception as e:
            logger.error(f"预连接失败: {str(e)}")
            await self._cleanup_preconnection()
            return False

    async def _cleanup_preconnection(self):
        """清理预连接资源"""
        try:
            if self._preconnected_session:
                await self._preconnected_session.close()
            self._preconnected_session = None
            self._preconnected_client = None
            self._is_preconnected = False
        except Exception as e:
            logger.error(f"清理预连接资源时出错: {str(e)}")

    async def cleanup(self):
        """清理所有资源"""
        try:
            # 清理会话管理器
            if hasattr(self, 'session_manager') and self.session_manager:
                await self.session_manager.cleanup()

            # 清理客户端管理器
            if hasattr(self, 'client_manager') and self.client_manager:
                await self.client_manager.cleanup()

            # 清理批处理器
            if hasattr(self, 'batch_processor'):
                await self.batch_processor.stop()

            # 清理缓存
            if hasattr(self, '_response_cache'):
                self._response_cache.clear()
            if hasattr(self, '_context_cache'):
                self._context_cache.clear()

            logger.info("AITranslator resources cleaned up successfully")

        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            raise AIError(f"清理资源失败: {str(e)}")

    async def translate_with_recovery(self, text: str, dest='en', src='auto',
                                      progress_callback=None) -> AITranslated:
        """带恢复机制的翻译方法"""
        recovery_state = TranslationRecoveryState()

        while recovery_state.attempts < self.max_retries:
            try:
                if progress_callback:
                    progress_callback(recovery_state.attempts, "正在尝试翻译...")

                result = await self.translate(text, dest, src)
                return result

            except Exception as e:
                recovery_state.record_attempt(e)
                should_retry, wait_time = await self._should_retry(e, recovery_state.attempts)

                if not should_retry:
                    raise AIError(f"翻译失败，无法恢复: {str(e)}")

                if progress_callback:
                    progress_callback(recovery_state.attempts,
                                      f"遇到错误，{wait_time}秒后重试...")

                await asyncio.sleep(wait_time)

        raise AIError(f"达到最大重试次数 ({self.max_retries})，翻译失败")


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

    async def __anext__(self):
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

                # 检查chunk是否有效并包含内容
                if (hasattr(chunk, 'choices') and
                    chunk.choices and
                    hasattr(chunk.choices[0], 'delta') and
                    hasattr(chunk.choices[0].delta, 'content') and
                        chunk.choices[0].delta.content):

                    # 累积文本
                    content = chunk.choices[0].delta.content
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

                # 如果chunk无效，继续获取下一个
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
            raise AIError(f"流式翻译错误: {str(e)}")

    def _cleanup(self):
        """清理资源"""
        self._response_stream = None
        self._accumulated_text = ""
        self._is_first_chunk = True
        self._translation_complete = True
        self._request_sent = False


class AITranslatorSync:
    """AITranslator的同步封装类，使异步API更易使用"""

    def __init__(self, api_key=None, model_name=None, base_url=None,
                 max_workers=MAX_WORKERS, glossary_path=None,
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

    @staticmethod
    def quick_translate(text: str, dest: str = 'en', src: str = 'auto') -> str:
        """
        快速翻译方法，无需手动创建实例

        Args:
            text: 要翻译的文本
            dest: 目标语言代码
            src: 源语言代码，默认自动检测

        Returns:
            str: 翻译后的文本

        Example:
            result = AITranslatorSync.quick_translate("你好", dest="en")
            print(result)  # 输出: "Hello"
        """
        try:
            with AITranslatorSync() as translator:
                result = translator.translate(text, dest=dest, src=src)
                return result.text if result else text
        except Exception as e:
            logger.error(f"快速翻译失败: {str(e)}")
            return text

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
        try:
            if hasattr(self, 'translator') and self.translator:
                await self.translator.cleanup()
        except Exception as e:
            logger.error(f"资源清理失败: {str(e)}")

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

    def test_connection(self):
        """测试连接"""
        self._ensure_translator()
        return self._loop.run_until_complete(self.translator.test_connection())

    def evaluate_translation(self, original, translated, src, dest):
        """评估翻译质量"""
        self._ensure_translator()
        return self._loop.run_until_complete(
            self.translator.evaluate_translation(
                original, translated, src, dest)
        )

    def get_config(self):
        """获取配置信息"""
        self._ensure_translator()
        return self.translator.get_config()

    def set_performance_config(self, **kwargs):
        """设置性能配置"""
        self._ensure_translator()
        self.translator.set_performance_config(**kwargs)

    def __del__(self):
        """析构时确保资源被清理"""
        if self._initialized:
            self.__exit__(None, None, None)


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
        auto_preconnect=True,
        **kwargs
    ).__enter__()

    # 自动预热
    try:
        translator.preconnect()
    except Exception as e:
        logger.warning(f"预热失败，但不影响使用: {str(e)}")

    return translator


def test_examples():
    """展示各种使用场景的示例"""

    print("\n===== 1. 快速翻译（一行代码）=====")
    try:
        # 最简单的使用方式
        result = AITranslatorSync.quick_translate("你好世界", dest="en")
        print(f"快速翻译结果: {result}")
    except Exception as e:
        print(f"快速翻译失败: {e}")

    print("\n===== 2. 使用工厂方法 =====")
    try:
        # 使用工厂方法创建预热好的实例
        translator = create(performance_mode='fast')
        result = translator.translate("这是一个测试", dest="en")
        print(f"工厂方法创建的翻译器结果: {result.text}")

        # 获取性能指标
        metrics = translator.get_metrics()
        print("性能指标:", json.dumps(metrics, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"工厂方法测试失败: {e}")

    print("\n===== 3. 上下文管理器 =====")
    try:
        # 使用 with 语句自动管理资源
        with create() as translator:
            result = translator.translate("自动管理资源", dest="en")
            print(f"上下文管理器翻译结果: {result.text}")
    except Exception as e:
        print(f"上下文管理器测试失败: {e}")

    print("\n===== 4. 流式翻译 =====")
    try:
        with create() as translator:
            text = "这是一个很长的句子，用来测试流式翻译功能。"
            print(f"原文: {text}")
            print("翻译过程:")
            for partial_result in translator.translate(text, dest="en", stream=True):
                print(f"\r当前结果: {partial_result.text}", end="", flush=True)
            print()  # 换行
    except Exception as e:
        print(f"流式翻译测试失败: {e}")

    print("\n===== 5. 批量翻译 =====")
    try:
        with create() as translator:
            texts = [
                "第一句话",
                "第二句话",
                "第三句话"
            ]
            results = translator.translate_batch_sync(texts, dest="en")
            print("批量翻译结果:")
            for text, result in zip(texts, results):
                print(f"{text} -> {result.text}")
    except Exception as e:
        print(f"批量翻译测试失败: {e}")

    print("\n===== 6. 风格化翻译 =====")
    try:
        with create() as translator:
            text = "这个产品非常好用"
            print(f"原文: {text}")

            # 测试不同风格
            styles = ['formal', 'casual', 'creative']
            for style in styles:
                result = translator.translate_with_style(
                    text, dest="en", style=style
                )
                print(f"{style} 风格: {result.text}")
    except Exception as e:
        print(f"风格化翻译测试失败: {e}")

    print("\n===== 7. 带上下文的翻译 =====")
    try:
        with create() as translator:
            context = "这是一篇关于人工智能的文章。"
            text = "它的发展非常迅速。"
            result = translator.translate_with_context(
                text, context=context, dest="en"
            )
            print(f"上下文: {context}")
            print(f"原文: {text}")
            print(f"翻译结果: {result.text}")
    except Exception as e:
        print(f"上下文翻译测试失败: {e}")

    print("\n===== 8. 性能模式测试 =====")
    try:
        # 测试不同性能模式
        modes = ['fast', 'balanced', 'accurate']
        text = "这是一个测试句子"
        print(f"原文: {text}")

        for mode in modes:
            with create(performance_mode=mode) as translator:
                start_time = time.time()
                result = translator.translate(text, dest="en")
                duration = time.time() - start_time
                print(f"{mode} 模式:")
                print(f"结果: {result.text}")
                print(f"耗时: {duration:.3f}秒")
    except Exception as e:
        print(f"性能模式测试失败: {e}")

    print("\n===== 9. 错误处理测试 =====")
    try:
        with create() as translator:
            # 测试无效的语言代码
            try:
                result = translator.translate("测试", dest="invalid_lang")
            except AIError as e:
                print(f"预期的错误处理 (无效语言): {e}")

            # 测试空文本
            result = translator.translate("", dest="en")
            print(f"空文本处理: {result.text}")

            # 测试超长文本
            long_text = "测试" * 1000
            try:
                result = translator.translate(long_text, dest="en")
                print(f"长文本处理成功，结果长度: {len(result.text)}")
            except Exception as e:
                print(f"长文本处理异常: {e}")
    except Exception as e:
        print(f"错误处理测试失败: {e}")


# ============= 使用示例和测试代码 =============

async def test_basic_translation():
    """基础翻译功能测试"""
    translator = AITranslator()
    try:
        # 单文本翻译测试
        result = await translator.translate("Hello world!", dest="zh")
        print(f"基础翻译测试: {result.text}")

        # 批量翻译测试
        texts = ["Hello", "Good morning", "Good night"]
        results = await translator.translate_batch(texts, dest="zh")
        print("\n批量翻译测试:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.text}")

    except AIError as e:
        print(f"翻译错误: {str(e)}")
    finally:
        await translator.cleanup()


async def test_error_recovery():
    """测试翻译错误恢复功能"""
    translator = AITranslator()
    text = "这是一个测试文本。" * 5  # 重复文本以增加处理时间

    # 模拟网络错误
    translator.api_base = "https://nonexistent-api.example.com"

    try:
        print("\n开始错误恢复测试...")
        result = await translator.translate(text, dest='en', src='zh')
        print(f"翻译结果: {result.text}")
        print("测试失败: 应该抛出异常")
    except AIError as e:
        print(f"预期的错误: {str(e)}")
        # 恢复正确的API地址
        translator.api_base = "https://api.openai.com/v1"

        try:
            # 重试翻译
            result = await translator.translate(text, dest='en', src='zh')
            print(f"恢复后的翻译: {result.text}")
            print("错误恢复测试通过!")
        except Exception as e:
            print(f"恢复失败: {str(e)}")
            raise


async def test_stream_translation():
    """流式翻译测试"""
    translator = AITranslator()
    try:
        text = "This is a test text for streaming translation."
        print(f"\n开始流式翻译测试: '{text}'")

        stream_translator = await translator.translate(text, dest="zh", stream=True)

        async for partial in stream_translator:
            status = partial.metadata.get('status', '')
            is_partial = partial.metadata.get('is_partial', True)

            if is_partial:
                print(f"部分译文: {partial.text}")
            else:
                print(f"翻译完成！最终结果: {partial.text}")
                break  # 收到完整结果后退出

    except AIError as e:
        print(f"流式翻译错误: {str(e)}")
    finally:
        await translator.cleanup()


async def test_batch_with_progress():
    """批量翻译进度测试"""
    translator = AITranslator()
    try:
        texts = [f"第{i}段测试文本。" for i in range(1, 6)]
        print("\n开始批量翻译测试...")

        def progress_callback(progress: TranslationProgress):
            """进度回调函数"""
            print(f"进度: {progress.percent:.1f}% - "
                  f"完成: {progress.completed}/{progress.total} - "
                  f"当前文本: {progress.current_text}")

        results = await translator.translate_batch(
            texts,
            dest="en",
            progress_callback=progress_callback
        )

        print("\n翻译结果:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.text}")

        print("批量翻译测试通过!")
        return True

    except Exception as e:
        print(f"批量翻译测试失败: {str(e)}")
        return False
    finally:
        await translator.cleanup()


async def test_error_handling():
    """测试各种错误情况的处理"""
    translator = AITranslator()

    print("\n开始错误处理测试...")

    # 测试空文本
    try:
        result = await translator.translate("", dest='en')
        assert result.text == "", "空文本应该返回空结果"
        print("✓ 空文本测试通过")
    except Exception as e:
        print(f"✗ 空文本测试失败: {str(e)}")

    # 测试无效的源语言
    try:
        await translator.translate("测试文本", src='invalid', dest='en')
        print("✗ 无效源语言测试失败: 应该抛出异常")
    except AIError as e:
        if "不支持的源语言" in str(e):
            print("✓ 无效源语言测试通过")
        else:
            print(f"✗ 无效源语言测试失败: {str(e)}")

    # 测试无效的目标语言
    try:
        await translator.translate("测试文本", src='zh', dest='invalid')
        print("✗ 无效目标语言测试失败: 应该抛出异常")
    except AIError as e:
        if "不支持的目标语言" in str(e):
            print("✓ 无效目标语言测试通过")
        else:
            print(f"✗ 无效目标语言测试失败: {str(e)}")

    # 测试流式翻译的批量处理
    try:
        await translator.translate(["文本1", "文本2"], stream=True)
        print("✗ 流式批量测试失败: 应该抛出异常")
    except ValueError as e:
        if "流式翻译不支持批量处理" in str(e):
            print("✓ 流式批量测试通过")
        else:
            print(f"✗ 流式批量测试失败: {str(e)}")

    print("错误处理测试完成")


async def run_tests():
    """运行所有测试用例"""
    test_functions = [
        test_basic_translation,
        test_error_recovery,
        test_stream_translation,
        test_batch_with_progress,
        test_error_handling
    ]

    for test_func in test_functions:
        print(f"\n运行测试: {test_func.__name__}")
        print("=" * 50)
        try:
            await test_func()
            print(f"✓ {test_func.__name__} 测试通过")
        except Exception as e:
            print(f"✗ {test_func.__name__} 测试失败: {str(e)}")
        print("=" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_tests())
