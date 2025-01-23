"""工具类模块"""

import asyncio
import logging
import time
from typing import Dict, Any, Callable
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


class BatchProcessor:
    """处理批量任务"""

    def __init__(self, max_workers: int = 5, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_workers)
        self._is_initialized = False

    async def initialize(self):
        """初始化处理器"""
        if not self._is_initialized:
            self._is_initialized = True
            logger.debug("BatchProcessor initialized")

    async def process(self, items: list, processor_func) -> list:
        """处理批量任务"""
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

    def get_current_batch(self) -> list:
        """获取当前批次并清空"""
        batch = self.current_batch[:]
        self.current_batch = []
        self.current_tokens = 0
        return batch

    async def process_batch(self, processor_func: Callable) -> list:
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

    def should_retry(self, error: Exception) -> tuple[bool, float]:
        """判断是否应该重试并返回等待时间"""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1

        if self.error_counts[error_type] > self.max_retries:
            return False, 0

        if isinstance(error, (Exception, ConnectionError)):
            wait_time = min(2 ** self.error_counts[error_type], 32)
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

    async def get(self, key: str) -> Any:
        """获取缓存值"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    self.hits += 1
                    return entry['value']
            self.misses += 1
            return None

    async def put(self, key: str, value: Any):
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


class PerformanceMetrics:
    """跟踪性能指标"""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.total_duration = 0.0
        self.min_duration = float('inf')
        self.max_duration = 0.0
        self._lock = asyncio.Lock()
        self._start_time = time.time()

    def record_request(self, duration: float, success: bool):
        """记录请求性能指标"""
        async def _record():
            async with self._lock:
                self.total_requests += 1
                if success:
                    self.successful_requests += 1
                    self.total_duration += duration
                    self.min_duration = min(self.min_duration, duration)
                    self.max_duration = max(self.max_duration, duration)

        asyncio.create_task(_record())

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
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
        async def _reset():
            async with self._lock:
                self.total_requests = 0
                self.successful_requests = 0
                self.total_duration = 0.0
                self.min_duration = float('inf')
                self.max_duration = 0.0
                self._start_time = time.time()

        asyncio.create_task(_reset())
