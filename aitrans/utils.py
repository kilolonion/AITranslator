"""工具类与辅助功能。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class BatchProcessor:
    """非常轻量的批量处理器。"""

    def __init__(self, max_workers: int = 5, batch_size: int = 10) -> None:
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._is_initialized = False

    async def initialize(self) -> None:
        self._is_initialized = True

    async def process(self, items: List[Any], processor: Callable[[Any], Any]) -> List[Any]:
        if not self._is_initialized:
            await self.initialize()

        results = []
        for item in items:
            results.append(await processor(item))
        return results

    async def stop(self) -> None:
        self._is_initialized = False


class SmartBatchProcessor:
    """基于简单阈值的批处理实现。"""

    def __init__(self, max_batch_size: int = 10, max_tokens: int = 4000) -> None:
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.current_batch: List[str] = []
        self.current_tokens = 0

    async def initialize(self) -> None:  # pragma: no cover - 兼容旧接口
        return None

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    def add_to_batch(self, text: str) -> bool:
        tokens = self.estimate_tokens(text)
        if len(self.current_batch) >= self.max_batch_size or self.current_tokens + tokens > self.max_tokens:
            return False
        self.current_batch.append(text)
        self.current_tokens += tokens
        return True

    def get_current_batch(self) -> List[str]:
        batch = self.current_batch[:]
        self.current_batch.clear()
        self.current_tokens = 0
        return batch

    async def process_batch(self, processor: Callable[[List[str]], Any]) -> Any:
        if not self.current_batch:
            return []
        batch = self.get_current_batch()
        return await processor(batch)


class DynamicRateLimiter:
    """简单的基于速率的限流器。"""

    def __init__(self, initial_rate: float = 10.0, max_rate: float = 20.0) -> None:
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.success_count = 0
        self.failure_count = 0
        self.last_adjustment = time.time()
        self.window_size = 60.0

    async def adjust_rate(self, success: bool) -> None:
        now = time.time()
        if now - self.last_adjustment > self.window_size:
            self.success_count = 0
            self.failure_count = 0
            self.last_adjustment = now

        if success:
            self.success_count += 1
            if self.success_count > 10:
                self.current_rate = min(self.current_rate * 1.2, self.max_rate)
        else:
            self.failure_count += 1
            if self.failure_count > 2:
                self.current_rate = max(self.current_rate * 0.8, 1.0)

    async def wait(self) -> None:
        await asyncio.sleep(1.0 / max(self.current_rate, 1.0))


class SmartRetryHandler:
    """基于错误类型的简单重试逻辑。"""

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries
        self.error_counts: Dict[str, int] = defaultdict(int)

    def should_retry(self, error: Exception) -> tuple[bool, float]:
        name = type(error).__name__
        self.error_counts[name] += 1
        if self.error_counts[name] > self.max_retries:
            return False, 0.0
        return True, min(2 ** self.error_counts[name], 10)


class PerformanceMetrics:
    """记录简单的性能指标。"""

    def __init__(self) -> None:
        self.total_requests = 0
        self.successful_requests = 0
        self.total_duration = 0.0
        self.min_duration = float("inf")
        self.max_duration = 0.0
        self.start_time = time.time()

    def record_request(self, duration: float, success: bool) -> None:
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_duration += duration
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)

    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "requests_per_second": 0.0,
                "uptime_seconds": uptime,
            }

        average_duration = (
            self.total_duration / self.successful_requests if self.successful_requests else 0.0
        )
        min_duration = 0.0 if self.min_duration == float("inf") else self.min_duration
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests,
            "average_duration": average_duration,
            "min_duration": min_duration,
            "max_duration": self.max_duration,
            "requests_per_second": self.total_requests / max(uptime, 1e-6),
            "uptime_seconds": uptime,
        }

    def reset(self) -> None:
        self.total_requests = 0
        self.successful_requests = 0
        self.total_duration = 0.0
        self.min_duration = float("inf")
        self.max_duration = 0.0
        self.start_time = time.time()
