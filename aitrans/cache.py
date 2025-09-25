"""简单的多级缓存实现。"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheItem:
    value: Any
    expire_at: float
    metadata: Dict[str, Any]


class MultiLevelCache:
    """以内存为主、可选文件落地的简单缓存。"""

    def __init__(
        self,
        memory_size: int = 10000,
        file_cache_path: Optional[Path] = None,
        default_ttl: int = 3600,
    ) -> None:
        self._memory_cache: Dict[str, CacheItem] = {}
        self._memory_size = memory_size
        self._file_cache_path = file_cache_path
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

        if file_cache_path:
            file_cache_path.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, text: str, from_lang: str, to_lang: str, **kwargs: Any) -> str:
        key_parts = [text[:100], from_lang, to_lang]
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
        return "::".join(key_parts)

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            item = self._memory_cache.get(key)
            if item and item.expire_at > time.time():
                return item.value
            if item:
                del self._memory_cache[key]

        if not self._file_cache_path:
            return None

        cache_file = self._file_cache_path / f"{key}.json"
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache_file.unlink(missing_ok=True)
            return None

        if data.get("expire_at", 0) <= time.time():
            cache_file.unlink(missing_ok=True)
            return None

        value = data.get("value")
        metadata = data.get("metadata", {})
        async with self._lock:
            self._memory_cache[key] = CacheItem(value, data["expire_at"], metadata)
        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        expire_at = time.time() + (ttl if ttl is not None else self._default_ttl)
        cache_item = CacheItem(value, expire_at, metadata or {})

        async with self._lock:
            if len(self._memory_cache) >= self._memory_size:
                oldest_key = min(self._memory_cache, key=lambda k: self._memory_cache[k].expire_at)
                self._memory_cache.pop(oldest_key, None)
            self._memory_cache[key] = cache_item

        if not self._file_cache_path:
            return

        cache_file = self._file_cache_path / f"{key}.json"
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "value": value,
                        "expire_at": expire_at,
                        "metadata": cache_item.metadata,
                    }
                )
            )
        except TypeError:
            cache_file.unlink(missing_ok=True)

    async def clear(self) -> None:
        async with self._lock:
            self._memory_cache.clear()

        if not self._file_cache_path:
            return

        for file in self._file_cache_path.glob("*.json"):
            file.unlink(missing_ok=True)


class SyncMultiLevelCache:
    """用于同步环境的简易包装。"""

    def __init__(self, async_cache: MultiLevelCache) -> None:
        self._async_cache = async_cache
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def get(self, key: str) -> Optional[Any]:
        loop = self._ensure_loop()
        return loop.run_until_complete(self._async_cache.get(key))

    def set(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        loop = self._ensure_loop()
        loop.run_until_complete(self._async_cache.set(key, value, ttl, metadata))

    def clear(self) -> None:
        if not self._loop:
            return
        try:
            self._loop.run_until_complete(self._async_cache.clear())
        finally:
            self._loop.close()
            self._loop = None
