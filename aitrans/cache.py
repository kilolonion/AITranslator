from typing import Optional, Any, Dict, Union
import json
import asyncio
from pathlib import Path
import time
from dataclasses import dataclass


@dataclass
class CacheItem:
    """缓存项"""
    value: Any
    expire_at: float
    metadata: dict


class MultiLevelCache:
    """多级缓存实现"""

    def __init__(
        self,
        memory_size: int = 10000,
        file_cache_path: Optional[Path] = None,
        default_ttl: int = 3600
    ):
        self._memory_cache: Dict[str, CacheItem] = {}
        self._file_cache_path = file_cache_path
        self._default_ttl = default_ttl
        self._memory_size = memory_size
        self._lock = asyncio.Lock()

        # 创建文件缓存目录
        if file_cache_path:
            file_cache_path.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, text: str, from_lang: str, to_lang: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [
            text[:100],  # 限制文本长度
            from_lang,
            to_lang
        ]
        # 添加额外参数到键中
        for k, v in sorted(kwargs.items()):
            if k in ['style', 'model', 'provider']:
                key_parts.append(f"{k}:{v}")

        return ":".join(key_parts)

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        # 1. 检查内存缓存
        if item := self._memory_cache.get(key):
            if time.time() < item.expire_at:
                return item.value
            else:
                del self._memory_cache[key]

        # 2. 检查文件缓存
        if self._file_cache_path:
            cache_file = self._file_cache_path / f"{key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    if time.time() < data['expire_at']:
                        # 提升到内存缓存
                        await self.set(key, data['value'],
                                       ttl=data['expire_at'] - time.time(),
                                       metadata=data.get('metadata', {}))
                        return data['value']
                    else:
                        cache_file.unlink()
                except Exception:
                    pass

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  metadata: Optional[dict] = None) -> None:
        """设置缓存值"""
        async with self._lock:
            expire_at = time.time() + (ttl or self._default_ttl)
            metadata = metadata or {}

            # 1. 更新内存缓存
            if len(self._memory_cache) >= self._memory_size:
                # 移除最旧的项
                oldest_key = min(self._memory_cache.keys(),
                                 key=lambda k: self._memory_cache[k].expire_at)
                del self._memory_cache[oldest_key]

            self._memory_cache[key] = CacheItem(value, expire_at, metadata)

            # 2. 更新文件缓存
            if self._file_cache_path:
                cache_file = self._file_cache_path / f"{key}.json"
                try:
                    cache_data = {
                        'value': value,
                        'expire_at': expire_at,
                        'metadata': metadata
                    }
                    cache_file.write_text(json.dumps(cache_data))
                except Exception:
                    pass

    async def clear(self) -> None:
        """清理所有缓存"""
        async with self._lock:
            self._memory_cache.clear()
            if self._file_cache_path:
                for cache_file in self._file_cache_path.glob("*.json"):
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass


class SyncMultiLevelCache:
    """同步多级缓存"""

    def __init__(self, async_cache: MultiLevelCache):
        self._async_cache = async_cache
        self._loop = None

    def _ensure_loop(self):
        """确保事件循环可用"""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def get(self, key: str) -> Optional[Any]:
        """同步获取缓存"""
        loop = self._ensure_loop()
        return loop.run_until_complete(
            self._async_cache.get(key)
        )

    def set(self, key: str, value: Any, ttl: Optional[int] = None,
            metadata: Optional[dict] = None) -> None:
        """同步设置缓存"""
        loop = self._ensure_loop()
        loop.run_until_complete(
            self._async_cache.set(key, value, ttl, metadata)
        )

    def clear(self) -> None:
        """同步清理缓存"""
        if self._loop:
            try:
                self._loop.run_until_complete(self._async_cache.clear())
            finally:
                if not self._loop.is_closed():
                    self._loop.close()
                self._loop = None
