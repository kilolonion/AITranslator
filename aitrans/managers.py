from typing import Dict, Optional, Any
import aiohttp
import asyncio
from contextlib import asynccontextmanager


class ClientManager:
    """连接池管理器"""

    def __init__(
        self,
        pool_size: int = 100,
        keepalive_timeout: int = 30,
        ttl_dns_cache: int = 300,
        enable_tcp_keepalive: bool = True,
        api_key: str = None
    ):
        self._pools: Dict[str, aiohttp.ClientSession] = {}
        self._config = {
            "connector": aiohttp.TCPConnector(
                limit=pool_size,
                ttl_dns_cache=ttl_dns_cache,
                enable_cleanup_closed=True,
                keepalive_timeout=keepalive_timeout
            ),
            "timeout": aiohttp.ClientTimeout(total=30),
            "headers": {
                "Connection": "keep-alive",
                "Keep-Alive": str(keepalive_timeout),
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        }
        self._enable_tcp_keepalive = enable_tcp_keepalive

    @asynccontextmanager
    async def get_session(self, service_name: str) -> aiohttp.ClientSession:
        """获取或创建会话"""
        if service_name not in self._pools:
            self._pools[service_name] = aiohttp.ClientSession(**self._config)

        try:
            yield self._pools[service_name]
        except Exception as e:
            # 如果会话出现问题，关闭并重新创建
            await self._pools[service_name].close()
            del self._pools[service_name]
            raise e

    async def warmup(self, urls: list[str]):
        """预热连接"""
        async def _warmup_single(url: str):
            async with self.get_session("warmup") as session:
                try:
                    async with session.get(url) as response:
                        await response.read()
                except Exception:
                    pass

        await asyncio.gather(*[_warmup_single(url) for url in urls])

    async def cleanup(self):
        """清理所有连接"""
        for session in self._pools.values():
            await session.close()
        self._pools.clear()

    def close(self):
        """同步关闭所有连接"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for session in list(self._pools.values()):
                loop.run_until_complete(session.close())
            self._pools.clear()
        finally:
            loop.close()


class SyncClientManager:
    """同步客户端管理器"""

    def __init__(self, async_manager: ClientManager):
        self._async_manager = async_manager
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

    def get_session(self, service_name: str):
        """同步获取会话"""
        loop = self._ensure_loop()
        return loop.run_until_complete(
            self._async_manager.get_session(service_name).__aenter__()
        )

    def warmup(self, urls: list[str]):
        """同步预热"""
        loop = self._ensure_loop()
        loop.run_until_complete(
            self._async_manager.warmup(urls)
        )

    def cleanup(self):
        """同步清理"""
        if self._loop:
            try:
                for session in list(self._async_manager._pools.values()):
                    self._loop.run_until_complete(session.close())
                self._async_manager._pools.clear()
            finally:
                if not self._loop.is_closed():
                    self._loop.close()
                self._loop = None
