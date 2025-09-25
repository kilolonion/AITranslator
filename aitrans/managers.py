"""简化后的客户端管理器。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict


class _DummySession:
    async def post(self, *args, **kwargs):  # pragma: no cover - 兼容旧接口
        return _DummyResponse()

    async def get(self, *args, **kwargs):  # pragma: no cover - 兼容旧接口
        return _DummyResponse()

    async def close(self) -> None:  # pragma: no cover - 兼容旧接口
        return None


class _DummyResponse:
    async def json(self) -> Dict[str, object]:  # pragma: no cover
        return {"choices": [{"message": {"content": ""}}]}

    @property
    def content(self):  # pragma: no cover
        async def generator():
            if False:
                yield b""

        return generator()


class ClientManager:
    """占位实现，用于保持接口兼容。"""

    def __init__(self, **_: object) -> None:
        self._sessions: Dict[str, _DummySession] = {}

    @asynccontextmanager
    async def get_session(self, service_name: str) -> AsyncIterator[_DummySession]:
        session = self._sessions.setdefault(service_name, _DummySession())
        yield session

    async def warmup(self, urls: list[str]) -> None:  # pragma: no cover - 兼容旧接口
        await asyncio.sleep(0)

    async def cleanup(self) -> None:  # pragma: no cover - 兼容旧接口
        self._sessions.clear()

    def close(self) -> None:  # pragma: no cover - 兼容旧接口
        self._sessions.clear()


class SyncClientManager:
    """同步包装，保留以兼容旧接口。"""

    def __init__(self, async_manager: ClientManager) -> None:
        self._async_manager = async_manager
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def cleanup(self) -> None:  # pragma: no cover - 兼容旧接口
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
