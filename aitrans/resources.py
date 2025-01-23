"""资源管理模块"""

import aiohttp
import openai
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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


class ResourceManager:
    """统一管理所有资源"""

    def __init__(self, api_key: str, base_url: str):
        self.session_manager = SessionManager()
        self.client_manager = ClientManager(api_key, base_url)
        self._is_initialized = False

    async def initialize(self):
        """初始化所有资源"""
        if not self._is_initialized:
            await self.session_manager.initialize()
            await self.client_manager.initialize()
            self._is_initialized = True
            logger.info("All resources initialized")

    async def cleanup(self):
        """清理所有资源"""
        try:
            await self.session_manager.cleanup()
            await self.client_manager.cleanup()
            self._is_initialized = False
            logger.info("All resources cleaned up")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
            raise

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
