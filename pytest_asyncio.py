"""为测试提供最小化的 asyncio 支持。"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

import pytest


def fixture(func: Callable[..., Any] | None = None, *args, **kwargs):
    if func is not None and callable(func) and not args and not kwargs:
        return fixture(*args, **kwargs)(func)

    def decorator(func: Callable[..., Any]):
        if inspect.isasyncgenfunction(func):

            @pytest.fixture(*args, **kwargs)
            def wrapper(*wargs, **wkwargs):
                loop = asyncio.new_event_loop()
                agen = func(*wargs, **wkwargs)
                asyncio.set_event_loop(loop)
                try:
                    value = loop.run_until_complete(agen.__anext__())
                    try:
                        yield value
                    finally:
                        try:
                            loop.run_until_complete(agen.__anext__())
                        except StopAsyncIteration:
                            pass
                finally:
                    loop.close()

            return wrapper

        if inspect.iscoroutinefunction(func):

            @pytest.fixture(*args, **kwargs)
            def wrapper(*wargs, **wkwargs):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(func(*wargs, **wkwargs))
                finally:
                    loop.close()

            return wrapper

        return pytest.fixture(*args, **kwargs)(func)

    return decorator


def mark_asyncio(func: Callable[..., Any]) -> Callable[..., Any]:
    return pytest.mark.asyncio(func)


def pytest_configure(config):  # pragma: no cover - pytest 钩子
    config.addinivalue_line("markers", "asyncio: mark test as asyncio aware")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):  # pragma: no cover - pytest 钩子
    if not inspect.iscoroutinefunction(pyfuncitem.function):
        return None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(pyfuncitem.obj(**pyfuncitem.funcargs))
    finally:
        loop.close()
    return True
