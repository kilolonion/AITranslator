"""
AITrans - 一个强大的AI翻译库

这个库提供了异步和同步的翻译接口，支持批量翻译、流式翻译和文档翻译等功能。
"""

__version__ = "1.2.4"

from .core import AITranslator
from .sync import AITranslatorSync, create
from .models import AITranslated, AIDetected
from .exceptions import (
    AIError, AIAuthenticationError, AIConnectionError,
    AIAPIError, AIConfigError, AIValidationError
)

__all__ = [
    'AITranslator',
    'AITranslatorSync',
    'create',
    'AITranslated',
    'AIDetected',
    'AIError',
    'AIAuthenticationError',
    'AIConnectionError',
    'AIAPIError',
    'AIConfigError',
    'AIValidationError'
]
