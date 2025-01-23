"""异常处理模块"""


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


class AITranslationError(AIError):
    """翻译错误"""
    pass
