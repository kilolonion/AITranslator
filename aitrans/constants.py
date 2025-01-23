"""常量和配置定义"""

# API 相关常量
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
MAX_RETRIES = 3
MAX_WORKERS = 5

# 性能配置
DEFAULT_PERFORMANCE_CONFIG = {
    'max_workers': 5,
    'cache_ttl': 3600,
    'min_request_interval': 0.1,
    'max_retries': 3,
    'timeout': 30,
    'temperature': 0.3,
    'max_tokens': 1024,
}

PERFORMANCE_PROFILES = {
    'fast': {
        'max_workers': 10,
        'cache_ttl': 1800,
        'min_request_interval': 0.05,
        'max_retries': 2,
        'timeout': 15,
        'temperature': 0.5,
        'max_tokens': 512,
    },
    'balanced': DEFAULT_PERFORMANCE_CONFIG,
    'accurate': {
        'max_workers': 3,
        'cache_ttl': 7200,
        'min_request_interval': 0.2,
        'max_retries': 5,
        'timeout': 60,
        'temperature': 0.1,
        'max_tokens': 2048,
    }
}

# 语言相关映射
LANGUAGE_NAMES = {
    'auto': '自动检测',
    'zh': '中文',
    'en': '英语',
    'ja': '日语',
    'ko': '韩语',
    'fr': '法语',
    'es': '西班牙语',
    'it': '意大利语',
    'de': '德语',
    'ru': '俄语',
    'pt': '葡萄牙语',
    'vi': '越南语',
    'th': '泰语',
    'ar': '阿拉伯语'
}

DOUBAO_LANGUAGES = {
    'zh': 'chinese',
    'en': 'english',
    'ja': 'japanese',
    'ko': 'korean',
    'fr': 'french',
    'es': 'spanish',
    'ru': 'russian',
    'de': 'german',
    'it': 'italian',
    'tr': 'turkish',
    'pt': 'portuguese',
    'vi': 'vietnamese',
    'id': 'indonesian',
    'th': 'thai',
    'ms': 'malay',
    'ar': 'arabic',
    'hi': 'hindi'
}

LANG_CODE_MAP = {
    'zh-cn': 'zh',
    'zh-tw': 'zh',
    'zh': 'zh',
    'en': 'en',
    'ja': 'ja',
    'ko': 'ko',
    'fr': 'fr',
    'es': 'es',
    'ru': 'ru',
    'de': 'de',
    'it': 'it',
    'tr': 'tr',
    'pt': 'pt',
    'vi': 'vi',
    'id': 'id',
    'th': 'th',
    'ms': 'ms',
    'ar': 'ar',
    'hi': 'hi'
}

# 翻译风格模板
STYLE_TEMPLATES = {
    'formal': """
        翻译要求：
        1. 使用正式的学术用语
        2. 保持严谨的句式结构
        3. 使用标准的专业术语
        4. 避免口语化和简化表达
    """,
    'casual': """
        翻译要求：
        1. 使用日常口语表达
        2. 保持语言自然流畅
        3. 使用简短句式
        4. 可以使用常见缩写
    """,
    'technical': """
        翻译要求：
        1. 严格使用技术术语
        2. 保持专业准确性
        3. 使用规范的技术表达
        4. 保持术语一致性
    """,
    'creative': """
        翻译要求：
        1. 提供2-3个不同的翻译版本
        2. 每个版本使用不同的表达方式
        3. 保持原文的核心含义
        4. 限制在3个版本以内
    """
}
