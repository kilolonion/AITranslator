# AITrans

AITrans 是一个强大的 AI 驱动的翻译库，支持多种语言之间的高质量翻译。

## 特点

- 支持多种语言之间的翻译
- 异步和同步 API
- 支持批量翻译
- 上下文感知翻译
- 自定义翻译风格
- 术语表支持
- 流式翻译
- 性能优化和缓存

## 安装

```bash
pip install aitrans
```

## 快速开始

### 注册和配置
1. 获取 API 密钥：
   - [DeepSeek](https://www.deepseek.com/)
   - [豆包](https://console.volcengine.com/ark)
   - [OpenAI](https://openai.com/api/)

2. 设置 API 密钥：
```python
import os
os.environ["ARK_API_KEY"] = "your-api-key"
```

或在创建实例时传入：
```python
translator = AITranslatorSync(api_key="your-api-key")
```

### 同步用法 (AITranslatorSync)

AITranslatorSync 提供了简单易用的同步接口，适合大多数使用场景。

#### 1. 基础翻译
```python
from aitrans import AITranslatorSync

# 创建翻译器实例
translator = AITranslatorSync()

# 单个文本翻译
result = translator.translate("你好，世界！", dest="en")
print(result.text)  # Hello, world!

# 检测语言
detected = translator.detect_language("Hello")
print(f"语言: {detected.lang}, 置信度: {detected.confidence}")
```

#### 2. 批量翻译
```python
# 批量翻译多个文本
texts = ["你好", "世界", "人工智能"]
results = translator.translate_batch(texts, dest="en")
for result in results:
    print(f"{result.origin} -> {result.text}")
```

#### 3. 上下文感知翻译
```python
# 使用上下文提高翻译质量
context = "这是一篇关于机器学习的文章。"
result = translator.translate_with_context(
    text="模型的准确率达到了95%。",
    context=context,
    dest="en"
)
print(result.text)
```

#### 4. 风格化翻译
```python
# 使用预定义风格
formal_result = translator.translate_with_style(
    text="你好",
    dest="en",
    style="formal"
)

# 使用自定义风格
custom_style = {
    "语气": "正式",
    "表达方式": "简洁",
    "专业程度": "高"
}
custom_result = translator.translate_with_style(
    text="你好",
    dest="en",
    style=custom_style
)
```

#### 5. 文档翻译
```python
# 翻译整个文档，保持上下文连贯性
paragraphs = [
    "第一段：人工智能简介。",
    "第二段：机器学习基础。",
    "第三段：深度学习应用。"
]
results = translator.translate_document_with_context(
    paragraphs=paragraphs,
    dest="en",
    context_window=2
)
for result in results:
    print(result.text)
```

#### 6. 术语表使用
```python
# 加载术语表
translator.load_glossary("glossary.json")

# 添加术语
translator.add_term("AI", {
    "en": "Artificial Intelligence",
    "zh": "人工智能",
    "ja": "人工知能"
})

# 使用术语表翻译
result = translator.translate_with_glossary("AI 技术", "zh", "en")
```

### 异步用法 (AITranslator)

AITranslator 提供了异步接口，适合高性能和并发场景。

#### 1. 基础异步翻译
```python
import asyncio
from aitrans import AITranslator

async def translate_example():
    async with AITranslator() as translator:
        # 基础翻译
        result = await translator.ai_translate("你好，世界！", dest="en")
        print(result.text)
        
        # 语言检测
        detected = await translator.ai_detect("Hello")
        print(f"语言: {detected.lang}, 置信度: {detected.confidence}")

asyncio.run(translate_example())
```

#### 2. 流式翻译
```python
async def stream_example():
    async with AITranslator() as translator:
        async for partial_result in await translator.ai_translate(
            "这是一个很长的文本...",
            dest="en",
            stream=True
        ):
            print(partial_result.text, end="", flush=True)

asyncio.run(stream_example())
```

#### 3. 并发批量翻译
```python
async def batch_example():
    async with AITranslator() as translator:
        texts = ["文本1", "文本2", "文本3", "文本4", "文本5"]
        results = await translator.translate_batch(
            texts,
            dest="en",
            batch_size=2  # 控制并发数
        )
        for result in results:
            print(f"{result.origin} -> {result.text}")

asyncio.run(batch_example())
```

#### 4. 异步文档翻译
```python
async def document_example():
    async with AITranslator() as translator:
        paragraphs = [
            "第一段内容...",
            "第二段内容...",
            "第三段内容..."
        ]
        results = await translator.translate_document_with_context(
            paragraphs=paragraphs,
            dest="en",
            context_window=2,
            batch_size=2
        )
        for result in results:
            print(result.text)

asyncio.run(document_example())
```

#### 5. 性能优化
```python
async def optimized_example():
    async with AITranslator() as translator:
        # 预热连接
        await translator.preconnect()
        
        # 设置性能配置
        translator.set_performance_config(
            max_workers=5,
            cache_ttl=3600,
            min_request_interval=0.1
        )
        
        # 批量处理
        texts = ["文本1", "文本2", "文本3"]
        results = await translator.translate_batch(
            texts,
            dest="en",
            batch_size=2
        )

asyncio.run(optimized_example())
```

### 错误处理

```python
from aitrans import AIError, AIAuthenticationError, AIConnectionError

try:
    translator = AITranslatorSync()
    result = translator.translate("你好")
except AIAuthenticationError:
    print("API 密钥无效")
except AIConnectionError:
    print("网络连接错误")
except AIError as e:
    print(f"翻译错误: {str(e)}")
```

### 性能优化建议

1. 使用异步 API 处理大量请求
2. 启用预连接功能
3. 合理设置批处理大小
4. 选择合适的性能配置模式
5. 利用缓存机制减少重复请求

## 支持的语言

- 中文 (zh)
- 英语 (en)
- 日语 (ja)
- 韩语 (ko)
- 法语 (fr)
- 德语 (de)
- 俄语 (ru)
- 西班牙语 (es)
- 更多语言可自行查看对应代号

## 高级功能

### 术语表支持

```python
translator = AITranslatorSync()
translator.load_glossary("path/to/glossary.json")
result = translator.translate_with_glossary("专业术语", "zh", "en")
```

### 自定义翻译风格

```python
translator = AITranslatorSync()
result = translator.translate_with_style(
    "你好",
    dest="en",
    style="formal"
)
```

## API 文档

### 同步 API (AITranslatorSync)

#### 基础翻译
```python
def translate(self, text: str, dest: str = 'en', src: str = 'auto') -> AITranslated:
    """
    翻译单个文本
    
    参数:
        text: 要翻译的文本
        dest: 目标语言代码
        src: 源语言代码，默认为'auto'（自动检测）
    
    返回:
        AITranslated 对象，包含:
        - src: 源语言
        - dest: 目标语言
        - origin: 原文
        - text: 译文
        - pronunciation: 发音（如果可用）
    """
```

#### 批量翻译
```python
def translate_batch(self, texts: List[str], dest: str = 'en', src: str = 'auto') -> List[AITranslated]:
    """
    批量翻译多个文本
    
    参数:
        texts: 要翻译的文本列表
        dest: 目标语言代码
        src: 源语言代码，默认为'auto'
    
    返回:
        AITranslated 对象列表
    """
```

#### 上下文翻译
```python
def translate_with_context(
    self, 
    text: str,
    context: str,
    dest: str = 'en',
    src: str = 'auto',
    style_guide: str = None
) -> AITranslated:
    """
    带上下文的翻译
    
    参数:
        text: 要翻译的文本
        context: 上下文信息
        dest: 目标语言代码
        src: 源语言代码
        style_guide: 风格指南（可选）
    """
```

#### 风格化翻译
```python
def translate_with_style(
    self,
    text: str,
    dest: str = 'en',
    src: str = 'auto',
    style: Union[str, Dict] = 'formal',
    context: str = None,
    max_versions: int = 3
) -> AITranslated:
    """
    使用特定风格翻译
    
    参数:
        text: 要翻译的文本
        dest: 目标语言代码
        src: 源语言代码
        style: 翻译风格，可以是预定义风格名称或自定义风格字典
        context: 上下文信息（可选）
        max_versions: 创意风格时的最大版本数量
    """
```

#### 文档翻译
```python
def translate_document_with_context(
    self,
    paragraphs: List[str],
    dest: str = 'en',
    src: str = 'auto',
    context_window: int = 2,
    batch_size: int = 5,
    style_guide: str = None
) -> List[AITranslated]:
    """
    翻译整个文档，保持上下文连贯性
    
    参数:
        paragraphs: 段落列表
        dest: 目标语言代码
        src: 源语言代码
        context_window: 上下文窗口大小
        batch_size: 批处理大小
        style_guide: 风格指南
    """
```

#### 语言检测
```python
def detect_language(self, text: str) -> AIDetected:
    """
    检测文本语言
    
    参数:
        text: 要检测的文本
    
    返回:
        AIDetected 对象，包含:
        - lang: 检测到的语言代码
        - confidence: 置信度（0-1）
    """
```

```python
def detect_language_enhanced(self, text: str) -> AIDetected:
    """
    增强的语言检测，使用多个检测器
    
    参数:
        text: 要检测的文本
    
    返回:
        AIDetected 对象，包含更详细的检测信息
    """
```

#### 术语表管理
```python
def load_glossary(self, path: Union[str, Path]) -> None:
    """加载术语表"""

def save_glossary(self, path: Union[str, Path]) -> None:
    """保存术语表"""

def add_term(self, term_id: str, translations: Dict[str, str]) -> None:
    """添加术语"""

def get_term(self, term_id: str) -> Optional[Dict[str, str]]:
    """获取术语"""
```

#### 性能和配置
```python
def preconnect(self) -> bool:
    """预热连接以提高性能"""

def test_connection(self) -> bool:
    """测试API连接"""

def get_config(self) -> dict:
    """获取当前配置"""

def set_performance_config(self, **kwargs) -> None:
    """
    设置性能配置
    
    参数:
        max_workers: 最大并发数
        cache_ttl: 缓存有效期
        min_request_interval: 最小请求间隔
        max_retries: 最大重试次数
        timeout: 超时时间
        temperature: 采样温度
        max_tokens: 最大标记数
    """
```

### 异步 API (AITranslator)

异步 API 提供与同步 API 相同的功能，但使用异步方式调用。所有方法名称前加上 `ai_` 前缀。

#### 基础翻译
```python
async def ai_translate(
    self,
    text: Union[str, List[str]],
    dest: str = 'en',
    src: str = 'auto',
    stream: bool = False
) -> Union[AITranslated, List[AITranslated], AsyncGenerator]:
    """
    异步翻译
    
    参数:
        text: 要翻译的文本或文本列表
        dest: 目标语言代码
        src: 源语言代码
        stream: 是否使用流式翻译
    """
```

#### 批量翻译
```python
async def translate_batch(
    self,
    texts: List[str],
    dest: str = 'en',
    src: str = 'auto',
    batch_size: int = 10
) -> List[AITranslated]:
    """
    异步批量翻译
    
    参数:
        texts: 要翻译的文本列表
        dest: 目标语言代码
        src: 源语言代码
        batch_size: 每批处理的文本数量
    """
```

#### 上下文和风格化翻译
```python
async def translate_with_context(
    self,
    text: str,
    context: str,
    dest: str = 'en',
    src: str = 'auto',
    style_guide: str = None
) -> AITranslated:
    """异步上下文翻译"""

async def translate_with_style(
    self,
    text: str,
    dest: str = 'en',
    src: str = 'auto',
    style: Union[str, Dict] = 'formal',
    context: str = None,
    max_versions: int = 3
) -> AITranslated:
    """异步风格化翻译"""
```

#### 语言检测
```python
async def ai_detect(self, text: str) -> AIDetected:
    """异步语言检测"""

async def ai_detect_enhanced(self, text: str) -> AIDetected:
    """异步增强语言检测"""
```

#### 评估和性能
```python
async def evaluate_translation(
    self,
    original: str,
    translated: str,
    src: str,
    dest: str
) -> Dict[str, float]:
    """
    评估翻译质量
    
    返回:
        包含各项评分的字典:
        - accuracy: 准确性
        - fluency: 流畅性
        - professionalism: 专业性
        - style: 风格匹配度
    """
```

### 预定义风格模板

```python
style_templates = {
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
        1. 提供多个不同的翻译版本
        2. 每个版本使用不同的表达方式
        3. 保持原文的核心含义
        4. 限制在指定版本数以内
    """
}
```

### 性能配置模式

```python
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
    'balanced': {
        'max_workers': 5,
        'cache_ttl': 3600,
        'min_request_interval': 0.1,
        'max_retries': 3,
        'timeout': 30,
        'temperature': 0.3,
        'max_tokens': 1024,
    },
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
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！ 