# AITrans 支持多款常见API的AI翻译器

[中文](README.md) | [English](README_EN.md)

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

方式一：通过环境变量设置
```python
import os
os.environ["ARK_API_KEY"] = "your-api-key"
os.environ["ARK_BASE_URL"] = "https://api.deepseek.com/v1"  # 或其他模型的 URL
os.environ["ARK_MODEL"] = "deepseek-chat"  # 或其他模型名称
```

方式二：创建实例时传入
```python
translator = AITranslatorSync(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1",
    model_name="deepseek-chat"
)
```

方式三：使用 .env 文件（推荐）
1. 找到 aitrans 安装目录（通常在 site-packages/aitrans）
2. 创建 .env 文件并添加以下内容：
```
ARK_API_KEY=你的API KEY
ARK_BASE_URL=https://api.deepseek.com/v1
ARK_MODEL=deepseek-chat
```

### 同步用法 (AITranslatorSync)
同步翻译是一部反映异步翻译的同步封装版本，不会阻塞线程，但会阻塞异步翻译

#### 1. 基础翻译
```python
from aitrans import AITranslatorSync

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
texts = ["你好", "世界", "人工智能"]
results = translator.translate_batch(texts, dest="en")
for result in results:
    print(f"{result.origin} -> {result.text}")
```

#### 3. 上下文感知翻译
```python
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
styles = ["formal", "casual", "creative"]
text = "这个产品非常好用。"

for style in styles:
    result = translator.translate_with_style(
        text=text,
        dest="en",
        style=style
    )
    print(f"{style}: {result.text}")
```

#### 5. 流式翻译
```python
text = "让我们测试一下流式翻译功能。"
for partial_result in translator.translate(text, dest="en", stream=True):
    print(partial_result.text, end="", flush=True)
```

### 异步用法 (AITranslator)

异步翻译提供了高性能的翻译接口，适合处理大量并发请求和流式翻译场景。

#### 1. 基础异步翻译
```python
import asyncio
from aitrans import AITranslator

async def translate_example():
    async with AITranslator() as translator:
        # 基础翻译
        result = await translator.translate("你好，世界！", dest="en")
        print(result.text)
        
        # 语言检测
        detected = await translator.detect("Hello")
        print(f"语言: {detected}, 置信度: 高")

asyncio.run(translate_example())
```

#### 2. 批量异步翻译
```python
async def batch_example():
    async with AITranslator() as translator:
        texts = ["你好", "世界", "人工智能", "机器学习", "深度学习"]
        results = await translator.translate_batch(
            texts,
            dest="en",
            batch_size=2  # 控制并发数
        )
        for result in results:
            print(f"{result.origin} -> {result.text}")

asyncio.run(batch_example())
```

#### 3. 异步上下文翻译
```python
async def context_example():
    async with AITranslator() as translator:
        context = "这是一篇关于机器学习的论文。"
        text = "模型在测试集上的准确率达到了98%。"
        
        result = await translator.translate_with_context(
            text=text,
            context=context,
            dest="en"
        )
        print(f"上下文: {context}")
        print(f"原文: {text}")
        print(f"译文: {result.text}")

asyncio.run(context_example())
```

#### 4. 异步风格化翻译
```python
async def style_example():
    async with AITranslator() as translator:
        text = "这个产品非常好用。"
        styles = ["formal", "casual", "creative"]
        
        for style in styles:
            result = await translator.translate_with_style(
                text=text,
                dest="en",
                style=style
            )
            print(f"{style}风格: {result.text}")

asyncio.run(style_example())
```

#### 5. 异步流式翻译
```python
async def stream_example():
    async with AITranslator() as translator:
        text = "这是一个很长的文本，用来测试流式翻译功能。"
        print(f"原文: {text}")
        print("流式翻译过程:")
        
        async for partial_result in await translator.translate(
            text,
            dest="en",
            stream=True
        ):
            print(partial_result.text, end="", flush=True)
        print()  # 换行

asyncio.run(stream_example())
```

#### 6. 异步文档翻译
```python
async def document_example():
    async with AITranslator() as translator:
        paragraphs = [
            "第一段：人工智能简介。",
            "第二段：机器学习基础。",
            "第三段：深度学习应用。"
        ]
        
        results = await translator.translate_batch(
            paragraphs,
            dest="en",
            batch_size=2
        )
        
        print("文档翻译结果:")
        for i, result in enumerate(results, 1):
            print(f"第{i}段: {result.text}")

asyncio.run(document_example())
```

#### 7. 异步性能优化
```python
async def optimized_example():
    async with AITranslator() as translator:
        # 预热连接
        await translator.preconnect()
        
        # 设置性能配置
        translator.set_performance_config(
            max_workers=5,
            cache_ttl=3600,
            min_request_interval=0.1,
            max_retries=3,
            timeout=30,
            temperature=0.3,
            max_tokens=1024
        )
        
        # 批量处理
        texts = ["文本1", "文本2", "文本3"]
        results = await translator.translate_batch(
            texts,
            dest="en",
            batch_size=2
        )
        
        for result in results:
            print(result.text)

asyncio.run(optimized_example())
```

### 性能优化

1. 使用预连接提高性能
```python
translator = AITranslatorSync()
translator.preconnect()  # 预热连接
```

2. 选择性能模式
```python
translator = AITranslatorSync(performance_mode="fast")  # 可选: fast, balanced, accurate
```

3. 自定义性能配置
```python
translator.set_performance_config(
    max_workers=5,
    cache_ttl=3600,
    min_request_interval=0.1,
    max_retries=3,
    timeout=30,
    temperature=0.3,
    max_tokens=1024
)
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

## 支持的语言

- 中文 (zh)
- 英语 (en)
- 日语 (ja)
- 韩语 (ko)
- 法语 (fr)
- 德语 (de)
- 俄语 (ru)
- 西班牙语 (es)
- 意大利语 (it)
- 葡萄牙语 (pt)
- 越南语 (vi)
- 泰语 (th)
- 阿拉伯语 (ar)
- 其他语言 (auto)

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！ 
