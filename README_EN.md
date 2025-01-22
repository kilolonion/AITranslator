# AITrans 

[中文](README.md) | [English](README_EN.md)

AITrans is a powerful AI-driven translation library that supports high-quality translation between multiple languages.

## Features

- Multi-language translation support
- Asynchronous and synchronous APIs
- Batch translation support
- Context-aware translation
- Custom translation styles
- Glossary support
- Streaming translation
- Performance optimization and caching

## Installation

```bash
pip install aitrans
```

## Quick Start

### Registration and Configuration
1. Get API Key from:
   - [DeepSeek](https://www.deepseek.com/)
   - [Doubao](https://console.volcengine.com/ark)
   - [OpenAI](https://openai.com/api/)

2. Set API Key:
```python
import os
os.environ["ARK_API_KEY"] = "your-api-key"
```

Or pass it when creating an instance:
```python
translator = AITranslatorSync(api_key="your-api-key")
```

Or create a `.env` file in your aitrans directory (usually at `C:\Users\YourUsername\AppData\Roaming\Python\Python311\site-packages\aitrans`) with the following parameters:
```
ARK_API_KEY=your-api-key
ARK_BASE_URL=https://api.deepseek.com/v1 (or replace with your AI model's URL)
ARK_MODEL=deepseek-chat (or replace with your AI model's name)
```

### Synchronous Usage (AITranslatorSync)

AITranslatorSync provides a simple and easy-to-use synchronous interface suitable for most use cases.

#### 1. Basic Translation
```python
from aitrans import AITranslatorSync

# Create translator instance
translator = AITranslatorSync()

# Single text translation
result = translator.translate("Hello, world!", dest="zh")
print(result.text)  # 你好，世界！

# Language detection
detected = translator.detect_language("Hello")
print(f"Language: {detected.lang}, Confidence: {detected.confidence}")
```

#### 2. Batch Translation
```python
# Translate multiple texts
texts = ["Hello", "World", "AI"]
results = translator.translate_batch(texts, dest="zh")
for result in results:
    print(f"{result.origin} -> {result.text}")
```

#### 3. Context-Aware Translation
```python
# Use context to improve translation quality
context = "This is an article about machine learning."
result = translator.translate_with_context(
    text="The model achieved 95% accuracy.",
    context=context,
    dest="zh"
)
print(result.text)
```

#### 4. Styled Translation
```python
# Use predefined style
formal_result = translator.translate_with_style(
    text="Hello",
    dest="zh",
    style="formal"
)

# Use custom style
custom_style = {
    "tone": "formal",
    "expression": "simple",
    "professionalism": "high"
}
custom_result = translator.translate_with_style(
    text="Hello",
    dest="zh",
    style=custom_style
)
```

#### 5. Document Translation
```python
# Translate entire document maintaining context coherence
paragraphs = [
    "Part 1: Introduction to AI",
    "Part 2: Machine Learning Basics",
    "Part 3: Deep Learning Applications"
]
results = translator.translate_document_with_context(
    paragraphs=paragraphs,
    dest="zh",
    context_window=2
)
for result in results:
    print(result.text)
```

#### 6. Glossary Usage
```python
# Load glossary
translator.load_glossary("glossary.json")

# Add term
translator.add_term("AI", {
    "en": "Artificial Intelligence",
    "zh": "人工智能",
    "ja": "人工知能"
})

# Use glossary in translation
result = translator.translate_with_glossary("AI technology", "en", "zh")
```

### Asynchronous Usage (AITranslator)

AITranslator provides asynchronous interfaces suitable for high-performance and concurrent scenarios.

#### 1. Basic Async Translation
```python
import asyncio
from aitrans import AITranslator

async def translate_example():
    async with AITranslator() as translator:
        # Basic translation
        result = await translator.ai_translate("Hello, world!", dest="zh")
        print(result.text)

        # Language detection
        detected = await translator.ai_detect("Hello")
        print(f"Language: {detected.lang}, Confidence: {detected.confidence}")

asyncio.run(translate_example())
```

#### 2. Streaming Translation
```python
async def stream_example():
    async with AITranslator() as translator:
        async for partial_result in await translator.ai_translate(
            "This is a very long text...",
            dest="zh",
            stream=True
        ):
            print(partial_result.text, end="", flush=True)

asyncio.run(stream_example())
```

#### 3. Concurrent Batch Translation
```python
async def batch_example():
    async with AITranslator() as translator:
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        results = await translator.translate_batch(
            texts,
            dest="zh",
            batch_size=2  # Control concurrency
        )
        for result in results:
            print(f"{result.origin} -> {result.text}")

asyncio.run(batch_example())
```

#### 4. Async Document Translation
```python
async def document_example():
    async with AITranslator() as translator:
        paragraphs = [
            "First paragraph content...",
            "Second paragraph content...",
            "Third paragraph content..."
        ]
        results = await translator.translate_document_with_context(
            paragraphs=paragraphs,
            dest="zh",
            context_window=2,
            batch_size=2
        )
        for result in results:
            print(result.text)

asyncio.run(document_example())
```

#### 5. Performance Optimization
```python
async def optimized_example():
    async with AITranslator() as translator:
        # Preheat connection
        await translator.preconnect()

        # Set performance configuration
        translator.set_performance_config(
            max_workers=5,
            cache_ttl=3600,
            min_request_interval=0.1
        )

        # Batch processing
        texts = ["Text 1", "Text 2", "Text 3"]
        results = await translator.translate_batch(
            texts,
            dest="zh",
            batch_size=2
        )

asyncio.run(optimized_example())
```

### Error Handling

```python
from aitrans import AIError, AIAuthenticationError, AIConnectionError

try:
    translator = AITranslatorSync()
    result = translator.translate("Hello")
except AIAuthenticationError:
    print("Invalid API key")
except AIConnectionError:
    print("Network connection error")
except AIError as e:
    print(f"Translation error: {str(e)}")
```

### Performance Optimization Tips

1. Use async API for handling large volumes of requests
2. Enable preconnection feature
3. Set appropriate batch size
4. Choose suitable performance configuration mode
5. Utilize caching mechanism to reduce repeated requests

## Supported Languages

- Chinese (zh)
- English (en)
- Japanese (ja)
- Korean (ko)
- French (fr)
- German (de)
- Russian (ru)
- Spanish (es)
- More languages can be found in the corresponding codes

## Advanced Features

### Glossary Support

```python
translator = AITranslatorSync()
translator.load_glossary("path/to/glossary.json")
result = translator.translate_with_glossary("Technical term", "en", "zh")
```

### Custom Translation Style

```python
translator = AITranslatorSync()
result = translator.translate_with_style(
    "Hello",
    dest="zh",
    style="formal"
)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

## Contribution

Welcome to submit issues and pull requests! 
