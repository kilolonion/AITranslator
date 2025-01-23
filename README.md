# AITrans

[English](README.md) | [简体中文](README.zh-cn.md)

A powerful AI-based translation library with both asynchronous and synchronous interfaces, providing efficient and flexible translation capabilities.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Performance Monitoring](#performance-monitoring)
- [Best Practices](#best-practices)
- [FAQ](#faq)
- [API Documentation](#api-documentation)
- [License](#license)

## Features

- **Async and Sync Interfaces**
  - Support for both async and sync calling methods
  - Automatic event loop and resource management
  - Nested event loop support (via nest-asyncio)

- **Streaming Translation**
  - Real-time translation results
  - Progress display support
  - Suitable for long text translation
  - Customizable streaming output handling

- **Batch Translation**
  - Support for parallel translation of multiple texts
  - Automatic load balancing
  - Intelligent task scheduling
  - Batch task progress tracking

- **Style-based Translation**
  - formal: Business and formal style
  - casual: Conversational style
  - Extensible style system

- **Context-aware Translation**
  - Context relevance analysis
  - Intelligent context understanding
  - Professional domain adaptation

- **Smart Error Recovery**
  - Exponential backoff retry mechanism
  - Automatic network fluctuation handling
  - Session state recovery
  - Intelligent error classification and handling

- **Performance Monitoring**
  - Request statistics
  - Response time analysis
  - Resource usage monitoring
  - Performance bottleneck detection

- **Cache Support**
  - Multi-level cache architecture
  - Configurable cache strategies
  - Cache TTL management
  - Cache hit rate statistics

- **Resource Management**
  - Automatic connection pool management
  - Smart event loop control
  - Automatic resource release
  - Memory usage optimization

## Installation

### Basic Installation

```bash
pip install aitrans
```

### Optional Dependencies

```bash
# Support for nested event loops
pip install nest-asyncio

# Support for performance monitoring
pip install psutil

# Support for colored logging
pip install colorama

# Install all optional dependencies
pip install aitrans[full]
```

## Quick Start

### Async Translation

```python
import asyncio
from aitrans import AITranslator

async def main():
    # Create translator instance
    translator = AITranslator(
        api_key="your_api_key",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        max_workers=5
    )
    await translator.initialize()

    try:
        # Basic translation
        result = await translator.translate("Hello world!", dest="zh")
        print(f"Translation: {result.text}")
        print(f"Source language: {result.src}")
        print(f"Target language: {result.dest}")
        print(f"Confidence: {result.confidence}")

        # Streaming translation
        text = "This is a long text for streaming translation."
        print("Starting streaming translation...")
        async for partial in translator.translate(text, dest="zh", stream=True):
            print(partial, end="", flush=True)
        print("\nStreaming translation completed")

        # Batch translation
        texts = ["Hello", "Good morning", "Good night"]
        results = await translator.translate_batch(texts, dest="zh")
        for text, result in zip(texts, results):
            print(f"{text} -> {result.text}")

    finally:
        await translator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Sync Translation

```python
from aitrans import create
from aitrans.exceptions import AIError

# Use context manager for automatic resource management
with create(
    api_key="your_api_key",
    model_name="deepseek-chat",
    performance_mode="balanced"
) as translator:
    try:
        # Basic translation
        result = translator.translate("Hello world!", dest="zh")
        print(f"Translation: {result.text}")

        # Batch translation
        texts = ["Good morning", "Good afternoon", "Good night"]
        results = translator.translate(texts, dest="zh")
        for text, result in zip(texts, results):
            print(f"{text} -> {result.text}")

        # Streaming translation (with progress bar)
        text = "This is a long text for testing streaming translation."
        print("\nStarting streaming translation:")
        print("[", end="")
        for i, partial in enumerate(translator.translate(text, dest="zh", stream=True)):
            print("#", end="", flush=True)
        print("]")

        # Style-based translation
        text = "This product is very user-friendly"
        # Formal style
        result = translator.translate_with_style(
            text, dest="zh", style="formal"
        )
        print(f"Formal style: {result.text}")
        # Casual style
        result = translator.translate_with_style(
            text, dest="zh", style="casual"
        )
        print(f"Casual style: {result.text}")

        # Context-aware translation
        context = "This is an article about computer programming."
        text = "Its performance is excellent."
        result = translator.translate_with_context(text, context, dest="zh")
        print(f"Context-aware translation: {result.text}")

    except AIError as e:
        print(f"Translation error: {str(e)}")
```

## Advanced Usage

### Custom Glossary

```python
from aitrans import create

# Create glossary
glossary = {
    "AI": "人工智能",
    "ML": "机器学习",
    "NLP": "自然语言处理"
}

# Use glossary for translation
with create(glossary=glossary) as translator:
    result = translator.translate(
        "AI and ML are important parts of NLP.",
        dest="zh"
    )
    print(result.text)
```

### Document Translation

```python
from aitrans import create
from aitrans.document import DocumentTranslator

with create() as translator:
    doc_translator = DocumentTranslator(translator)
    
    # Translate Markdown document
    doc_translator.translate_file(
        "input.md",
        "output.md",
        src="en",
        dest="zh"
    )
    
    # Translate HTML document
    doc_translator.translate_file(
        "input.html",
        "output.html",
        src="en",
        dest="zh"
    )
```

## Best Practices

### 1. Choosing Between Async and Sync Interfaces

#### Async Interface (AITranslator)
Suitable for:
- Web services (Flask, FastAPI, etc.)
- Real-time applications (chatbots, live translation)
- Data pipelines (ETL processes)

```python
async def translate_text():
    translator = AITranslator()
    await translator.initialize()
    try:
        result = await translator.translate("Hello world!", dest="zh")
        print(result.text)
    finally:
        await translator.cleanup()
```

#### Sync Interface (AITranslatorSync)
Suitable for:
- Script tools
- GUI applications
- CLI tools

```python
with create() as translator:
    result = translator.translate("Hello world!", dest="zh")
    print(result.text)
```

### 2. Configuration Optimization

#### API Key Management
```python
import os
from aitrans import create

# Recommended: Get API key from environment variables
translator = create(api_key=os.getenv("AI_API_KEY"))

# Recommended: Get API key from .env config file
translator = create(api_key=config.get("api_key"))

# Not recommended: Hardcoded API key
translator = create(api_key="your-api-key")
```

#### Model Selection
```python
# General translation scenarios
translator = create(model_name="deepseek-chat")

# High-quality translation scenarios
translator = create(model_name="gpt-4")
```

### 3. Error Handling

#### Fine-grained Error Handling
```python
try:
    result = translator.translate("Hello", dest="zh")
except AIAuthenticationError:
    # Handle authentication errors
    refresh_api_key()
except AIRateLimitError:
    # Handle rate limit errors
    apply_rate_limiting()
except AINetworkError:
    # Handle network errors
    retry_with_backoff()
except AIError as e:
    # Handle other errors
    log_error(e)
```

### 4. Performance Optimization

#### Performance Metrics Collection
```python
# Enable performance monitoring
translator = create(
    enable_metrics=True,
    metrics_window=3600  # 1-hour statistics window
)

# Get performance metrics
metrics = translator.get_metrics()
print(f"Average response time: {metrics['avg_response_time']}ms")
print(f"Success rate: {metrics['success_rate']}%")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
```

## FAQ

### Q1: How to handle large-scale translation tasks?

```python
from aitrans import create
from concurrent.futures import ThreadPoolExecutor
import queue

def translate_large_dataset(file_path, dest="zh", batch_size=100):
    with create() as translator:
        # Create task queue
        task_queue = queue.Queue()
        
        # Read file and process in batches
        with open(file_path) as f:
            batch = []
            for line in f:
                batch.append(line.strip())
                if len(batch) >= batch_size:
                    task_queue.put(batch)
                    batch = []
            if batch:
                task_queue.put(batch)
        
        # Create worker thread pool
        with ThreadPoolExecutor(max_workers=5) as executor:
            while not task_queue.empty():
                batch = task_queue.get()
                results = translator.translate_batch(batch, dest=dest)
                process_results(results)
```

### Q2: How to optimize translation quality?

```python
# Use context-aware translation
context = "This is a technical document."
text = "The implementation is elegant."
result = translator.translate_with_context(
    text,
    context=context,
    dest="zh",
    quality_preference="high"
)

# Use glossary
glossary = {
    "elegant": "优雅",
    "implementation": "实现"
}
result = translator.translate(
    text,
    dest="zh",
    glossary=glossary
)
```

## License

MIT License

## Contributing

Feel free to submit [Issues](https://github.com/yourusername/aitrans/issues) and Pull Requests!