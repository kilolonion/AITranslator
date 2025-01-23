# AITrans

[English](README.md) | 简体中文

一个基于 AI 的翻译库，支持异步和同步接口，提供高效、灵活的翻译功能。

## 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [高级用法](#高级用法)
- [配置选项](#配置选项)
- [错误处理](#错误处理)
- [性能监控](#性能监控)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)
- [API 文档](#api-文档)
- [许可证](#许可证)

## 特性

- **异步和同步接口**
  - 支持异步和同步两种调用方式
  - 自动管理事件循环和资源
  - 支持嵌套事件循环（通过 nest-asyncio）

- **流式翻译**
  - 实时返回翻译结果
  - 支持进度显示
  - 适用于长文本翻译
  - 可自定义流式输出处理

- **批量翻译**
  - 支持多文本并行翻译
  - 自动负载均衡
  - 智能任务调度
  - 支持批量任务进度跟踪

- **风格化翻译**
  - formal：正式商务风格
  - casual：日常会话风格
  - 可扩展的风格系统

- **上下文翻译**
  - 支持上下文相关性分析
  - 智能语境理解
  - 专业领域适配

- **智能错误恢复**
  - 指数退避重试机制
  - 自动处理网络波动
  - 会话状态恢复
  - 智能错误分类和处理

- **性能监控**
  - 请求统计
  - 响应时间分析
  - 资源使用监控
  - 性能瓶颈检测

- **缓存支持**
  - 多级缓存架构
  - 可配置的缓存策略
  - 缓存有效期管理
  - 缓存命中率统计

- **资源管理**
  - 自动连接池管理
  - 智能事件循环控制
  - 资源自动释放
  - 内存使用优化

## 安装

### 基础安装

```bash
pip install aitrans
```

### 安装可选依赖

```bash
# 支持嵌套事件循环
pip install nest-asyncio

# 支持性能监控
pip install psutil

# 支持日志着色
pip install colorama

# 安装所有可选依赖
pip install aitrans[full]
```

## 快速开始

### 异步翻译

```python
import asyncio
from aitrans import AITranslator

async def main():
    # 创建翻译器实例
    translator = AITranslator(
        api_key="your_api_key",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        max_workers=5
    )
    await translator.initialize()

    try:
        # 基础翻译
        result = await translator.translate("Hello world!", dest="zh")
        print(f"翻译结果: {result.text}")
        print(f"源语言: {result.src}")
        print(f"目标语言: {result.dest}")
        print(f"置信度: {result.confidence}")

        # 流式翻译
        text = "This is a long text for streaming translation."
        print("开始流式翻译...")
        async for partial in translator.translate(text, dest="zh", stream=True):
            print(partial, end="", flush=True)
        print("\n流式翻译完成")

        # 批量翻译
        texts = ["Hello", "Good morning", "Good night"]
        results = await translator.translate_batch(texts, dest="zh")
        for text, result in zip(texts, results):
            print(f"{text} -> {result.text}")

    finally:
        await translator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### 同步翻译

```python
from aitrans import create
from aitrans.exceptions import AIError

# 使用上下文管理器自动管理资源
with create(
    api_key="your_api_key",
    model_name="deepseek-chat",
    performance_mode="balanced"
) as translator:
    try:
        # 基础翻译
        result = translator.translate("Hello world!", dest="zh")
        print(f"翻译结果: {result.text}")

        # 批量翻译
        texts = ["早上好", "下午好", "晚上好"]
        results = translator.translate(texts, dest="en")
        for text, result in zip(texts, results):
            print(f"{text} -> {result.text}")

        # 流式翻译（带进度条）
        text = "这是一个很长的文本，用于测试流式翻译功能。"
        print("\n开始流式翻译:")
        print("[", end="")
        for i, partial in enumerate(translator.translate(text, dest="en", stream=True)):
            print("#", end="", flush=True)
        print("]")

        # 风格化翻译
        text = "这个产品非常好用"
        # 正式风格
        result = translator.translate_with_style(
            text, dest="en", style="formal"
        )
        print(f"正式风格: {result.text}")
        # 随意风格
        result = translator.translate_with_style(
            text, dest="en", style="casual"
        )
        print(f"随意风格: {result.text}")

        # 上下文翻译
        context = "这是一篇关于计算机编程的文章。"
        text = "它的性能很好。"
        result = translator.translate_with_context(text, context, dest="en")
        print(f"带上下文翻译: {result.text}")

        # 获取原始响应
        result = translator.translate(
            "Hello", dest="zh", return_raw=True
        )
        print(f"原始响应: {result.raw_response}")

    except AIError as e:
        print(f"翻译出错: {str(e)}")
```

## 高级用法

### 自定义术语表

```python
from aitrans import create

# 创建术语表
glossary = {
    "AI": "人工智能",
    "ML": "机器学习",
    "NLP": "自然语言处理"
}

# 使用术语表进行翻译
with create(glossary=glossary) as translator:
    result = translator.translate(
        "AI and ML are important parts of NLP.",
        dest="zh"
    )
    print(result.text)  # 输出: 人工智能和机器学习是自然语言处理的重要组成部分。
```

### 文档翻译

```python
from aitrans import create
from aitrans.document import DocumentTranslator

with create() as translator:
    doc_translator = DocumentTranslator(translator)
    
    # 翻译 Markdown 文档
    doc_translator.translate_file(
        "input.md",
        "output.md",
        src="en",
        dest="zh"
    )
    
    # 翻译 HTML 文档
    doc_translator.translate_file(
        "input.html",
        "output.html",
        src="en",
        dest="zh"
    )
```

### 性能配置

```python
from aitrans import create
from aitrans.constants import PerformanceMode

with create(
    max_workers=10,
    performance_mode=PerformanceMode.SPEED,
    connection_timeout=30,
    read_timeout=60,
    retry_count=3,
    retry_interval=1,
    cache_ttl=3600
) as translator:
    # 配置性能监控
    translator.set_performance_config(
        enable_metrics=True,
        sample_rate=0.1,
        metrics_window=3600
    )
    
    # 执行翻译
    result = translator.translate("Hello", dest="zh")
    
    # 获取性能指标
    metrics = translator.get_metrics()
    print(f"平均响应时间: {metrics['avg_response_time']}ms")
    print(f"请求成功率: {metrics['success_rate']}%")
    print(f"缓存命中率: {metrics['cache_hit_rate']}%")
```

### 自定义缓存

```python
from aitrans import create
from aitrans.cache import CacheBackend

class CustomCache(CacheBackend):
    async def get(self, key: str):
        # 实现获取缓存
        pass
        
    async def set(self, key: str, value: any, ttl: int = None):
        # 实现设置缓存
        pass
        
    async def clear(self):
        # 实现清理缓存
        pass

# 使用自定义缓存
with create(cache_backend=CustomCache()) as translator:
    result = translator.translate("Hello", dest="zh")
```

### 批量任务控制

```python
from aitrans import create

with create() as translator:
    # 设置批量任务参数
    translator.set_batch_config(
        batch_size=100,        # 每批次最大任务数
        interval=0.1,          # 批次间隔
        max_concurrency=10     # 最大并发数
    )
    
    # 执行批量翻译
    texts = ["text1", "text2", "text3", ...]
    results = translator.translate_batch(
        texts,
        dest="zh",
        progress_callback=lambda x: print(f"进度: {x}%")
    )
```

## 配置选项

### 基础配置

```python
from aitrans import create

translator = create(
    # 必需参数
    api_key="your_api_key",      # API 密钥
    
    # 可选参数
    model_name="deepseek-chat",  # 模型名称
    base_url="https://api.example.com",  # API 基础 URL
    max_workers=5,               # 最大并发数
    glossary_path="glossary.json",  # 术语表路径
    performance_mode="balanced", # 性能模式：balanced, speed, quality
    auto_preconnect=True,       # 自动预热连接
    
    # 超时设置
    connection_timeout=30,       # 连接超时（秒）
    read_timeout=60,            # 读取超时（秒）
    
    # 重试设置
    retry_count=3,              # 最大重试次数
    retry_interval=1,           # 重试间隔（秒）
    
    # 缓存设置
    enable_cache=True,          # 启用缓存
    cache_ttl=3600,            # 缓存有效期（秒）
    
    # 日志设置
    log_level="INFO",          # 日志级别
    log_file="aitrans.log",    # 日志文件
    
    # 代理设置
    proxy="http://proxy.example.com:8080",  # HTTP 代理
    
    # SSL 设置
    verify_ssl=True,           # 验证 SSL 证书
    
    # 自定义请求头
    headers={                  
        "User-Agent": "Custom User Agent",
        "X-Custom-Header": "Value"
    }
)
```

### 性能模式

```python
from aitrans.constants import PerformanceMode

# 速度优先模式
translator = create(performance_mode=PerformanceMode.SPEED)

# 质量优先模式
translator = create(performance_mode=PerformanceMode.QUALITY)

# 平衡模式
translator = create(performance_mode=PerformanceMode.BALANCED)
```

## 错误处理

### 错误类型

```python
from aitrans import create
from aitrans.exceptions import (
    AIError,              # 基础错误类
    AIAuthenticationError,  # 认证错误
    AIRateLimitError,      # 频率限制错误
    AINetworkError,        # 网络错误
    AIServiceUnavailableError,  # 服务不可用错误
    AIInvalidRequestError,  # 无效请求错误
    AITimeoutError,        # 超时错误
    AIValueError          # 值错误
)

with create() as translator:
    try:
        result = translator.translate("Hello", dest="zh")
    except AIAuthenticationError as e:
        print(f"认证失败: {e}")
    except AIRateLimitError as e:
        print(f"超出频率限制: {e}")
    except AINetworkError as e:
        print(f"网络错误: {e}")
    except AIServiceUnavailableError as e:
        print(f"服务不可用: {e}")
    except AITimeoutError as e:
        print(f"请求超时: {e}")
    except AIError as e:
        print(f"其他错误: {e}")
```

### 自定义重试策略

```python
from aitrans import create
from aitrans.utils import RetryStrategy

class CustomRetryStrategy(RetryStrategy):
    def should_retry(self, error: Exception) -> bool:
        # 实现重试判断逻辑
        return isinstance(error, AINetworkError)
    
    def get_retry_interval(self, attempt: int) -> float:
        # 实现重试间隔计算逻辑
        return min(2 ** attempt, 60)

# 使用自定义重试策略
with create(retry_strategy=CustomRetryStrategy()) as translator:
    result = translator.translate("Hello", dest="zh")
```

## 性能监控

### 基础指标

```python
with create() as translator:
    metrics = translator.get_metrics()
    
    # 请求统计
    print(f"总请求数: {metrics['total_requests']}")
    print(f"成功请求数: {metrics['successful_requests']}")
    print(f"失败请求数: {metrics['failed_requests']}")
    
    # 响应时间
    print(f"平均响应时间: {metrics['avg_response_time']}ms")
    print(f"最大响应时间: {metrics['max_response_time']}ms")
    print(f"最小响应时间: {metrics['min_response_time']}ms")
    
    # 缓存统计
    print(f"缓存命中数: {metrics['cache_hits']}")
    print(f"缓存未命中数: {metrics['cache_misses']}")
    print(f"缓存命中率: {metrics['cache_hit_rate']}%")
    
    # 并发统计
    print(f"当前活动连接数: {metrics['active_connections']}")
    print(f"最大并发连接数: {metrics['max_connections']}")
```

### 自定义指标收集

```python
from aitrans import create
from aitrans.metrics import MetricsCollector

class CustomMetricsCollector(MetricsCollector):
    def collect_request_metric(self, duration: float, success: bool):
        # 实现请求指标收集逻辑
        pass
        
    def collect_cache_metric(self, hit: bool):
        # 实现缓存指标收集逻辑
        pass

# 使用自定义指标收集器
with create(metrics_collector=CustomMetricsCollector()) as translator:
    result = translator.translate("Hello", dest="zh")
```

## 最佳实践

### 1. 异步与同步接口的选择

#### 异步接口 (AITranslator)
适用场景：
- Web 服务（Flask、FastAPI 等）
- 实时应用（聊天机器人、实时翻译）
- 数据管道（ETL 流程）

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

#### 同步接口 (AITranslatorSync)
适用场景：
- 脚本工具
- GUI 应用
- CLI 工具

```python
with create() as translator:
    result = translator.translate("Hello world!", dest="zh")
    print(result.text)
```

### 2. 初始化配置优化

#### API 密钥管理
```python
import os
from aitrans import create

# 推荐：从环境变量获取密钥
translator = create(api_key=os.getenv("AI_API_KEY"))

# 推荐：从.env配置文件获取密钥
translator = create(api_key=config.get("api_key"))

#.env配置文件结构
(
ARK_API_KEY=your_api_key
ARK_BASE_URL=https://api.deepseek.com/v1 # 可选
ARK_MODEL=deepseek-chat # 可选
ARK_MAX_WORKERS=5 # 可选
ARK_GLOSSARY_PATH=glossary.json # 可选
ARK_PERFORMANCE_MODE=balanced # 可选
ARK_AUTO_PRECONNECT=True # 可选
ARK_CONNECTION_TIMEOUT=30 # 可选
ARK_READ_TIMEOUT=60
ARK_RETRY_COUNT=3
ARK_RETRY_INTERVAL=1
ARK_ENABLE_CACHE=True # 可选
ARK_CACHE_TTL=3600 # 可选
ARK_LOG_LEVEL=INFO # 可选
ARK_LOG_FILE=aitrans.log # 可选
ARK_PROXY=http://proxy.example.com:8080 # 可选
ARK_VERIFY_SSL=True # 可选
ARK_HEADERS={"User-Agent": "Custom User Agent", "X-Custom-Header": "Value"} # 可选
)


# 不推荐：硬编码密钥
translator = create(api_key="your-api-key")
```

#### 模型选择
```python
# 通用翻译场景
translator = create(model_name="deepseek-chat")

# 高质量翻译场景
translator = create(model_name="gpt-4")
```

#### 性能调优
```python
# 高并发场景
translator = create(
    max_workers=10,
    performance_mode="speed",
    connection_timeout=30,
    read_timeout=60
)

# 资源受限场景
translator = create(
    max_workers=3,
    performance_mode="balanced",
    cache_ttl=3600
)
```

### 3. 翻译功能使用

#### 基础翻译
```python
# 单文本翻译
result = translator.translate("Hello world!", dest="zh")

# 自动语言检测
result = translator.translate("Bonjour le monde!", dest="zh")
```

#### 流式翻译
适用于长文本翻译场景：
```python
# 带进度显示的流式翻译
text = "这是一个很长的文本..."
print("翻译进度: [", end="")
for partial in translator.translate(text, dest="en", stream=True):
    print("#", end="", flush=True)
    process_partial_result(partial)
print("]")
```

#### 批量翻译
适用于大量文本处理场景：
```python
# 推荐：批量处理
texts = ["text1", "text2", "text3"]
results = translator.translate_batch(
    texts,
    dest="zh",
    batch_size=100,
    progress_callback=lambda x: print(f"进度: {x}%")
)

# 不推荐：循环处理
for text in texts:
    result = translator.translate(text, dest="zh")
```

#### 风格化翻译
```python
# 正式商务场景
result = translator.translate_with_style(
    "这个产品非常好用",
    dest="en",
    style="formal"
)

# 日常会话场景
result = translator.translate_with_style(
    "这个产品非常好用",
    dest="en",
    style="casual"
)
```

#### 上下文翻译
```python
# 技术文档翻译
context = "这是一篇关于Python编程的文章。"
text = "它的性能很好。"
result = translator.translate_with_context(
    text,
    context=context,
    dest="en"
)
```

### 4. 错误处理与重试

#### 细粒度错误处理
```python
try:
    result = translator.translate("Hello", dest="zh")
except AIAuthenticationError:
    # 处理认证错误
    refresh_api_key()
except AIRateLimitError:
    # 处理频率限制
    apply_rate_limiting()
except AINetworkError:
    # 处理网络错误
    retry_with_backoff()
except AIServiceUnavailableError:
    # 处理服务不可用
    switch_to_backup_service()
except AITimeoutError:
    # 处理超时
    increase_timeout()
except AIError as e:
    # 处理其他错误
    log_error(e)
```

#### 自定义重试策略
```python
class CustomRetryStrategy(RetryStrategy):
    def should_retry(self, error: Exception) -> bool:
        return isinstance(error, (AINetworkError, AITimeoutError))
    
    def get_retry_interval(self, attempt: int) -> float:
        return min(2 ** attempt, 60)  # 最大等待60秒

translator = create(retry_strategy=CustomRetryStrategy())
```

### 5. 性能监控与优化

#### 性能指标收集
```python
# 启用性能监控
translator = create(
    enable_metrics=True,
    metrics_window=3600  # 1小时统计窗口
)

# 获取性能指标
metrics = translator.get_metrics()
print(f"平均响应时间: {metrics['avg_response_time']}ms")
print(f"请求成功率: {metrics['success_rate']}%")
print(f"缓存命中率: {metrics['cache_hit_rate']}%")
```

#### 缓存优化
```python
# 启用多级缓存
translator = create(
    enable_cache=True,
    cache_ttl=3600,
    cache_size=1000
)

# 自定义缓存实现
class RedisCache(CacheBackend):
    async def get(self, key: str):
        return await redis.get(key)
    
    async def set(self, key: str, value: any, ttl: int = None):
        await redis.set(key, value, ex=ttl)

translator = create(cache_backend=RedisCache())
```

### 6. 资源管理

#### 上下文管理器
```python
# 推荐：使用上下文管理器
with create() as translator:
    result = translator.translate("Hello", dest="zh")
    # 资源会自动清理

# 不推荐：手动管理
translator = create()
try:
    result = translator.translate("Hello", dest="zh")
finally:
    translator.cleanup()
```

#### 异步资源管理
```python
async def translate_with_resource_management():
    translator = AITranslator()
    try:
        await translator.initialize()
        result = await translator.translate("Hello", dest="zh")
        return result
    finally:
        await translator.cleanup()
```

#### 内存优化
```python
# 处理大文件
def process_large_file(file_path: str, dest: str = "zh"):
    with create() as translator:
        with open(file_path) as f:
            # 使用生成器逐行处理
            for line in f:
                result = translator.translate(line.strip(), dest=dest)
                yield result

# 定期清理缓存
translator.cache.clear()
```

### 7. 生产环境适配

#### Web 服务集成
```python
from fastapi import FastAPI
from aitrans import create

app = FastAPI()
translator = create(auto_preconnect=True)

@app.post("/translate")
async def translate_text(text: str, dest: str = "en"):
    try:
        result = translator.translate(text, dest=dest)
        return {"translated": result.text}
    except AIError as e:
        return {"error": str(e)}
```

#### 微服务架构
```python
# 健康检查
@app.get("/health")
async def health_check():
    try:
        is_healthy = translator.test_connection()
        metrics = translator.get_metrics()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "metrics": metrics
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

#### 容器化部署
```python
# 资源限制场景
translator = create(
    max_workers=os.cpu_count(),
    connection_timeout=30,
    read_timeout=60,
    retry_count=3
)
```


## 常见问题

### Q1: 如何处理大规模翻译任务？

```python
from aitrans import create
from concurrent.futures import ThreadPoolExecutor
import queue

def translate_large_dataset(file_path, dest="zh", batch_size=100):
    with create() as translator:
        # 创建任务队列
        task_queue = queue.Queue()
        
        # 读取文件并分批处理
        with open(file_path) as f:
            batch = []
            for line in f:
                batch.append(line.strip())
                if len(batch) >= batch_size:
                    task_queue.put(batch)
                    batch = []
            if batch:
                task_queue.put(batch)
        
        # 创建工作线程池
        with ThreadPoolExecutor(max_workers=5) as executor:
            while not task_queue.empty():
                batch = task_queue.get()
                results = translator.translate_batch(batch, dest=dest)
                process_results(results)
```

### Q2: 如何优化翻译质量？

```python
# 使用上下文翻译
context = "这是一篇技术文档。"
text = "它的实现很优雅。"
result = translator.translate_with_context(
    text,
    context=context,
    dest="en",
    quality_preference="high"
)

# 使用术语表
glossary = {
    "优雅": "elegant",
    "实现": "implementation"
}
result = translator.translate(
    text,
    dest="en",
    glossary=glossary
)
```

### Q3: 如何处理超时问题？

```python
# 设置合理的超时时间
translator = create(
    connection_timeout=30,
    read_timeout=60,
    retry_count=3
)

# 使用异步超时控制
async def translate_with_timeout(text, timeout=30):
    try:
        async with asyncio.timeout(timeout):
            return await translator.translate(text, dest="zh")
    except asyncio.TimeoutError:
        print("翻译超时")
```

## API 文档

### 核心类

#### AITranslator

异步翻译器类，提供所有翻译相关的异步方法。

```python
class AITranslator:
    async def translate(
        self,
        text: Union[str, List[str]],
        dest: str = "en",
        src: str = "auto",
        stream: bool = False,
        **kwargs
    ) -> Union[AITranslated, List[AITranslated], AsyncGenerator[str, None]]:
        """翻译文本"""
        pass

    async def translate_batch(
        self,
        texts: List[str],
        dest: str = "en",
        src: str = "auto",
        **kwargs
    ) -> List[AITranslated]:
        """批量翻译文本"""
        pass

    async def translate_with_style(
        self,
        text: str,
        style: str,
        dest: str = "en",
        src: str = "auto",
        **kwargs
    ) -> AITranslated:
        """带风格的翻译"""
        pass

    async def translate_with_context(
        self,
        text: str,
        context: str,
        dest: str = "en",
        src: str = "auto",
        **kwargs
    ) -> AITranslated:
        """带上下文的翻译"""
        pass
```

#### AITranslatorSync

同步翻译器类，提供所有翻译相关的同步方法。

```python
class AITranslatorSync:
    def translate(
        self,
        text: Union[str, List[str]],
        dest: str = "en",
        src: str = "auto",
        stream: bool = False,
        **kwargs
    ) -> Union[AITranslated, List[AITranslated], Generator[str, None, None]]:
        """翻译文本"""
        pass

    def translate_batch(
        self,
        texts: List[str],
        dest: str = "en",
        src: str = "auto",
        **kwargs
    ) -> List[AITranslated]:
        """批量翻译文本"""
        pass

    def translate_with_style(
        self,
        text: str,
        style: str,
        dest: str = "en",
        src: str = "auto",
        **kwargs
    ) -> AITranslated:
        """带风格的翻译"""
        pass

    def translate_with_context(
        self,
        text: str,
        context: str,
        dest: str = "en",
        src: str = "auto",
        **kwargs
    ) -> AITranslated:
        """带上下文的翻译"""
        pass
```

### 数据模型

#### AITranslated

翻译结果类，包含翻译结果的所有信息。

```python
class AITranslated:
    text: str            # 翻译后的文本
    src: str            # 源语言
    dest: str           # 目标语言
    confidence: float   # 翻译置信度
    raw_response: dict  # 原始 API 响应
```

### 工具类

#### DocumentTranslator

文档翻译器，支持各种格式的文档翻译。

```python
class DocumentTranslator:
    def translate_file(
        self,
        input_path: str,
        output_path: str,
        src: str = "auto",
        dest: str = "en",
        **kwargs
    ) -> None:
        """翻译文件"""
        pass
```

## 许可证

MIT License

## 贡献

欢迎提交 [Issue](https://github.com/kilolonion/aitrans/issues)