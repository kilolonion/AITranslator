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

### 基本用法

```python
from aitrans import AITranslatorSync

# 创建翻译器实例
translator = AITranslatorSync()

# 简单翻译
result = translator.translate("你好，世界！", dest="en")
print(result.text)  # 输出: Hello, world!

# 批量翻译
texts = ["你好", "世界"]
results = translator.translate_batch(texts, dest="en")
for result in results:
    print(result.text)
```

### 异步用法

```python
import asyncio
from aitrans import AITranslator

async def main():
    async with AITranslator() as translator:
        result = await translator.ai_translate("你好，世界！", dest="en")
        print(result.text)

asyncio.run(main())
```

### 上下文翻译

```python
translator = AITranslatorSync()
context = "这是关于计算机科学的讨论。"
result = translator.translate_with_context(
    "该算法的复杂度是 O(n)。",
    context=context,
    dest="en"
)
print(result.text)
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
- 更多语言支持中...

## 配置

在使用前，需要设置 API 密钥：

```python
import os
os.environ["ARK_API_KEY"] = "your-api-key"
```

或者在创建实例时传入：

```python
translator = AITranslatorSync(api_key="your-api-key")
```

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

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！ 