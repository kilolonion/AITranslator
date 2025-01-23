"""核心功能测试模块"""

from aitrans.exceptions import AIError
from aitrans.models import AITranslated, AIDetected
from aitrans.core import AITranslator
import os
import sys
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


@pytest_asyncio.fixture
async def translator():
    """创建翻译器实例"""
    translator = AITranslator()
    await translator.initialize()
    yield translator
    await translator.cleanup()


@pytest.mark.asyncio
async def test_basic_translation(translator):
    """测试基础翻译功能"""
    result = await translator.translate("Hello world!", dest="zh")
    assert isinstance(result, AITranslated)
    assert result.src == "en"
    assert result.dest == "zh"
    assert result.text


@pytest.mark.asyncio
async def test_batch_translation(translator):
    """测试批量翻译功能"""
    texts = ["Hello", "Good morning", "Good night"]
    results = await translator.translate(texts, dest="zh")
    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, AITranslated)
        assert result.text


@pytest.mark.asyncio
async def test_language_detection(translator):
    """测试语言检测功能"""
    result = await translator.detect("Hello world!")
    assert isinstance(result, AIDetected)
    assert result.lang == "en"
    assert 0 <= result.confidence <= 1


@pytest.mark.asyncio
async def test_stream_translation(translator):
    """测试流式翻译功能"""
    text = "This is a test text for streaming translation."

    # 创建多个模拟响应
    mock_responses = []
    for char in "测试翻译":
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].delta.content = char
        mock_responses.append(mock_response)

    # 模拟API调用
    with patch('aitrans.core.AITranslator._get_client') as mock_client:
        mock_client.return_value = AsyncMock()
        mock_client.return_value.chat = AsyncMock()
        mock_client.return_value.chat.completions = AsyncMock()
        mock_client.return_value.chat.completions.create = AsyncMock()
        mock_client.return_value.chat.completions.create.return_value = AsyncMock()
        mock_client.return_value.chat.completions.create.return_value.__aiter__.return_value = mock_responses

        result = await translator.translate(text, dest="zh", stream=True)
        full_text = ""
        async for partial in result:
            assert isinstance(partial, str)
            full_text += partial

        assert len(full_text) > 0
        assert isinstance(full_text, str)
        assert full_text == "测试翻译"


@pytest.mark.asyncio
async def test_error_handling(translator):
    """测试错误处理"""
    with pytest.raises(AIError):
        await translator.translate("", dest="invalid_lang")


@pytest.mark.asyncio
async def test_style_translation(translator):
    """测试风格化翻译"""
    text = "Hello world!"
    result = await translator.translate_with_style(
        text, dest="zh", style="formal"
    )
    assert isinstance(result, AITranslated)
    assert result.text


@pytest.mark.asyncio
async def test_context_translation(translator):
    """测试上下文翻译"""
    context = "This is about programming."
    text = "It is very useful."
    result = await translator.translate_with_context(
        text, context, dest="zh"
    )
    assert isinstance(result, AITranslated)
    assert result.text


@pytest.mark.asyncio
async def test_translation_recovery(translator):
    """测试翻译恢复机制"""
    text = "Test recovery"
    result = await translator.translate_with_recovery(
        text, dest="zh"
    )
    assert isinstance(result, AITranslated)
    assert result.text


@pytest.mark.asyncio
async def test_performance_metrics(translator):
    """测试性能指标"""
    # 执行一些翻译操作
    await translator.translate("Hello", dest="zh")
    metrics = translator.metrics.get_metrics()
    assert "total_requests" in metrics
    assert "success_rate" in metrics


@pytest.mark.asyncio
async def test_cache_functionality(translator):
    """测试缓存功能"""
    text = "Cache test"
    # 第一次翻译
    result1 = await translator.translate(text, dest="zh")
    # 第二次翻译（应该从缓存获取）
    result2 = await translator.translate(text, dest="zh")
    assert result1.text == result2.text


if __name__ == "__main__":
    pytest.main([__file__])
