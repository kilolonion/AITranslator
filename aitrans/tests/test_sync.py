"""同步接口测试模块"""

import pytest
from aitrans.sync import AITranslatorSync, create
from aitrans.models import AITranslated
from aitrans.exceptions import AIError


@pytest.fixture
def translator():
    """创建同步翻译器实例"""
    with create() as translator:
        yield translator


def test_basic_translation(translator):
    """测试基础翻译功能"""
    result = translator.translate("Hello world!", dest="zh")
    assert isinstance(result, AITranslated)
    assert result.text


def test_quick_translate():
    """测试快速翻译功能"""
    result = AITranslatorSync.quick_translate("Hello", dest="zh")
    assert isinstance(result, str)
    assert result


def test_batch_translation(translator):
    """测试批量翻译功能"""
    texts = ["Hello", "Good morning", "Good night"]
    results = translator.translate_batch_sync(texts, dest="zh")
    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, AITranslated)
        assert result.text


def test_style_translation(translator):
    """测试风格化翻译"""
    result = translator.translate_with_style(
        "Hello world!",
        dest="zh",
        style="formal"
    )
    assert isinstance(result, AITranslated)
    assert result.text


def test_context_translation(translator):
    """测试上下文翻译"""
    result = translator.translate_with_context(
        "It is useful.",
        context="This is about programming.",
        dest="zh"
    )
    assert isinstance(result, AITranslated)
    assert result.text


def test_stream_translation(translator):
    """测试流式翻译"""
    text = "This is a test for streaming."
    for partial in translator.translate(text, dest="zh", stream=True):
        assert isinstance(partial, AITranslated)
        assert partial.text


def test_error_handling(translator):
    """测试错误处理"""
    with pytest.raises(AIError):
        translator.translate("", dest="invalid_lang")


def test_performance_config(translator):
    """测试性能配置"""
    translator.set_performance_config(
        max_workers=3,
        cache_ttl=1800
    )
    config = translator.get_config()
    assert config['performance_config']['max_workers'] == 3
    assert config['performance_config']['cache_ttl'] == 1800


def test_metrics(translator):
    """测试性能指标"""
    translator.translate("Test metrics", dest="zh")
    metrics = translator.get_metrics()
    assert metrics['total_requests'] > 0


def test_connection(translator):
    """测试连接功能"""
    assert translator.test_connection()


def test_factory_method():
    """测试工厂方法"""
    translator = create(performance_mode='fast')
    assert translator is not None
    result = translator.translate("Test factory", dest="zh")
    assert isinstance(result, AITranslated)
    assert result.text


def test_context_manager():
    """测试上下文管理器"""
    with AITranslatorSync() as translator:
        result = translator.translate("Test context manager", dest="zh")
        assert isinstance(result, AITranslated)
        assert result.text


if __name__ == "__main__":
    pytest.main([__file__])
