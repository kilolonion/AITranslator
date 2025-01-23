from aitrans import AITranslatorSync

translator = AITranslatorSync()

text = "你好，世界"
result = translator.translate(text, dest='en', src='zh', stream=True)
for partial_result in result:
    print(f"当前翻译: {partial_result.text}")
    print(f"完成度: {len(partial_result.text)}/{len(text)}")

# 可能的输出：
# 当前翻译: Hello
# 完成度: 5/5
# 当前翻译: Hello,
# 完成度: 6/5
# 当前翻译: Hello, world
# 完成度: 11/5