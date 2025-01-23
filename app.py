import sys
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QAction
from threading import Thread
from PySide6.QtCore import Signal, QObject, QTimer
from aitrans import AITranslatorSync
import dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


class Signals(QObject):
    translation_done = Signal(str)
    preconnect_done = Signal()
    progress_updated = Signal(int)  # 添加进度信号
    error_occurred = Signal(str)    # 添加错误信号


class MainWindow():
    def __init__(self):
        try:
            # 创建全局翻译器实例
            self.translator = AITranslatorSync().__enter__()

            # 创建UI加载器
            loader = QUiLoader()
            self.ui = loader.load('untitled.ui')
            self.signals = Signals()

            # 初始化UI组件
            self._init_ui_components()

            # 使用QTimer延迟启动预热
            QTimer.singleShot(100, self.start_preconnect)

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    def _init_ui_components(self):
        """初始化UI组件"""
        try:
            # 设置语言选项
            supported_langs = self.translator.get_supported_languages()
            self.ui.targetlang.addItems(
                [x for x in supported_langs.values() if x != '自动检测'])
            self.ui.sourcelang.addItems(supported_langs.values())

            # 初始化菜单
            self._init_menus()

            # 连接信号
            self._connect_signals()

        except Exception as e:
            logger.error(f"UI组件初始化失败: {str(e)}")
            raise

    def _init_menus(self):
        """初始化菜单"""
        try:
            # 为每个菜单创建动作
            action1 = QAction('主页面', self.ui)
            action2 = QAction('设置', self.ui)
            action3 = QAction('帮助', self.ui)
            action4 = QAction('历史', self.ui)

            # 将动作添加到对应的菜单
            self.ui.menu1.addAction(action1)
            self.ui.menu2.addAction(action2)
            self.ui.menu3.addAction(action3)
            self.ui.menu4.addAction(action4)

            # 连接动作的触发信号到对应的槽函数
            action1.triggered.connect(
                lambda: self.ui.stackedWidget.setCurrentIndex(0))
            action4.triggered.connect(
                lambda: self.ui.stackedWidget.setCurrentIndex(1))
            action2.triggered.connect(
                lambda: self.ui.stackedWidget.setCurrentIndex(2))
            action3.triggered.connect(
                lambda: self.ui.stackedWidget.setCurrentIndex(3))
        except Exception as e:
            logger.error(f"菜单初始化失败: {str(e)}")
            raise

    def _connect_signals(self):
        """连接信号和槽"""
        try:
            # 连接翻译按钮的点击事件
            self.ui.translatebutton.clicked.connect(self.start_translation)

            # 连接信号到槽函数
            self.signals.translation_done.connect(
                self.update_translation_result)
            self.signals.progress_updated.connect(self._update_progress)
            self.signals.error_occurred.connect(self._handle_error)
        except Exception as e:
            logger.error(f"信号连接失败: {str(e)}")
            raise

    def _update_progress(self, value):
        """更新进度条"""
        try:
            self.ui.progressBar.setValue(value)
        except Exception as e:
            logger.error(f"进度更新失败: {str(e)}")

    def _handle_error(self, error_msg):
        """处理错误"""
        logger.error(f"翻译错误: {error_msg}")
        self.ui.translatedtext.setPlainText(f"错误: {error_msg}")
        self.ui.translatebutton.setEnabled(True)
        self.ui.progressBar.setValue(0)

    def start_preconnect(self):
        """在后台线程中预热翻译器"""
        try:
            thread = Thread(target=self._preconnect_task)
            thread.daemon = True  # 设置为守护线程
            thread.start()
        except Exception as e:
            logger.error(f"预热启动失败: {str(e)}")
            self.signals.error_occurred.emit(str(e))

    def _preconnect_task(self):
        """预热任务"""
        try:
            logger.info("开始预热翻译器")
            self.translator.preconnect()
            logger.info("翻译器预热完成")
        except Exception as e:
            logger.error(f"预热失败: {str(e)}")
            self.signals.error_occurred.emit(str(e))

    def start_translation(self):
        """开始翻译"""
        try:
            self.ui.translatebutton.setEnabled(False)
            self.ui.progressBar.setValue(0)

            thread = Thread(target=self.translate_task)
            thread.daemon = True
            thread.start()
        except Exception as e:
            logger.error(f"翻译启动失败: {str(e)}")
            self.signals.error_occurred.emit(str(e))
            self.ui.translatebutton.setEnabled(True)

    def translate_task(self):
        """翻译任务"""
        try:
            # 获取输入
            text = self.ui.originaltext.toPlainText()
            if not text.strip():
                raise ValueError("请输入要翻译的文本")

            # 获取语言设置
            dest = self.translator.get_language_code(
                self.ui.targetlang.currentText()) or 'en'
            src = self.translator.get_language_code(
                self.ui.sourcelang.currentText()) or 'auto'

            # 获取性能模式
            model = {
                '平衡': 'balanced',
                '快速': 'fast',
                '精准': 'precise'
            }.get(self.ui.modelgroup.checkedButton().text(), 'balanced')

            # 设置性能配置
            self.translator.set_performance_config(performance_mode=model)

            # 开始流式翻译
            result = self.translator.translate(
                text, src=src, dest=dest, stream=True)
            total_length = len(text)
            current_length = 0

            for partial_result in result:
                current_length = len(partial_result.text)
                progress = min(int((current_length / total_length) * 100), 100)
                self.signals.progress_updated.emit(progress)
                self.signals.translation_done.emit(partial_result.text)

        except Exception as e:
            logger.error(f"翻译过程出错: {str(e)}")
            self.signals.error_occurred.emit(str(e))
        finally:
            self.signals.progress_updated.emit(100)

    def update_translation_result(self, result):
        """更新翻译结果"""
        try:
            self.ui.translatedtext.setPlainText(result)
            self.ui.translatebutton.setEnabled(True)
        except Exception as e:
            logger.error(f"结果更新失败: {str(e)}")
            self.signals.error_occurred.emit(str(e))

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'translator'):
                self.translator.__exit__(None, None, None)
        except Exception as e:
            logger.error(f"资源清理失败: {str(e)}")


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.ui.show()

        exit_code = app.exec()

        # 清理资源
        window.cleanup()

        sys.exit(exit_code)

    except Exception as e:
        logger.critical(f"应用程序异常退出: {str(e)}")
        sys.exit(1)
