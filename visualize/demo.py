import os
import sys
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QEvent, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QComboBox, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon, QDesktopServices
import platform
from pathlib import Path
from transformers import PLBartTokenizer, PretrainedConfig
import torch

fp = Path(__file__)
sys.path.append(str(fp.parent.parent))

from dual_model import DualModel, PLBartForConditionalGeneration
from utils import load_json

java_checkpoint_dir = "./result_models/checkpoint-290400"
python_checkpoint_dir = "./result_models/py_model"


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def load_model(checkpoint_dir, code_lang):
    cs_model = PLBartForConditionalGeneration.from_pretrained(
            "./pretrained_cache/plbart-base")
    cs_tokenizer = PLBartTokenizer.from_pretrained("./pretrained_cache/plbart-base",
                                                    src_lang = code_lang,
                                                    tgt_lang = "__en_XX__")
    cg_model = PLBartForConditionalGeneration.from_pretrained(
        "./pretrained_cache/plbart-base")
    cg_tokenizer = PLBartTokenizer.from_pretrained("./pretrained_cache/plbart-base",
                                                    src_lang = "__en_XX__",
                                                    tgt_lang = code_lang)
    config_dict = load_json(os.path.join(checkpoint_dir, "config.json"))
    config = PretrainedConfig.from_dict(config_dict)
    dual_model = DualModel(config=config,
                           cs_model=cs_model,
                           cg_model=cg_model,
                           cs_tokenizer=cs_tokenizer,
                           cg_tokenizer=cg_tokenizer,
                           device=device)
    model_state_dir = os.path.join(checkpoint_dir, "pytorch_model.bin")
    dual_model.load_state_dict(torch.load(model_state_dir))
    return dual_model

class CodeExplanationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.java_model = load_model(java_checkpoint_dir, "java")
        self.python_model = load_model(python_checkpoint_dir, "python")

    def initUI(self):
        # 创建主布局
        main_layout = QVBoxLayout()

        # 创建上方水平布局
        top_layout = QHBoxLayout()

        # 创建左侧布局
        left_layout = QVBoxLayout()

        # 创建语言选择菜单
        self.language_combo_box = QComboBox()
        self.language_combo_box.addItem("Java")
        self.language_combo_box.addItem("Python")
        self.language_combo_box.currentIndexChanged.connect(self.updateCodeExplanation)
        left_layout.addWidget(self.language_combo_box)

        # 创建可编辑文本框, 存放代码
        self.code_text_edit = QTextEdit()
        self.code_text_edit.setReadOnly(False)
        self.code_text_edit.setStyleSheet("font-size:12pt")
        left_layout.addWidget(self.code_text_edit)

        top_layout.addLayout(left_layout)

        # 创建右侧布局
        right_layout = QVBoxLayout()

        # 创建语言解释选择菜单
        self.explanation_language_combo_box = QComboBox()
        self.explanation_language_combo_box.addItem("English (En)")
        self.explanation_language_combo_box.currentIndexChanged.connect(self.updateCodeExplanation)
        right_layout.addWidget(self.explanation_language_combo_box)

        # 创建可编辑文本框, 存放代码摘要结果
        self.output_text_label = QTextEdit()
        self.output_text_label.setReadOnly(False)
        self.output_text_label.setStyleSheet("font-size:12pt")
        right_layout.addWidget(self.output_text_label)

        top_layout.addLayout(right_layout)

        # 在主布局中添加上方水平布局
        main_layout.addLayout(top_layout)

        # 创建翻译按钮布局
        translate_layout = QHBoxLayout()

        # 创建左侧翻译按钮
        left_translate_button = QPushButton("代码->描述", self)
        left_translate_button.clicked.connect(self.updateCodeExplanation)
        translate_layout.addWidget(left_translate_button)

        # 创建右侧翻译按钮
        right_translate_button = QPushButton("描述->代码", self)
        right_translate_button.clicked.connect(self.updateCodeGeneration)
        translate_layout.addWidget(right_translate_button)

        # 将翻译按钮布局添加到主布局的最下方
        main_layout.addLayout(translate_layout)

        self.setLayout(main_layout)

        self.setGeometry(100, 100, 800, 400)
        self.setWindowTitle('Code Explanation App')
        self.show()

    def updateCodeExplanation(self):
        input_text = self.code_text_edit.toPlainText()
        selected_language = self.language_combo_box.currentText()
        
        explaination_text = ""
        # 在这里根据用户输入的文本和选择的语言更新摘要(右侧的文本框)内容
        try:
            if selected_language == "Java":
                explaination_text = self.java_model.cs_generate(input_text, 1)
            else:
                explaination_text = self.python_model.cs_generate(input_text, 1)
            
            self.output_text_label.setText(explaination_text)
        
        except:
            QMessageBox.information(None, "WARNING", "出错了, 请检查输入内容后重试")
        
        

    def updateCodeGeneration(self):
        input_text = self.output_text_label.toPlainText()
        selected_language = self.language_combo_box.currentText()
        
        generate_code = ""
        # 在这里根据用户输入的文本和选择的语言更新代码(左侧的文本框)内容
        try:
            if selected_language == "Java":
                generate_code = self.java_model.cg_generate(input_text, 1)
            else:
                generate_code = self.python_model.cg_generate(input_text, 1)
            generate_code = generate_code[0].replace("DCNL", "\n").replace("DCSP", "    ")
            self.code_text_edit.setText(generate_code)
        
        except:
            QMessageBox.information(None, "WARNING", "出错了, 请检查输入内容后重试")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CodeExplanationApp()
    sys.exit(app.exec_())
