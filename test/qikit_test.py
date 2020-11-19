# -*- coding: utf-8 -*-
'''
   @Time    : 2020/10/3 23:42
   @Author  : liuxu
   @File    : qikit_test.py
   @Software: PyCharm
'''

import sys
import random
from PySide2 import QtCore,QtWidgets,QtGui

import os
import PySide2


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("click me!")
        self.text = QtWidgets.QLabel("hello world!")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.magic)


    def magic(self):
        self.text.setText(random.choice(self.hello))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())