import os

from PyQt5 import QtWidgets, QtCore


class Clickable_Label(QtWidgets.QLabel):
    # 自定义信号，点击时发送图片路径
    clicked = QtCore.pyqtSignal(str)

    def __init__(self, image_path=""):
        super().__init__()
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        self.setCursor(QtCore.Qt.PointingHandCursor)  # 设置鼠标指针为手型

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # 发射信号并传递图片路径
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)
