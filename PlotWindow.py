import os

import cv2
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseEvent

from from_matrix import DepthBackProjector


class PlotWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, WORK_DIR=None, image_path=None, plane=None, unit_distance=None):
        super().__init__(parent)
        self.setWindowTitle("图像测量")
        self.resize(1000, 800)

        # 初始化属性
        self.WORK_DIR = WORK_DIR
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        self.plane = plane
        self.unit_distance = unit_distance
        self.projector = DepthBackProjector(WORK_DIR, self.plane)
        self.projector.load_data(self.image_name)

        self.cha_coords = []

        # 设置布局
        layout = QtWidgets.QVBoxLayout(self)

        # 创建 matplotlib 图
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.img = cv2.imread(self.image_path)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(self.img_rgb)
        self.ax.axis('off')

        self.canvas.mpl_connect("button_press_event", self.onclick)
        layout.addWidget(self.canvas)

        # 添加按钮
        btn_layout = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("清除点")
        self.clear_btn.clicked.connect(self.clear_points)
        btn_layout.addWidget(self.clear_btn)

        self.save_btn = QtWidgets.QPushButton("保存图像")
        self.save_btn.clicked.connect(self.save_image)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def onclick(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata + 0.5)  # 添加0.5并取整以提高精度
        y = int(event.ydata + 0.5)
        if 0 <= y < self.img.shape[0] and 0 <= x < self.img.shape[1]:
            coord, error = self.projector.pixel_to_world(x, y)
            if isinstance(coord, (int, float)) and coord == -1:
                return
            self.ax.plot(x, y, 'r+', markersize=10)
            self.cha_coords.append([x, y])
            if len(self.cha_coords) % 2 == 0:
                p1, p2 = self.cha_coords[-2], self.cha_coords[-1]
                dist, _ = self.projector.compute_distance(p1, p2)
                real_dist = dist * self.unit_distance
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-')
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                self.ax.text(mx, my, f"{real_dist:.2f}", color='red', fontsize=10, ha='center', va='center')

        self.canvas.draw()

    def clear_points(self):
        self.cha_coords = []
        self.ax.clear()
        self.ax.imshow(self.img_rgb)
        self.ax.axis('off')
        self.canvas.draw()

    def save_image(self):
        save_path = os.path.join(self.WORK_DIR, "save_images", f"{self.image_name}.jpg")
        self.figure.savefig(save_path, dpi=600, bbox_inches='tight')
        QtWidgets.QMessageBox.information(self, "保存成功", f"图像已保存至：{save_path}")
