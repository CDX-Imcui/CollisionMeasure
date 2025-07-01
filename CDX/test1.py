import os

import cv2
import matplotlib
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib import pyplot as plt

from from_matrix import DepthBackProjector

WORK_DIR = os.path.join(os.path.dirname(__file__), 'WORK_DIR')
save_path = os.path.join(os.path.dirname(__file__), 'WORK_DIR', 'save_images')
os.makedirs(save_path, exist_ok=True)
image_path = os.path.join(os.path.dirname(__file__), 'WORK_DIR', 'input', 'image_00014.jpg')
image_name = os.path.basename(image_path)
base_name = os.path.splitext(image_name)[0]  # 获取不带扩展名的文件名

plane = [0.1084, 0.9936, -0.0301, 0.6306]
PointCoordinate = []
ChaCoordinate = []
Unit_distance_length = 1.44344028
# cv2.imread 默认以 BGR 格式读取图像，将图片从 BGR 转换为 RGB，以便 matplotlib 正确显示颜色
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 创建 Matplotlib 图形窗口，并设置响应更快的后端
plt.switch_backend('Qt5Agg')  # 使用Qt5Agg图形后端

# fig, ax = plt.subplots(figsize=(10, 8))  # 设置更大的图像尺寸
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[9, 1])
fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)  # 为顶部按钮预留空间

# # 注册工具
# # 自定义工具
# from matplotlib.backend_tools import ToolBase, ToolToggleBase
# matplotlib.rcParams['toolbar'] = 'toolmanager'  # 启用工具管理器
# class ClearPointsTool(ToolBase):
#     description = '清除所有点'
#     default_keymap = 'c'
#     def trigger(self, *args, **kwargs):
#         clear_points()
# class SaveImageTool(ToolBase):
#     def trigger(self, *args, **kwargs):
#         save_images()
#
# fig.canvas.manager.toolmanager.add_tool("ClearPoints", ClearPointsTool)
# fig.canvas.manager.toolbar.add_tool("ClearPoints", "navigation")
# fig.canvas.manager.toolmanager.add_tool("SaveImage", SaveImageTool)
# fig.canvas.manager.toolbar.add_tool("SaveImage", "navigation")

# 创建主图形区域和按钮区域
ax = fig.add_subplot(gs[0])  # 主图形区域
button_ax = fig.add_subplot(gs[1])  # 按钮区域
button_ax.axis('off')  # 隐藏按钮区域的坐标轴

ax.imshow(img_rgb)
ax.axis('off')  # 关闭坐标轴

# if Unit_distance_length is None:
#     return

projector = DepthBackProjector(WORK_DIR, plane)
projector.load_data(image_name)


def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # 获取点击的像素坐标 (x, y)
        # event.xdata 和 event.ydata 是浮点数，需要转换为整数
        # 对于图像，y 是行，x 是列
        x = int(event.xdata + 0.5)  # 添加0.5并取整以提高精度
        y = int(event.ydata + 0.5)

        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:  # 确保坐标在图片范围内
            # b, g, r = img[y, x]  # 获取该像素的颜色值
            # print(f"点击坐标 (X={x}, Y={y}) 的像素颜色 (RGB): ({r}, {g}, {b})")

            t, error = projector.pixel_to_world(x, y)  # t:x, y, z（未经投影的点）一维NumPy数组
            if (isinstance(t, (int, float)) and t == -1) or (
                    isinstance(t, np.ndarray) and np.all(t == -1)) and error == -1:
                print(f"点 ({x}, {y}) 在深度图中未找到对应的三维坐标")
                return
            ax.plot(x, y, 'r+', markersize=10)  # 在图像上显示十字标记
            plt.draw()  # 立即更新绘图
            ChaCoordinate.append([x, y])

            # # 确保t始终是一维NumPy数组
            # if not isinstance(t, np.ndarray):
            #     t = np.array(t, dtype=np.float64)

            # PointCoordinate.append(t)  # 将三维坐标添加到列表中
            # print("三维坐标 (world):",
            #       [np.round(point, 4).tolist() if hasattr(point, 'tolist') else point for point in
            #        PointCoordinate])
            # print("三维坐标 (world):", np.round(PointCoordinate, 4))
            # PointCoordinate中每有两个点，就绘制一条线，并标上距离
            if len(ChaCoordinate) % 2 == 0:
                # 绘制线段
                p1 = ChaCoordinate[-2]
                p2 = ChaCoordinate[-1]

                line, error = projector.compute_distance(p1, p2)  # 计算距离
                real_distance = line * Unit_distance_length  # 恢复成现实单位

                ax.plot([ChaCoordinate[-2][0], ChaCoordinate[-1][0]],
                        [ChaCoordinate[-2][1], ChaCoordinate[-1][1]], 'g-')
                # 计算线段中点位置用于放置距离标签
                mid_x = (ChaCoordinate[-2][0] + ChaCoordinate[-1][0]) / 2
                mid_y = (ChaCoordinate[-2][1] + ChaCoordinate[-1][1]) / 2
                # 在线段中点显示距离值
                ax.text(mid_x, mid_y, f"{real_distance:.2f}", color='red', fontsize=10,
                        ha='center', va='center')


cid = fig.canvas.mpl_connect('button_press_event', onclick)


def clear_points():
    """清空选点"""
    PointCoordinate = []
    ChaCoordinate = []
    # 重新读取并显示图片以清除所有标记
    ax.clear()
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis('off')
    plt.draw()  # 立即更新图像


def save_images():
    """保存图像和选点"""
    if not ChaCoordinate:
        print("没有选点，无法保存")
        return

    # 保存图像
    # save_path = os.path.join(save_path, f"{image_name}.jpg")
    fig.savefig(os.path.join(save_path, f"{base_name}.jpg"), bbox_inches='tight', pad_inches=0.1, dpi=600)
    print(f"图像已保存到")
    # 使用局部变量引用主窗口
    # main_window = self
    # QtCore.QTimer.singleShot(100, lambda: QtWidgets.QMessageBox.information(main_window, "保存成功",
    #                                                                         f"图像已保存到: {save_path}"))
    QtWidgets.QMessageBox.information(None, "保存成功", f"图像已保存到: {save_path}")

import matplotlib as mpl
mpl.rcParams['keymap.save'] = []
def on_key(event):
    if event.key == 'ctrl+c':  # 按下 C 键
        clear_points()
    elif event.key == 'ctrl+s':  # 按下 S 键
        save_images()
    elif event.key == 'q':  # 按下 Q 键退出
        plt.close(fig)
fig.canvas.mpl_connect('key_press_event', on_key)

# 分离交互区域，确保tight_layout不影响按钮位置
fig.canvas.draw()
# plt.tight_layout(left=0.05, right=0.95, bottom=0.1, top=0.9)  # 调整布局，但保留底部10%和顶部5%的空间给按钮
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
plt.show(block=True)  # 确保阻塞式显示

QtWidgets.QApplication.processEvents()
plt.show()
# 创建模态对话框效果，阻止与主窗口交互但不阻塞事件循环
fig.canvas.manager.window.setWindowModality(QtCore.Qt.ApplicationModal)
