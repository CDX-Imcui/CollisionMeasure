# 图片展示FLOOR_COLOR = [80, 50, 50]

import cv2
import matplotlib.pyplot as plt
import numpy as np


def onclick(event):
    """
    处理鼠标点击事件的回调函数。
    """
    if event.xdata is not None and event.ydata is not None:
        # 获取点击的像素坐标 (x, y)
        # 注意：event.xdata 和 event.ydata 是浮点数，需要转换为整数
        # 对于图像，y 是行，x 是列
        x = int(event.xdata + 0.5)  # 添加0.5并取整以提高精度
        y = int(event.ydata + 0.5)

        # 确保坐标在图片范围内
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            # 获取该像素的颜色值
            # OpenCV 读取的图片是 BGR 格式，需要转换为 RGB
            b, g, r = img[y, x]
            print(f"点击坐标 (X={x}, Y={y}) 的像素颜色 (RGB): ({r}, {g}, {b})")

            # 在图像上显示十字标记
            ax.plot(x, y, 'r+', markersize=10)
            plt.draw()  # 立即更新绘图
        else:
            print(f"点击位置 ({x}, {y}) 超出图片范围。")


def pick_color_from_image(image_path):
    """
    显示图片并允许用户点击获取像素颜色。
    """
    global img, ax  # 将 img 和 ax 声明为全局变量，以便 onclick 函数可以访问
    img = cv2.imread(image_path)

    if img is None:
        print(f"错误：无法读取图片 '{image_path}'。请检查路径或文件是否存在。")
        return

    # cv2.imread 默认以 BGR 格式读取图像，将图片从 BGR 转换为 RGB，以便 matplotlib 正确显示颜色
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 创建 Matplotlib 图形窗口，并设置响应更快的后端
    plt.switch_backend('TkAgg')  # 使用更快的图形后端
    fig, ax = plt.subplots(figsize=(10, 8))  # 设置更大的图像尺寸
    ax.imshow(img_rgb)
    ax.set_title(f"点击图片获取像素颜色 - {image_path.split('/')[-1]}")
    ax.axis('off')  # 关闭坐标轴

    # 连接鼠标点击事件到 onclick 函数
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # 添加键盘事件处理（按q退出）
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # 设置紧凑布局以最大化图像显示区域
    plt.tight_layout()

    print("图片已打开。请点击图片上的任意位置来获取像素颜色。")
    print("按 'q' 键或关闭窗口退出。")

    plt.show(block=True)  # 确保阻塞式显示


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        image_to_load = sys.argv[1]
    else:
        image_to_load = "ADE_val_00001519.png"

    # 检查文件是否存在
    if not os.path.exists(image_to_load):
        print(f"警告：文件 '{image_to_load}' 不存在，请提供正确的图片路径")
        sys.exit(1)

    pick_color_from_image(image_to_load)
