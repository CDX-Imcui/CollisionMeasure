import os
import shutil

import cv2
from PIL import Image
from PyQt5 import QtCore


# def convert_to_jpg(source_dir):
#     image_extensions = ['.png', '.bmp', '.tiff', '.jpeg', '.heic']  # 添加对HEIC的支持
#     # 遍历源目录中的所有文件
#     for filename in os.listdir(source_dir):
#         # 检查文件扩展名是否在列表中
#         if any(filename.lower().endswith(ext) for ext in image_extensions):
#             # 构建完整的文件路径
#             img_path = os.path.join(source_dir, filename)
#             output_path = os.path.join(source_dir, os.path.splitext(filename)[0] + '.jpg')
#             # 对于HEIC文件，使用Pillow读取
#             if filename.lower().endswith('.heic'):
#                 img = Image.open(img_path)
#                 img = img.convert("RGB")  # 确保转换为通用格式
#                 img.save(output_path, "JPEG")  # 直接保存为JPG
#             else:
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     continue  # 如果文件不是图像跳过
#                 cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 设置JPEG质量为95%
#
#             # 如果新的文件创建成功，删除原文件
#             if os.path.exists(output_path):
#                 os.remove(img_path)
#             else:
#                 print(f"Failed to convert {img_path}")


class AddImagesWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(list)  # 完成后传递图像文件路径列表

    def __init__(self, images, WORK_DIR):
        super(AddImagesWorker, self).__init__()
        self.images = images
        self.WORK_DIR = WORK_DIR

    def run(self):
        self.finished.emit(self.images)  # 先不着急搬，先显示出来 间隙去搬
        for file_path in self.images:  # 复制选择的每个图像文件到工作目录
            shutil.copy(file_path, self.WORK_DIR)

import cv2
import hyperlpr3 as lpr3
class Align_according_to_license_plate(QtCore.QThread):
    finished = QtCore.pyqtSignal(list)

    def __init__(self, images, WORK_DIR):
        super(Align_according_to_license_plate, self).__init__()
        self.images = images
        self.WORK_DIR = WORK_DIR

    def run(self):

        catcher = lpr3.LicensePlateCatcher()
        # load image
        image = cv2.imread("image_005.jpg")
        # print result
        print(catcher(image))#[['闽D2ER09', 0.99985033, 0, [479, 396, 578, 430]]]


        self.finished.emit(self.images)
