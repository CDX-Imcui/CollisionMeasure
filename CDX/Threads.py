import os
import shutil
import time

import cv2
from PIL import Image
from PyQt5 import QtCore, QtWidgets
import cv2
import hyperlpr3 as lpr3

from CDX.colmap_pipeline import Colmap
from from_matrix import DepthBackProjector


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


class Parsing_video(QtCore.QThread):
    finished = QtCore.pyqtSignal(list)

    def __init__(self, video_path, frames_dir, image_paths):
        super(Parsing_video, self).__init__()
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.image_paths = image_paths

    def resize_keep_aspect_ratio(self, img, max_size=1280):
        """调整图像大小，保持横纵比，最长边不超过max_size"""
        height, width = img.shape[0], img.shape[1]
        # 如果图像已经小于等于最大尺寸，则不需要调整
        if width <= max_size and height <= max_size:
            return img
        # 计算缩放因子
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # 调整图像大小
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_img

    def run(self):
        video_capture = cv2.VideoCapture(self.video_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // 100, 1)  # 计算帧间隔
        frame_count = 0
        saved_count = 0
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            frame_count += 1
            # 每隔几帧保存一次(可以根据需要调整间隔)
            if frame_count % step == 0:
                frame_path = os.path.join(self.frames_dir, f"image_{saved_count:05d}.jpg")
                frame = self.resize_keep_aspect_ratio(frame, max_size=1280)  # 调整图像大小
                cv2.imwrite(frame_path, frame)
                self.image_paths.append(frame_path)
                saved_count += 1
        video_capture.release()
        print(f"从视频中提取了 {len(self.image_paths)} 帧")
        self.finished.emit(self.image_paths)


class ColmapWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str)
    # error = QtCore.pyqtSignal(str)  # 错误信号
    log_message = QtCore.pyqtSignal(str)  # 进度消息信号

    def __init__(self, source_path, colmap_cmd="colmap", use_gpu=1):
        super(ColmapWorker, self).__init__()
        self.Colmap = Colmap(source_path, colmap_cmd, use_gpu)

    def run(self):
        try:
            self.log_message.emit('feature extract')  # 发送进度信息
            print('feature extract')
            start=time.time()
            self.Colmap.feature_extraction()
            self.log_message.emit('feature matching')  # 发送进度信息
            print('feature matching')
            self.Colmap.feature_matching()
            self.log_message.emit('reconstruction')  # 发送进度信息
            print('reconstruction')
            self.Colmap.sparse_reconstruction()
            self.log_message.emit('image undistorter')  # 发送进度信息
            print('image undistorter')
            self.Colmap.image_undistortion()
            self.log_message.emit('patch match stereo')  # 发送进度信息
            print('patch match stereo')
            self.Colmap.dense_stereo()
            self.log_message.emit('stereo fusion')  # 发送进度信息
            print('stereo fusion')
            self.Colmap.stereo_fusion()
            end = time.time()
            self.log_message.emit(
                f'COLMAP reconstruction completed successfully in {int((end - start) // 60)}分{int((end - start) % 60)}秒.')

            self.finished.emit(self.Colmap.point_cloud)  # 完成后去触发显示结果'point_cloud.ply'
        except Exception as e:
            # self.error.emit(str(e))
            self.log_message.emit(str(e))


class Align_according_to_LicensePlate_Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(float)

    def __init__(self, images, WORK_DIR):
        super(Align_according_to_LicensePlate_Worker, self).__init__()
        self.image_paths = images
        self.WORK_DIR = WORK_DIR

    def get_long_edge_points(self, box):
        x1, y1, x2, y2 = box
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width >= height:
            # 水平长边，返回上边两个顶点
            return [(x1, y1), (x2, y1)]
        else:
            # 垂直长边，返回左边两个顶点
            return [(x1, y1), (x1, y2)]

    def run(self):
        catcher = lpr3.LicensePlateCatcher()
        projector = DepthBackProjector(self.WORK_DIR)
        # distances = []
        # 存储误差最小的距离数据
        real_distances = None
        min_error = float('inf')  # 初始化为最大值
        # 遍历image
        for image_path in self.image_paths:
            result = catcher(cv2.imread(image_path))  # [['闽D2ER09', 0.99985033, 0, [479, 396, 578, 430]]]
            if not result or result[0][1] < 0.5:
                continue
            box = result[0][3]
            points = self.get_long_edge_points(box)

            if projector.load_data(os.path.basename(image_path)) == -1:  # 图像数据不在images_bin
                continue
            result = projector.compute_distance(points[0], points[1])  # 计算车牌两个点之间的距离

            # 检查返回值类型，处理计算失败的情况
            if isinstance(result, int) and result == -1:
                continue
            distance, error = result  # 如果计算成功，解包返回的距离和误差

            if error < min_error:
                min_error = error
                real_distances = distance
                print(f"更新最小误差: {min_error}, 对应距离: {real_distances}, 图像: {os.path.basename(image_path)}")
        self.finished.emit(real_distances)
