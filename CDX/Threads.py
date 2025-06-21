import os
import shutil
import time

import numpy as np
from PyQt5 import QtCore
import cv2
import hyperlpr3 as lpr3

from colmap_pipeline import Colmap
from from_matrix import DepthBackProjector
from SemanticSegmentation import SemanticSegmentation


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

    def __init__(self, images, WORK_DIR):
        super(AddImagesWorker, self).__init__()
        self.images = images
        self.WORK_DIR = WORK_DIR

    def run(self):
        for file_path in self.images:  # 复制选择的每个图像文件到工作目录
            shutil.copy(file_path, os.path.join(self.WORK_DIR, "input"))


class Parsing_video(QtCore.QThread):
    finished = QtCore.pyqtSignal(list)
    now_sizeSignal = QtCore.pyqtSignal(int)

    def __init__(self, video_path, WORK_DIR, image_paths, max_size=1920):
        super(Parsing_video, self).__init__()
        self.video_path = video_path
        self.WORK_DIR = WORK_DIR
        self.frames_dir = os.path.join(WORK_DIR, "input")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.image_paths = image_paths
        self.max_size = max_size

    def resize_keep_aspect_ratio(self, img, max_size=1920):
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
        now_size = None

        while True:
            success, frame = video_capture.read()
            if not success:
                break
            frame_count += 1
            # 每隔几帧保存一次(可以根据需要调整间隔)
            if frame_count % step == 0:
                frame_path = os.path.join(self.frames_dir, f"image_{saved_count:05d}.jpg")
                frame = self.resize_keep_aspect_ratio(frame, max_size=self.max_size)  # 调整图像大小
                now_size = max(frame.shape[0], frame.shape[1])  # 获取当前帧的最大尺寸
                cv2.imwrite(frame_path, frame)
                self.image_paths.append(frame_path)
                saved_count += 1
        video_capture.release()
        if now_size > self.max_size:
            now_size = self.max_size
        print(f"从视频中提取了 {len(self.image_paths)} 帧")
        self.finished.emit(self.image_paths)
        self.now_sizeSignal.emit(now_size)


class ColmapWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str)
    log_message = QtCore.pyqtSignal(str)  # 进度消息信号

    def __init__(self, source_path):
        super(ColmapWorker, self).__init__()
        self.Colmap = Colmap(source_path)

    def run(self):
        try:
            self.log_message.emit('feature extract')  # 发送进度信息
            print('feature extract')
            start = time.time()
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


from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor


class SemanticSegmentation_Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool, list, int, str, list, list)  # 成功与否，平面方程系数
    log_message = QtCore.pyqtSignal(str)
    updateView = QtCore.pyqtSignal(list)

    def __init__(self, image_paths, WORK_DIR,now_size):
        super(SemanticSegmentation_Worker, self).__init__()
        self.image_paths = image_paths
        self.WORK_DIR = WORK_DIR
        self.now_size = now_size

    def remove_outliers_statistical(self, point_array, k=10, std_ratio=2.0):
        """
        使用统计滤波去除离群点
        :param point_array: numpy array, shape (N, 3)
        :param k: 邻居数量
        :param std_ratio: 距离标准差倍数阈值
        :return: 去噪后的点集
        """
        if not isinstance(point_array, np.ndarray):
            point_array = np.array(point_array)

        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(point_array)
        distances, _ = neighbors.kneighbors(point_array)
        mean_dists = distances[:, 1:].mean(axis=1)  # 去掉自身

        threshold = mean_dists.mean() + std_ratio * mean_dists.std()
        mask = mean_dists < threshold
        return point_array[mask]

    def fit_plane_ransac(self, points, residual_threshold=0.01):
        """
        对去噪后的三维点进行 RANSAC 平面拟合
        ax + by + cz + d = 0
        :param points: numpy array, shape (N, 3)
        :return: normal vector (a, b, c), d, inlier mask
        """
        points = np.array(points)
        X = points[:, :2]
        y = points[:, 2]

        model = RANSACRegressor(residual_threshold=residual_threshold)
        model.fit(X, y)

        a, b = model.estimator_.coef_
        c = -1.0
        d = model.estimator_.intercept_

        # ax + by + cz + d = 0 → ax + by - z + d = 0 → normal = [a, b, -1]
        normal = np.array([a, b, c])
        norm = np.linalg.norm(normal)
        normal /= norm
        d /= norm

        # 内点掩码
        inlier_mask = model.inlier_mask_
        return normal, d, inlier_mask

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
        try:
            catcher = lpr3.LicensePlateCatcher()
            projector = DepthBackProjector(self.WORK_DIR)
            best_image_id = None
            best_image_name = None
            best_image_path = None
            best_points = None
            min_error = float('inf')  # 初始化为最大值

            # 遍历image
            paths_to_remove = []
            for image_path in self.image_paths:
                print("开始遍历图像:", image_path)
                result = catcher(cv2.imread(image_path))  # [['闽D2ER09', 0.99985033, 0, [479, 396, 578, 430]]]
                if not result or result[0][1] < 0.5:
                    continue
                box = result[0][3]
                points = self.get_long_edge_points(box)
                print("points", points)

                if projector.load_data(os.path.basename(image_path)) is False:  # 图像数据不在images_bin
                    print("projector.load_data(os.path.basename(image_path)) is False")
                    paths_to_remove.append(image_path)  # 记录需要删除的路径
                    continue
                print("projector.load_data(os.path.basename(image_path)) is True")
                distance, error = projector.compute_distance(points[0], points[1])  # 计算车牌两个点之间的距离
                print("distance", distance, "   error", error)
                # if result < min_error:
                #     min_error = result
                # 检查返回值类型，处理计算失败的情况
                if distance == -1 and error == -1:
                    continue

                if error < min_error:
                    min_error = error
                    real_distances = distance
                    best_image_name = os.path.basename(image_path)  # 'image_00059.jpg'
                    best_image_path = image_path  # 完整路径
                    best_image_id = best_image_name.split('_')[1].split('.')[0]  # '004'
                    best_points = points  # 记录最佳车牌点对 [(x1, y1), (x1, y2)]
                    print(f"更新最小误差: {min_error}, 对应距离: {real_distances}, 图像: {best_image_name}")
            # 循环结束后，一次性删除所有无效路径
            print("循环结束后，一次性删除所有无效路径")
            for path in paths_to_remove:
                if path in self.image_paths:
                    self.image_paths.remove(path)

            ####################################
            segmentation = SemanticSegmentation(best_image_path)
            images = segmentation.run()
            # image = images[best_image_id]  # 获取最优图像
            image = images[0]  # 获取最优图像
            FLOOR_COLOR = [140, 140, 140]  # 注意顺序是 RGB
            ground_pixels = []

            # 根据分割图像，20个像素为步长遍历获取地面的特征点集合
            print("# 根据分割图像，20个像素为步长遍历获取地面的特征点集合")
            for y in range(0, image.shape[0], 20):
                for x in range(0, image.shape[1], 20):
                    pixel = image[y, x, :]  # 获取该点 RGB
                    if np.array_equal(pixel, FLOOR_COLOR):
                        ground_pixels.append((int(round(x * (self.now_size / 640))), int(round(y * (self.now_size / 640)))))  # 记录图像坐标

            projector.load_data(best_image_name)
            # 遍历ground_pixels，得到三维坐标列表
            print("# 遍历ground_pixels，得到三维坐标列表")
            _3Dcoordinates = []
            for pixel in ground_pixels:
                l, r = projector.pixel_to_world(pixel[0], pixel[1])
                if (isinstance(l, np.ndarray) and l.size == 0) or (
                        isinstance(r, np.ndarray) and r.size == 0) or np.array_equal(l, -1) or np.array_equal(r,
                                                                                                              -1):  # 如果像素点超出范围或深度无效
                    continue
                x, y, z = l
                _3Dcoordinates.append([x, y, z])  # 计算每个点的三维坐标

            # 根据_3Dcoordinates地面特定点拟合地面方程
            print("# 根据_3Dcoordinates地面特定点拟合地面方程")
            cleaned_points = self.remove_outliers_statistical(_3Dcoordinates, k=10, std_ratio=2.0)
            normal, d, inliers = self.fit_plane_ransac(cleaned_points)
            a, b, c = normal
            print(f"RANSAC 拟合平面: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            print(
                f"内点数量: {np.sum(inliers)}, 平均拟合误差: {np.mean(np.abs(cleaned_points[inliers] @ normal + d)):.4f}")

            self.finished.emit(True, [a, b, c, d], int(best_image_id), best_image_name, best_points, _3Dcoordinates)
            self.updateView.emit(self.image_paths)  # 更新视图
            for i in _3Dcoordinates:
                print(i, " - ")
            print("ground_pixels 地面特征点数量:", len(ground_pixels))
            for i in ground_pixels:
                print(i, " - ")
        except Exception as e:
            self.log_message.emit(str(e))
            # self.finished.emit(False, [0, 0, 0, 0], -1, "",[])  # 如果发生错误，返回一个无效的平面方程系数


class Align_according_to_LicensePlate_Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(float)

    def __init__(self, WORK_DIR, plane, best_image_name, best_points):
        super(Align_according_to_LicensePlate_Worker, self).__init__()
        self.WORK_DIR = WORK_DIR
        self.plane = plane
        self.best_image_name = best_image_name
        self.best_points = best_points

    def run(self):
        projector = DepthBackProjector(self.WORK_DIR, self.plane)  # 投影情况下
        projector.load_data(self.best_image_name)
        real_distances, error = projector.compute_distance(self.best_points[0], self.best_points[1])  # 计算车牌两个点之间的距离
        self.finished.emit(real_distances)

# class Choose_Point_Worker(QtCore.QThread):
#     finished = QtCore.pyqtSignal(list)
#
#     def __init__(self, images, WORK_DIR):
#         super(Choose_Point_Worker, self).__init__()
#         self.images = images
#         self.WORK_DIR = WORK_DIR
#
#     def run(self):
#         self.finished.emit(self.images)
#         for file_path in self.images:
#             shutil.copy(file_path, self.WORK_DIR)
