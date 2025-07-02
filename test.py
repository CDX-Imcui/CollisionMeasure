# import os
# import cv2
# import numpy as np
#
# from CDX.from_matrix import DepthBackProjector
# from SemanticSegmentation import SemanticSegmentation
# from sklearn.neighbors import NearestNeighbors
# from sklearn.linear_model import RANSACRegressor
#
#
# def remove_outliers_statistical(point_array, k=10, std_ratio=2.0):
#     """
#     使用统计滤波去除离群点
#     :param point_array: numpy array, shape (N, 3)
#     :param k: 邻居数量
#     :param std_ratio: 距离标准差倍数阈值
#     :return: 去噪后的点集
#     """
#     if not isinstance(point_array, np.ndarray):
#         point_array = np.array(point_array)
#
#     neighbors = NearestNeighbors(n_neighbors=k + 1).fit(point_array)
#     distances, _ = neighbors.kneighbors(point_array)
#     mean_dists = distances[:, 1:].mean(axis=1)  # 去掉自身
#
#     threshold = mean_dists.mean() + std_ratio * mean_dists.std()
#     mask = mean_dists < threshold
#     return point_array[mask]
#
#
# def fit_plane_ransac(points, residual_threshold=0.01):
#     """
#     对去噪后的三维点进行 RANSAC 平面拟合
#     ax + by + cz + d = 0
#     :param points: numpy array, shape (N, 3)
#     :return: normal vector (a, b, c), d, inlier mask
#     """
#     points = np.array(points)
#     X = points[:, :2]
#     y = points[:, 2]
#
#     model = RANSACRegressor(residual_threshold=residual_threshold)
#     model.fit(X, y)
#
#     a, b = model.estimator_.coef_
#     c = -1.0
#     d = model.estimator_.intercept_
#
#     # ax + by + cz + d = 0 → ax + by - z + d = 0 → normal = [a, b, -1]
#     normal = np.array([a, b, c])
#     norm = np.linalg.norm(normal)
#     normal /= norm
#     d /= norm
#
#     # 内点掩码
#     inlier_mask = model.inlier_mask_
#     return normal, d, inlier_mask
#
#
# # segmentation = SemanticSegmentation(os.path.join(os.getcwd(), 'WORK_DIR/input'))
# segmentation = SemanticSegmentation("image_00059.jpg")
# images = segmentation.run()
# # image = images[59]  # 获取最优图像
# image = images[0]  # 获取最优图像
# FLOOR_COLOR = [140, 140, 140]  # 注意顺序是 RGB
# ground_pixels = []
#
# blank_image = np.zeros_like(image)
# # 根据分割图像，20个像素为步长遍历获取地面的特征点集合
# for y in range(0, image.shape[0], 20):
#     for x in range(0, image.shape[1], 20):
#         pixel = image[y, x, :]  # 获取该点 RGB
#         if np.array_equal(pixel, FLOOR_COLOR):
#             ground_pixels.append((x, y))  # 记录图像坐标
#             cv2.circle(blank_image, (x, y), 1, (0, 255, 0), -1)  # 标记为绿色点
#
# cv2.imwrite("AAAAA.png", blank_image)
#
# # 遍历ground_pixels，得到三维坐标列表
# projector = DepthBackProjector(os.path.join(os.getcwd(), 'WORK_DIR'))
# projector.load_data('image_00059.jpg')
# _3Dcoordinates = []
# for pixel in ground_pixels:
#     l, r = projector.pixel_to_world(pixel[0], pixel[1])
#     if (isinstance(l, np.ndarray) and l.size == 0) or (isinstance(r, np.ndarray) and r.size == 0) or np.array_equal(l, -1) or np.array_equal(r, -1):  # 如果像素点超出范围或深度无效
#         continue
#     x, y, z = l
#     _3Dcoordinates.append([x, y, z])  # 计算每个点的三维坐标
#
# # 根据_3Dcoordinates地面特定点拟合地面方程
# cleaned_points =remove_outliers_statistical(_3Dcoordinates, k=10, std_ratio=2.0)
# normal, d, inliers = fit_plane_ransac(cleaned_points)
# a, b, c = normal
# print(f"RANSAC 拟合平面: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
# print(f"内点数量: {np.sum(inliers)}, 平均拟合误差: {np.mean(np.abs(cleaned_points[inliers] @ normal + d)):.4f}")
import os
import cv2
import numpy as np

from from_matrix import DepthBackProjector
from SemanticSegmentation import SemanticSegmentation
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor


def remove_outliers_statistical(point_array, k=10, std_ratio=2.0):
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


def fit_plane_ransac(points, residual_threshold=0.01):
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


# segmentation = SemanticSegmentation(os.path.join(os.getcwd(), 'WORK_DIR/input'))
segmentation = SemanticSegmentation("image_00059.jpg")
images = segmentation.run()
# image = images[59]  # 获取最优图像
image = images[0]  # 获取最优图像
FLOOR_COLOR = [140, 140, 140]  # 注意顺序是 RGB
ground_pixels = []

ori_image=cv2.imread("image_00059.jpg")
blank_image = np.zeros_like(image)
# 根据分割图像，20个像素为步长遍历获取地面的特征点集合
for y in range(0, image.shape[0], 20):
    for x in range(0, image.shape[1], 20):
        pixel = image[y, x, :]  # 获取该点 RGB
        if np.array_equal(pixel, FLOOR_COLOR):
            ground_pixels.append((x, y))  # 记录图像坐标
            cv2.circle(ori_image, (2*x, 2*y), 1, (0, 255, 0), -1)  # 标记为绿色点
            # cv2.drawMarker(ori_image, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10,
            #                thickness=2)  # 在原图上画绿色十字

cv2.imwrite("AAAAA.png", ori_image)

# 遍历ground_pixels，得到三维坐标列表
projector = DepthBackProjector(os.path.join(os.getcwd(), 'WORK_DIR'))
projector.load_data('image_00059.jpg')
_3Dcoordinates = []
for pixel in ground_pixels:
    l, r = projector.pixel_to_world(pixel[0], pixel[1])
    if (isinstance(l, np.ndarray) and l.size == 0) or (isinstance(r, np.ndarray) and r.size == 0) or np.array_equal(l, -1) or np.array_equal(r, -1):  # 如果像素点超出范围或深度无效
        continue
    x, y, z = l
    _3Dcoordinates.append([x, y, z])  # 计算每个点的三维坐标

# 根据_3Dcoordinates地面特定点拟合地面方程
cleaned_points =remove_outliers_statistical(_3Dcoordinates, k=10, std_ratio=2.0)
normal, d, inliers = fit_plane_ransac(cleaned_points)
a, b, c = normal
print(f"RANSAC 拟合平面: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
print(f"内点数量: {np.sum(inliers)}, 平均拟合误差: {np.mean(np.abs(cleaned_points[inliers] @ normal + d)):.4f}")
