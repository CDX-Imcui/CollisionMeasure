import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from read_write_dense import read_array
from read_write_model import read_images_binary, read_cameras_binary
import open3d as o3d


class DepthBackProjector:
    def __init__(self, workspace, plane=None, to_Project=None):
        self.workspace = workspace
        self.images_bin = os.path.join(workspace, "sparse", "images.bin")
        self.cameras_bin = os.path.join(workspace, "sparse", "cameras.bin")
        self.images = read_images_binary(self.images_bin)
        self.cameras = read_cameras_binary(self.cameras_bin)
        self.PointCoordinate = None  # 用于存储三维坐标
        self.plane = None
        if plane is not None:
            self.plane = plane
        self.to_Project = None
        if to_Project is not None:
            self.to_Project = to_Project
        self.image_file = None
        self.image_file = None
        self.point_cloud = os.path.join(self.workspace, 'point_cloud.ply')

    def load_data(self, image_name, workspace=None):
        if workspace is not None:
            self.workspace = workspace
            self.images_bin = os.path.join(workspace, "sparse", "images.bin")
            self.cameras_bin = os.path.join(workspace, "sparse", "cameras.bin")
            self.images = read_images_binary(self.images_bin)
            self.cameras = read_cameras_binary(self.cameras_bin)

        self.image_name = image_name
        self.image_file = os.path.join(self.workspace, "input", image_name)
        self.depth_file = os.path.join(self.workspace, "stereo", "depth_maps", image_name + ".geometric.bin")
        if not os.path.exists(self.depth_file):
            print(f"深度文件 {self.depth_file} 不存在,跳过加载")
            return False
        self.img_data = next((img for img in self.images.values() if img.name.endswith(self.image_name)), None)
        if self.img_data is None:
            print(f"{self.image_name} 不在 {self.images_bin},跳过加载")
            return False  # 图像数据未找到
        # assert self.img_data is not None, f"{self.image_name} 不在 {self.images_bin}"
        self.cam = self.cameras[self.img_data.camera_id]
        self.fx, self.fy, self.cx, self.cy = self.cam.params[:4]
        self.depth = read_array(self.depth_file)
        if self.depth.ndim != 2:
            print("深度图维度错误,跳过加载")
            return False
        # assert self.depth.ndim == 2, "深度图维度错误"
        self.img = cv2.imread(self.image_file)
        if self.img is None:
            print(f"无法读取 {self.image_file},跳过加载")
            return False
        # assert self.img is not None, f"无法读取 {self.image_file}"
        print(f"{self.image_name} load_data完成")
        return True

    def test_image_existOrNot(self):
        """检查图像是否存在"""
        if not os.path.exists(self.image_file):
            print(f"图像文件 {self.image_file} 不存在")
            return False
        if not os.path.exists(self.depth_file):
            print(f"深度文件 {self.depth_file} 不存在")
            return False
        return True

    def world_to_pixel(self, X_world):
        if not isinstance(X_world, np.ndarray) or X_world.shape != (3,):
            print("形状为 (3,) 的 np.ndarray")
            return -1, -1,-1
        q = self.img_data.qvec  # 四元数表示相机姿态
        t = np.array(self.img_data.tvec)  # 相机平移向量

        # 注意这里不是转置，是原始旋转矩阵（从世界到相机）
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        X_cam = R_mat @ X_world + t  # 从世界坐标系转换到相机坐标系

        # 投影到图像平面
        x, y, z = X_cam
        if z <= 0:
            return -1, -1  ,-1# 点在相机背后或深度无效

        u = x * self.fx / z + self.cx
        v = y * self.fy / z + self.cy

        return int(round(u)), int(round(v)), z

    def build_pixel_to_point_map(self):
        assert hasattr(self, 'point_cloud'), "请先指定 self.point_cloud 文件路径"
        if not os.path.exists(self.point_cloud):
            raise FileNotFoundError(f"{self.point_cloud} 不存在")

        pcd = o3d.io.read_point_cloud(self.point_cloud)
        points = np.asarray(pcd.points)

        pixel_map = dict()  # 初始化：只保留每个像素最近的点

        for point in points:
            u, v, z = self.world_to_pixel(point)
            if u < 0 or v < 0 or u >= self.depth.shape[1] or v >= self.depth.shape[0]:
                continue
            key = (u, v)
            if key not in pixel_map or z < pixel_map[key]['depth']:
                pixel_map[key] = {'point': point, 'depth': z}
        return pixel_map  # dict[(u,v)] = {'point': [x,y,z], 'depth': z}

    def find_nearest_valid_pixel(self, pix2point, x, y, max_search_radius=20):
        min_dist = float('inf')
        nearest_key = None
        for (u, v) in pix2point:
            dx = x - u
            dy = y - v
            dist = dx * dx + dy * dy  # 避免开根，提高效率
            if dist < min_dist:
                min_dist = dist
                nearest_key = (u, v)

        if nearest_key and np.sqrt(min_dist) <= max_search_radius:
            return nearest_key, pix2point[nearest_key]
        else:
            return None, None

    def compute_patch_variance(self, u, v, win=1):
        """计算像素点邻域内深度方差"""
        h, w = self.depth.shape
        us = np.clip([u - win, u, u + win], 0, w - 1)
        vs = np.clip([v - win, v, v + win], 0, h - 1)
        patch = self.depth[vs[0]:vs[-1] + 1, us[0]:us[-1] + 1].astype(np.float32)
        # 仅统计有效深度
        vals = patch[patch > 0]
        if vals.size == 0:
            return 0.0
        return float(np.var(vals))

    def pixel_to_world(self, u, v):
        if not (0 <= v < self.depth.shape[0] and 0 <= u < self.depth.shape[1]):
            return -1, -1
            # raise ValueError("像素超出图像范围")

        z = self.depth[v, u]
        if z <= 0:
            return -1, -1
            # raise ValueError(f"无效深度: {z}")
        # 计算邻域深度方差
        error = self.compute_patch_variance(u, v)
        # 相机坐标系下反投影
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy
        X_cam = np.array([x_cam, y_cam, z])
        q = self.img_data.qvec
        t = np.array(self.img_data.tvec)
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        X_world = R_mat.T @ (X_cam - t)
        return X_world, error  # 返回三维坐标[x, y, z]和方差 （X_world 不是 list，而是一维NumPy数组）

    def compute_distance(self, pt1, pt2):  # 输入二维点
        try:
            p1, error1 = self.pixel_to_world(*pt1)
            p2, error2 = self.pixel_to_world(*pt2)
            # 检查点是否有效
            if isinstance(p1, int) or isinstance(p2, int):
                return -1, -1  # 无效点
            if self.to_Project is True:
                p1 = self.Projection(p1)
                p2 = self.Projection(p2)
            return np.linalg.norm(p1 - p2), np.sqrt(error1 ** 2 + error2 ** 2)  # 返回距离和误差
        except ValueError as e:
            print(str(e))
            return -1, -1  # 返回-1表示计算失败

    def Projection(self, point):  # [x, y, z]
        if self.to_Project is False:
            return point
        """将点投影到平面上"""
        # 检查点是否有效（不是 -1）
        if isinstance(point, int):
            return -1
        a, b, c, d = self.plane
        # 平面方程: ax + by + cz + d = 0
        x, y, z = point
        normal = np.array([a, b, c])
        point = np.array([x, y, z])
        # 计算从点到平面的有符号距离
        distance = (a * x + b * y + c * z + d) / (a ** 2 + b ** 2 + c ** 2)
        # 用距离乘法向量，得到从点垂直指向平面的矢量
        projection_point = point - distance * normal
        return projection_point

    def interactive_view(self, callback=None):
        # cv2.imread 默认以 BGR 格式读取图像，将图片从 BGR 转换为 RGB，以便 matplotlib 正确显示颜色
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # 创建 Matplotlib 图形窗口，并设置响应更快的后端
        plt.switch_backend('TkAgg')  # 使用更快的图形后端
        fig, ax = plt.subplots(figsize=(10, 8))  # 设置更大的图像尺寸
        ax.imshow(img_rgb)
        ax.axis('off')  # 关闭坐标轴

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                # 获取点击的像素坐标 (x, y)
                # event.xdata 和 event.ydata 是浮点数，需要转换为整数
                # 对于图像，y 是行，x 是列
                x = int(event.xdata + 0.5)  # 添加0.5并取整以提高精度
                y = int(event.ydata + 0.5)

                if 0 <= y < self.img.shape[0] and 0 <= x < self.img.shape[1]:  # 确保坐标在图片范围内
                    b, g, r = self.img[y, x]  # 获取该像素的颜色值
                    print(f"点击坐标 (X={x}, Y={y}) 的像素颜色 (RGB): ({r}, {g}, {b})")

                    # 在图像上显示十字标记
                    ax.plot(x, y, 'r+', markersize=10)
                    plt.draw()  # 立即更新绘图
                    self.PointCoordinate, error = self.pixel_to_world(x, y)
                    print(f"点击像素: ({x}, {y}), 附近一圈像素点的深度方差 = {error:.4f}")
                    print("三维坐标 (world):", np.round(self.PointCoordinate, 4))
                    # 执行回调函数，将坐标传递给外部
                    if callback is not None:
                        callback(self.PointCoordinate, error, (x, y))
                else:
                    print(f"点击位置 ({x}, {y}) 超出图片范围。")

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        def on_key(event):
            if event.key == 'q':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()  # 设置紧凑布局以最大化图像显示区域
        plt.show(block=True)  # 确保阻塞式显示
