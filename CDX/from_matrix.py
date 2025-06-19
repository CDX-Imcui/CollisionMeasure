import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from read_write_dense import read_array
from read_write_model import read_images_binary, read_cameras_binary


class DepthBackProjector:
    def __init__(self, workspace):
        self.workspace = workspace
        self.images_bin = os.path.join(workspace, "sparse", "images.bin")
        self.cameras_bin = os.path.join(workspace, "sparse", "cameras.bin")
        self.images = read_images_binary(self.images_bin)
        self.cameras = read_cameras_binary(self.cameras_bin)

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

        self.img_data = next((img for img in self.images.values() if img.name.endswith(self.image_name)), None)
        if self.img_data is None:
            print(f"{self.image_name} 不在 {self.images_bin}")
            return -1  # 图像数据未找到
        # assert self.img_data is not None, f"{self.image_name} 不在 {self.images_bin}"
        self.cam = self.cameras[self.img_data.camera_id]
        self.fx, self.fy, self.cx, self.cy = self.cam.params[:4]
        self.depth = read_array(self.depth_file)
        assert self.depth.ndim == 2, "深度图维度错误"
        self.img = cv2.imread(self.image_file)
        assert self.img is not None, f"无法读取 {self.image_file}"
        return 0

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
            raise ValueError("像素超出图像范围")
        z = self.depth[v, u]
        if z <= 0:
            raise ValueError(f"无效深度: {z}")
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
        return X_world, error  # 返回三维坐标和方差

    def compute_distance(self, pt1, pt2):
        try:
            p1, error1 = self.pixel_to_world(*pt1)
            p2, error2 = self.pixel_to_world(*pt2)
            return np.linalg.norm(p1 - p2), np.sqrt(error1 ** 2 + error2 ** 2)  # 返回距离和误差
        except ValueError as e:
            print(str(e))
            return -1  # 返回-1表示计算失败

    def interactive_view(self):
        def on_mouse(event, u, v, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                try:
                    X_world, error = self.pixel_to_world(u, v)
                    print(f"点击像素: ({u}, {v}), 深度方差 = {error:.4f}")
                    print("三维坐标 (world):", np.round(X_world, 4))
                    cv2.circle(self.img, (u, v), 5, (0, 0, 255), -1)
                except ValueError as e:
                    print(str(e))

        cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Viewer", 1200, 800)
        cv2.setMouseCallback("Viewer", on_mouse)
        while True:
            cv2.imshow("Viewer", self.img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

# projector = DepthBackProjector("car")
# projector.load_data("image_001.jpg")
# projector.interactive_view()

# d = projector.compute_distance((100, 150), (200, 250))
# print("三维点距离：", d)

# # --- 配置参数 ---
# # IMAGE_NAME = "image_005.jpg"
# # IMAGE_NAME = "image_025.jpg"
# IMAGE_NAME = "image_001.jpg"
# WORKSPACE = "car"  # 稠密重建设备根目录
# DEPTH_FILE = os.path.join(WORKSPACE, "stereo", "depth_maps", IMAGE_NAME + ".geometric.bin")
#
# IMAGE_FILE = os.path.join(WORKSPACE,  "input", IMAGE_NAME)  # 与 COLMAP 深度对应图像路径
# CAMERAS_BIN = os.path.join(WORKSPACE,  "sparse", "cameras.bin")
# IMAGES_BIN = os.path.join(WORKSPACE,  "sparse", "images.bin")
#
# # --- 加载数据 ---
# images = read_images_binary(IMAGES_BIN)
# cameras = read_cameras_binary(CAMERAS_BIN)
#
# # 找到当前图像对应相机内外参
# img_data = None
# for img in images.values():
#     if img.name.endswith(IMAGE_NAME):
#         img_data = img
#         break
# assert img_data is not None, f">{IMAGE_NAME} 不在 {IMAGES_BIN}"
#
# cam = cameras[img_data.camera_id]
# fx, fy, cx, cy = cam.params[:4]
#
# # 读取图像和深度图（H×W）
# img = cv2.imread(IMAGE_FILE)
# assert img is not None, f"无法读取 {IMAGE_FILE}"
# depth = read_array(DEPTH_FILE)
# assert depth.ndim == 2, "深度图维度错误"
#
# # # --- 标记所有有效深度像素 ---
# # # img_marked = img.copy()
# # ys, xs = np.where(depth > 0)
# # for u, v in zip(xs, ys):
# #     # cv2.circle(img, (u, v), 1, (0, 0, 255), -1)  #中心像素及其周围一圈像素
# #     img[v, u] = (0, 0, 255)  # 直接赋值，红色小圆点
#
#
# # --- 鼠标点击回调 ---
# def on_mouse(event, u, v, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
#             print("点击超出图像范围")
#             return
#         z = depth[v, u]
#         if z <= 0:
#             print(f"深度值无效: {z}")
#             return
#         x_cam = (u - cx) * z / fx
#         y_cam = (v - cy) * z / fy
#         X_cam = np.array([x_cam, y_cam, z])
#         q = img_data.qvec
#         t = np.array(img_data.tvec)
#         R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
#         X_world = R_mat.T.dot(X_cam - t)
#
#         # 在图像上标记点击位置
#         cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)  # 红色表示点击位置
#
#         print(f"点击像素: ({u}, {v}), 深度 = {z:.3f}")
#         print("三维坐标 (world): ", np.round(X_world, 4))
#
# # --- 显示窗口并绑定 ---
# cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Viewer", 1200, 800)
#
# cv2.setMouseCallback("Viewer", on_mouse)
#
# while True:
#     cv2.imshow("Viewer", img)
#     if cv2.waitKey(1) == 27:
#         break
# cv2.destroyAllWindows()

