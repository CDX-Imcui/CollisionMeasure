

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
from SemanticSegmentation import SemanticSegmentation
from read_write_dense import read_array
from read_write_model import read_images_binary, read_cameras_binary


class DepthBackProjector:
    def __init__(self, workspace, plane=None,to_Project=None):
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

    def pixel_to_z(self, u, v):
        if not (0 <= v < self.depth.shape[0] and 0 <= u < self.depth.shape[1]):
            return -1
            # raise ValueError("像素超出图像范围")
        z = self.depth[v, u]
        if z <= 0:
            return -1

        return z

    def pixel_to_world(self, u, v):
        if not (0 <= v < self.depth.shape[0] and 0 <= u < self.depth.shape[1]):
            return [0, 0, 0]
            # raise ValueError("像素超出图像范围")

        z = self.depth[v, u]
        if z <= 0:
            return [0, 0, 0]
            # raise ValueError(f"无效深度: {z}")
        # # 计算邻域深度方差
        # error = self.compute_patch_variance(u, v)
        # 相机坐标系下反投影
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy
        X_cam = np.array([x_cam, y_cam, z])
        q = self.img_data.qvec
        t = np.array(self.img_data.tvec)
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        X_world = R_mat.T @ (X_cam - t)

        X_cam = R_mat * X_world + t
        u = X_cam[0] / X_cam[2] * self.fx + self.cx
        v = X_cam[1] / X_cam[2] * self.fy + self.cy

        return X_world # 返回三维坐标[x, y, z]和方差 （X_world 不是 list，而是一维NumPy数组）
work=os.path.join(os.getcwd(),"output","1746024855000cd0721.mp4")
projector = DepthBackProjector(work,to_Project=False)
projector.load_data("image_00012.jpg")

# segmentation = SemanticSegmentation("image_00059.jpg")
# images = segmentation.run()
# # image = images[59]  # 获取最优图像
# image = images[0]  # 获取最优图像
points=[]
image =cv2.imread(os.path.join(os.getcwd(),"output","1746024855000cd0721.mp4","input", "image_00012.jpg"))
# blank_image = np.zeros_like(image)
depth_map = np.full((image.shape[0], image.shape[1]), -1.0, dtype=np.float32)
z=None
count=0
#
# depth_values = []
# # 收集所有z值
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         z = projector.pixel_to_z(x, y)
#         if z != -1:
#             depth_values.append(z)
#             # print(z)
# # 归一化
# min_z, max_z = 0, 32
# depth_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         z = projector.pixel_to_z(x, y)
#         if z == -1 or z>32 or z<0:
#             continue
#
#         norm_z = int(255 * (z - min_z) / (max_z - min_z))
#         # print(norm_z)
#         depth_image[y, x] = norm_z
# cv2.imshow("Viewer", depth_image)
# cv2.imwrite("1746024855000cd0721.jpg", depth_image)
#
#
for y in range(0, image.shape[0], 100):
    for x in range(0, image.shape[1], 100):
        # z=projector.pixel_to_z(x, y)  # 计算深度
        # if z==-1:
        #     continue

        z=projector.pixel_to_world(x,y)
        # if isinstance(z, tuple) and np.all(np.array(z) == -1):
        #     continue
        # z[0]=x
        # z[1]=y
        z[0]=z[0]*100
        z[1]=z[1]*100
        z[2]=z[2]*100
        points.append(z)
print("三维点",projector.pixel_to_world(int(530.35), int(396.00)))



# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
# 给所有点上红色
pcd.paint_uniform_color([1.0, 0.0, 0.0])  # RGB：红色
# 显示点云（可交互旋转）
o3d.visualization.draw_geometries([pcd],
                                  window_name='3D Red Points',
                                  width=800,
                                  height=600,
                                  point_show_normal=False)