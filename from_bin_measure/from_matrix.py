import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from read_write_dense import read_array
from read_write_model import read_images_binary, read_cameras_binary

# --- 配置参数 ---
# IMAGE_NAME = "image_005.jpg"
# IMAGE_NAME = "image_025.jpg"
IMAGE_NAME = "image_00078.jpg"
WORKSPACE = "car1"  # 稠密重建设备根目录
DEPTH_FILE = os.path.join(WORKSPACE, "stereo", "depth_maps", IMAGE_NAME + ".geometric.bin")

IMAGE_FILE = os.path.join(WORKSPACE,  "input", IMAGE_NAME)  # 与 COLMAP 深度对应图像路径
CAMERAS_BIN = os.path.join(WORKSPACE,  "sparse", "cameras.bin")
IMAGES_BIN = os.path.join(WORKSPACE,  "sparse", "images.bin")

# --- 加载数据 ---
images = read_images_binary(IMAGES_BIN)
cameras = read_cameras_binary(CAMERAS_BIN)

# 找到当前图像对应相机内外参
img_data = None
for img in images.values():
    if img.name.endswith(IMAGE_NAME):
        img_data = img
        break
assert img_data is not None, f">{IMAGE_NAME} 不在 {IMAGES_BIN}"

cam = cameras[img_data.camera_id]
fx, fy, cx, cy = cam.params[:4]

# 读取图像和深度图（H×W）
img = cv2.imread(IMAGE_FILE)
assert img is not None, f"无法读取 {IMAGE_FILE}"
depth = read_array(DEPTH_FILE)
assert depth.ndim == 2, "深度图维度错误"

# # --- 标记所有有效深度像素 ---
# # img_marked = img.copy()
# ys, xs = np.where(depth > 0)
# for u, v in zip(xs, ys):
#     # cv2.circle(img, (u, v), 1, (0, 0, 255), -1)  #中心像素及其周围一圈像素
#     img[v, u] = (0, 0, 255)  # 直接赋值，红色小圆点


# --- 鼠标点击回调 ---
def on_mouse(event, u, v, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            print("点击超出图像范围")
            return
        z = depth[v, u]
        if z <= 0:
            print(f"深度值无效: {z}")
            return
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        X_cam = np.array([x_cam, y_cam, z])
        q = img_data.qvec
        t = np.array(img_data.tvec)
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        X_world = R_mat.T.dot(X_cam - t)

        # 在图像上标记点击位置
        cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)  # 红色表示点击位置

        print(f"点击像素: ({u}, {v}), 深度 = {z:.3f}")
        print("三维坐标 (world): ", np.round(X_world, 4))

# --- 显示窗口并绑定 ---
cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Viewer", 1200, 800)

cv2.setMouseCallback("Viewer", on_mouse)

while True:
    cv2.imshow("Viewer", img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
