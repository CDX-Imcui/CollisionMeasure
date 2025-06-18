import cv2
import numpy as np
from read_write_model import read_points3D_binary, read_images_binary

# 配置文件路径
image_path = "image_001.jpg"
images_bin_path = "images.bin"
points3D_bin_path = "points3D.bin"

# 读取 COLMAP 数据
images = read_images_binary(images_bin_path)
points3D = read_points3D_binary(points3D_bin_path)

# 找到目标图像对应的数据
image_data = None
for img in images.values():
    if img.name == image_path or img.name.endswith(image_path):
        image_data = img
        break

if image_data is None:
    raise RuntimeError(f"图像 {image_path} 不在 {images_bin_path} 中")

# 获取2D特征点及其对应的3D点id
xy = image_data.xys
pt3D_ids = image_data.point3D_ids
# 创建只包含有效3D点的索引
valid_indices = [i for i, pt_id in enumerate(pt3D_ids) if pt_id != -1 and pt_id in points3D]
valid_xy = xy[valid_indices]
valid_pt3D_ids = pt3D_ids[valid_indices]

# 加载图像
img = cv2.imread(image_path)
if img is None:
    raise RuntimeError(f"无法加载图像 {image_path}")
# 创建全局变量保存显示图像
display_img = img.copy()
# 跟踪点击的点
points_clicked = []
nearest_3d_points = []

# 定义鼠标点击事件回调
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global display_img, points_clicked, nearest_3d_points
        click = np.array([x, y])

        # 计算所有有效特征点到点击位置的距离
        dists = np.linalg.norm(valid_xy - click, axis=1)  # 使用valid_xy而不是xy
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]
        nearest_2D = valid_xy[nearest_idx]
        nearest_3D_id = valid_pt3D_ids[nearest_idx]

        # 在图像上标记点击位置和最近特征点
        cv2.circle(display_img, (int(x), int(y)), 5, (0, 0, 255), -1)  # 红色表示点击位置
        cv2.circle(display_img, (int(nearest_2D[0]), int(nearest_2D[1])), 5, (0, 255, 0), -1)  # 绿色表示最近特征点
        cv2.line(display_img, (int(x), int(y)), (int(nearest_2D[0]), int(nearest_2D[1])), (255, 0, 0), 1)

        print(f"\n点击位置: ({x}, {y})")

        # 获取对应的3D点信息
        pt3D = points3D[nearest_3D_id].xyz
        print(f"→ 最近的二维特征点: ({nearest_2D[0]:.2f}, {nearest_2D[1]:.2f}), 距离: {nearest_dist:.2f}像素")
        print(f"→ 对应的三维坐标: {pt3D}")

        # 保存最近特征点
        points_clicked.append(nearest_2D)
        nearest_3d_points.append(pt3D)

        # 如果已点击两次，计算两点之间的距离
        if len(points_clicked) == 2:
            # 计算2D距离
            dist_2d = np.linalg.norm(points_clicked[1] - points_clicked[0])
            # 计算3D距离
            dist_3d = np.linalg.norm(nearest_3d_points[1] - nearest_3d_points[0])

            # 在两点之间画线
            cv2.line(display_img,
                    (int(points_clicked[0][0]), int(points_clicked[0][1])),
                    (int(points_clicked[1][0]), int(points_clicked[1][1])),
                    (255, 255, 0), 2)

            # 计算线的中点，用于显示距离文本
            mid_point = ((points_clicked[0] + points_clicked[1]) / 2).astype(int)

            # 在图像上显示距离信息
            cv2.putText(display_img,
                       f"2D: {dist_2d:.2f}px",
                       (mid_point[0]+10, mid_point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_img,
                       f"3D: {dist_3d:.2f}",
                       (mid_point[0]+10, mid_point[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            print(f"\n两点之间的2D距离: {dist_2d:.2f}像素")
            print(f"两点之间的3D距离: {dist_3d:.2f}")

            # 重置点击计数，可以重新开始测量
            points_clicked = []
            nearest_3d_points = []

# 在显示界面之前，先在图像上显示所有有效的特征点
print(f"正在标记所有有效的特征点坐标，共 {len(valid_xy)} 个...")

# 直接在原始图像上标记所有特征点，而不是创建副本
for i, point in enumerate(valid_xy):
    # 绘制点
    # cv2.circle(display_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    # 添加一个外圈
    cv2.circle(display_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), 1)  # 红色外圈
    # 在点周围添加十字标记，增强可见性
    cv2.line(display_img, (int(point[0])-5, int(point[1])), (int(point[0])+5, int(point[1])), (255, 255, 0), 2)
    cv2.line(display_img, (int(point[0]), int(point[1])-5), (int(point[0]), int(point[1])+5), (255, 255, 0), 2)

# 创建窗口并绑定回调
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1200, 800)
cv2.setMouseCallback("Image", mouse_callback)

# 显示图像并等待交互
while True:
    cv2.imshow("Image", display_img)
    key = cv2.waitKey(1)
    if key == 27:  # 按下 ESC 键退出
        break

cv2.destroyAllWindows()
