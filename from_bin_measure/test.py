import cv2
import numpy as np
import json,vtk
from read_write_model import read_points3D_binary, read_images_binary


# 在import部分后添加自定义交互样式类
class MyStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, point_cloud=None):
        super().__init__()
        self.point_cloud = point_cloud
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.AddObserver("KeyReleaseEvent", self.on_key_release)
        self.AddObserver("LeftButtonPressEvent", self.on_left_click)
        self.AddObserver("RightButtonPressEvent", self.on_right_click)
        self.space_pressed = False

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "space":
            self.space_pressed = True

    def on_key_release(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "space":
            self.space_pressed = False

    def on_left_click(self, obj, event):
        if self.space_pressed:
            Position = self.GetInteractor().GetEventPosition()
            self.align_center_click(Position)
        else:
            # 保留默认旋转行为
            self.OnLeftButtonDown()

    def on_right_click(self, obj, event):
        Position = self.GetInteractor().GetEventPosition()
        self.pick_point(Position)
        # 保留默认行为
        self.OnRightButtonDown()

    def align_center_click(self, click_pos):
        """空格加单击 将点击的点设置为视图中心"""
        if self.point_cloud is None:
            return

        picker = vtk.vtkPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetCurrentRenderer())
        point_id = picker.GetPointId()

        if point_id != -1:
            pos = self.point_cloud.GetPoint(point_id)
            print(f"点击点坐标: {pos}")
            camera = self.GetCurrentRenderer().GetActiveCamera()

            # 获取当前相机位置和焦点
            old_pos = np.array(camera.GetPosition())
            old_focal = np.array(camera.GetFocalPoint())

            # 将新的焦点设置为点击点
            new_focal = np.array(pos)
            camera.SetFocalPoint(*new_focal)

            # 保持相机相对于焦点的方向和距离不变
            direction = old_pos - old_focal
            new_pos = new_focal + direction
            camera.SetPosition(*new_pos)

            # 重新渲染
            self.GetInteractor().GetRenderWindow().Render()
        else:
            print("未拾取到点")

    def pick_point(self, click_pos):
        """右键点击拾取点"""
        if self.point_cloud is None:
            return

        picker = vtk.vtkPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetCurrentRenderer())
        point_id = picker.GetPointId()

        if point_id != -1:
            pos = self.point_cloud.GetPoint(point_id)
            print(f"拾取点ID: {point_id}, 坐标: {pos}")
            # 这里可以添加标记点或其他操作
# 配置路径
IMAGE_PATH = "image_001.jpg"
IMAGES_BIN_PATH = "images.bin"
POINTS3D_BIN_PATH = "points3D.bin"
OUTPUT_JSON = "selected_points.json"

# 读取 COLMAP 二进制
images = read_images_binary(IMAGES_BIN_PATH)
points3D = read_points3D_binary(POINTS3D_BIN_PATH)

# 找到当前图像的 metadata
image_data = next((img for img in images.values()
                   if img.name == IMAGE_PATH or img.name.endswith(IMAGE_PATH)), None)
if image_data is None:
    raise RuntimeError(f"图像 {IMAGE_PATH} 未在 {IMAGES_BIN_PATH} 中找到")

xy = image_data.xys
pt3D_ids = image_data.point3D_ids
valid_inds = [i for i, pid in enumerate(pt3D_ids) if pid != -1 and pid in points3D]
valid_xy = xy[valid_inds]
valid_pid = [pt3D_ids[i] for i in valid_inds]

# 准备 OpenCV 窗口
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError(f"无法加载图像 {IMAGE_PATH}")
display_img = img.copy()
selected = []


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # 找到最近的有效特征点
    click = np.array([x, y])
    dists = np.linalg.norm(valid_xy - click, axis=1)
    idx = int(np.argmin(dists))
    p2d = valid_xy[idx].tolist()
    pid = int(valid_pid[idx])
    xyz = points3D[pid].xyz.tolist()

    # 画标记
    cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
    cv2.circle(display_img, (int(p2d[0]), int(p2d[1])), 5, (0, 255, 0), -1)
    cv2.line(display_img, (x, y), (int(p2d[0]), int(p2d[1])), (255, 0, 0), 1)
    print(f"点击: ({x},{y}), 最近特征: {p2d}, 3D 点 ID={pid}, 坐标={xyz}")

    selected.append({
        "click_xy": [x, y],
        "feat_xy": p2d,
        "pt3d_id": pid,
        "pt3d_xyz": xyz
    })


cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", on_mouse)

print("请在窗口中点击选点，按 ESC 结束并保存。")
while True:
    cv2.imshow("Image", display_img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

# 保存到 JSON
with open(OUTPUT_JSON, "w") as fp:
    json.dump(selected, fp, indent=2, ensure_ascii=False)
print(f"已保存 {len(selected)} 个点到 {OUTPUT_JSON}")

import json
import vtk
from PointCloud import getPointCloud, showPointCloud, mark_point, draw_line_between_points

# 配置路径
PLY_PATH = "input.ply"
SELECTED_JSON = "selected_points.json"

# 读取点云和已选点
data = json.load(open(SELECTED_JSON, "r", encoding="utf8"))
xyzs = [item["pt3d_xyz"] for item in data]

# VTK 初始化
point_cloud = getPointCloud(PLY_PATH)

# 1. 创建 RenderWindow / Renderer / Interactor
renWin = vtk.vtkRenderWindow()
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.05, 0.05, 0.1)  # 深蓝灰
renWin.AddRenderer(renderer)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
# 2. 创建并绑定你想要的交互风格
# style = vtk.vtkInteractorStyleTrackballCamera()
# # 这一句非常关键：告诉 style 它要控制哪个 renderer
# style.SetDefaultRenderer(renderer)
# iren.SetInteractorStyle(style)
# 替换原来的交互器样式设置
style = MyStyle(point_cloud)  # 传入点云对象
style.SetDefaultRenderer(renderer)
iren.SetInteractorStyle(style)

showPointCloud(point_cloud, vtkWidget=None, renderer=renderer)
# 标记所有点
for idx, xyz in enumerate(xyzs):
    mark_point(xyz, vtkWidget=None, renderer=renderer)

# 3. （可选）设置定时器或其他 Observer
# iren.AddObserver('TimerEvent', my_timer_cb)
# iren.CreateRepeatingTimer(30)

renderer.ResetCamera()
iren.Initialize()
renWin.Render()
iren.Start()
