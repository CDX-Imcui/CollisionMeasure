"""
高斯点云VTK交互式渲染器
使用VTK实现对高斯点云的3D交互式渲染
"""

import os
import sys
import numpy as np
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from scene.gaussian_model import GaussianModel

class GaussianPointCloudStyle(vtk.vtkInteractorStyleTrackballCamera):
    """自定义VTK交互样式"""

    def __init__(self, callback=None):
        super().__init__()
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.AddObserver("KeyReleaseEvent", self.on_key_release)
        self.AddObserver("LeftButtonPressEvent", self.on_left_click)
        self.AddObserver("RightButtonPressEvent", self.on_right_click)
        self.space_pressed = False
        self.callback = callback

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "space":
            self.space_pressed = True
            print("空格键按下")
        elif key == "r" or key == "R":
            # 重置相机位置
            if self.parent:
                self.parent.reset_camera()
        elif key == "plus" or key == "equal":
            # 增加点大小
            if self.parent:
                self.parent.increase_point_size()
        elif key == "minus":
            # 减小点大小
            if self.parent:
                self.parent.decrease_point_size()

    def on_key_release(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "space":
            self.space_pressed = False
            print("空格键释放")

    def on_left_click(self, obj, event):
        if self.space_pressed:
            # 空格+左键点击，将点云中的点设为中心
            position = self.GetInteractor().GetEventPosition()
            self.callback(position)
        else:
            # 保留默认的旋转行为
            self.OnLeftButtonDown()

    def on_right_click(self, obj, event):
        # 保留默认的右键行为
        self.OnRightButtonDown()

class VTKGaussianViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("高斯点云VTK查看器")
        self.setMinimumSize(800, 600)

        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)

        # 创建VTK窗口
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.main_layout.addWidget(self.vtk_widget, 1)

        # 创建控制面板
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_frame.setStyleSheet("background-color: #f0f0f0;")
        control_layout = QHBoxLayout(control_frame)

        # 点大小控制
        point_size_layout = QVBoxLayout()
        point_size_layout.addWidget(QLabel("点大小:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(20)
        self.point_size_slider.setValue(1)
        self.point_size_slider.setTickPosition(QSlider.TicksBelow)
        self.point_size_slider.valueChanged.connect(self.on_point_size_changed)
        point_size_layout.addWidget(self.point_size_slider)
        control_layout.addLayout(point_size_layout)

        # 不透明度控制
        opacity_layout = QVBoxLayout()
        opacity_layout.addWidget(QLabel("不透明度:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(10)
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        control_layout.addLayout(opacity_layout)

        # 重置相机按钮
        self.reset_camera_button = QPushButton("重置视图")
        self.reset_camera_button.clicked.connect(self.reset_camera)
        control_layout.addWidget(self.reset_camera_button)

        # 将控制面板添加到主布局
        self.main_layout.addWidget(control_frame, 0)

        # 设置VTK渲染
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.3)  # 深蓝色背景
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # 创建交互器并设置交互模式
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.style = GaussianPointCloudStyle(callback=self.align_center_click)
        self.interactor.SetInteractorStyle(self.style)

        # 初始化VTK窗口
        self.vtk_widget.Initialize()

        # 属性初始化
        self.point_cloud_actor = None
        self.gaussian_model = None
        self.point_size = 3.0
        self.opacity = 1.0
        self.point_data = vtk.vtkPolyData()

    def set_gaussian_model(self, model: GaussianModel):
        """设置高斯点云模型并渲染"""
        if model is None:
            return

        self.gaussian_model = model

        # 从GaussianModel中提取点云数据
        if self.point_cloud_actor:
            self.renderer.RemoveActor(self.point_cloud_actor)

        # 创建VTK点云数据
        self.point_data = self.create_vtk_point_cloud(model)

        # 创建Mapper
        if self.point_data.GetNumberOfPoints() < 100000:
            # 对于较小的点云使用高斯渲染
            mapper = vtk.vtkPointGaussianMapper()
            mapper.SetInputData(self.point_data)
            mapper.SetScaleFactor(0.0)  # 让渲染器自动计算比例
        else:
            # 对于大型点云使用普通渲染
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.point_data)

        # 配置颜色映射
        scalars = self.point_data.GetPointData().GetScalars()
        if scalars and scalars.GetNumberOfComponents() >= 3:
            mapper.SetColorModeToDirectScalars()
        else:
            mapper.SetColorModeToDefault()

        mapper.SetScalarVisibility(True)
        mapper.SetScalarModeToUsePointData()

        # 创建Actor
        self.point_cloud_actor = vtk.vtkActor()
        self.point_cloud_actor.SetMapper(mapper)
        self.point_cloud_actor.GetProperty().SetPointSize(self.point_size)
        self.point_cloud_actor.GetProperty().SetOpacity(self.opacity)

        # 添加到渲染器
        self.renderer.AddActor(self.point_cloud_actor)
        self.reset_camera()

    def create_vtk_point_cloud(self, model: GaussianModel):
        """从GaussianModel创建VTK点云数据"""
        # 从模型中提取数据
        points = model.get_xyz.detach().cpu().numpy()
        colors = model.get_features_dc.detach().cpu().squeeze().numpy()

        # 确保颜色在正确范围内
        colors = np.clip(colors, 0, 1)
        colors_uint8 = (colors * 255).astype(np.uint8)

        # 创建VTK点
        vtk_points = vtk.vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(point)

        # 创建VTK颜色
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")

        for color in colors_uint8:
            vtk_colors.InsertNextTuple3(color[0], color[1], color[2])

        # 创建VTK多边形数据
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # 将点添加为顶点单元
        verts = vtk.vtkCellArray()
        for i in range(len(points)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        polydata.SetVerts(verts)

        # 添加颜色
        polydata.GetPointData().SetScalars(vtk_colors)

        return polydata

    def align_center_click(self, click_pos):
        """空格加单击 将点击的点设置为视图中心"""
        print("空格加单击触发")
        picker = vtk.vtkPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        point_id = picker.GetPointId()

        if point_id != -1:
            pos = self.point_data.GetPoint(point_id)
            print(f"双击点坐标: {pos}")
            camera = self.renderer.GetActiveCamera()

            # 获取当前相机位置和焦点
            old_pos = np.array(camera.GetPosition())
            old_focal = np.array(camera.GetFocalPoint())

            # 将新的焦点设置为点击点
            new_focal = np.array(pos)
            camera.SetFocalPoint(*new_focal)

            # 保持相机相对于焦点的方向和距离不变，平移位置
            direction = old_pos - old_focal
            new_pos = new_focal + direction
            camera.SetPosition(*new_pos)

            # 重新渲染
            self.vtk_widget.GetRenderWindow().Render()
        else:
            print("未拾取到点")

    def reset_camera(self):
        """重置相机视角"""
        if self.renderer:
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

    def increase_point_size(self):
        """增加点大小"""
        if self.point_cloud_actor:
            self.point_size += 0.5
            self.point_cloud_actor.GetProperty().SetPointSize(self.point_size)
            self.point_size_slider.setValue(min(20, int(self.point_size)))
            self.vtk_widget.GetRenderWindow().Render()

    def decrease_point_size(self):
        """减小点大小"""
        if self.point_cloud_actor:
            self.point_size = max(1.0, self.point_size - 0.5)
            self.point_cloud_actor.GetProperty().SetPointSize(self.point_size)
            self.point_size_slider.setValue(max(1, int(self.point_size)))
            self.vtk_widget.GetRenderWindow().Render()

    def on_point_size_changed(self, value):
        """点大小滑块改变事件"""
        if self.point_cloud_actor:
            self.point_size = float(value)
            self.point_cloud_actor.GetProperty().SetPointSize(self.point_size)
            self.vtk_widget.GetRenderWindow().Render()

    def on_opacity_changed(self, value):
        """不透明度滑块改变事件"""
        if self.point_cloud_actor:
            self.opacity = value / 100.0
            self.point_cloud_actor.GetProperty().SetOpacity(self.opacity)
            self.vtk_widget.GetRenderWindow().Render()

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 释放VTK资源
        self.vtk_widget.GetRenderWindow().Finalize()
        self.vtk_widget.close()
        super().closeEvent(event)

def show_gaussian_model_vtk(gaussian_model):
    """显示高斯模型的VTK查看器"""
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = VTKGaussianViewer()
    viewer.set_gaussian_model(gaussian_model)
    viewer.show()
    return viewer, app

# 如果直接运行该文件
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    viewer = VTKGaussianViewer()

    # 如果有命令行参数，尝试加载点云文件
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
        if os.path.exists(ply_file):
            # 这里需要实现加载PLY文件的功能
            print(f"加载点云文件: {ply_file}")

    viewer.show()
    sys.exit(app.exec_())
