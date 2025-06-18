# -*- coding: utf-8 -*-
import numpy as np
import vtk
import os

# 导入交互样式类
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera


def getPointCloud(file_path):
    reader = None
    # 根据文件扩展名决定使用的读取器
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".ply":
        reader = vtk.vtkPLYReader()
    elif ext == ".obj":
        reader = vtk.vtkOBJReader()
    elif ext == ".stl":
        reader = vtk.vtkSTLReader()
    elif ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()  # 需要更新阅读器才能真正读取文件

    # point_cloud = reader.GetOutput()  # reader的输出是一个 vtkPolyData 点云对象
    point_cloud = vtk.vtkPolyData()
    point_cloud.ShallowCopy(reader.GetOutput())  # 改用ShallowCopy确保不出现非预期行为

    # --- 关键修改: 添加 vtkVertexCells 使点可拾取 ---
    # vtkPointPicker 需要点数据被组织成 vtkVertex 单元格才能正确拾取
    # verts = vtk.vtkCellArray()
    # for i in range(point_cloud.GetNumberOfPoints()):
    #     verts.InsertNextCell(1)  # 每个单元格包含一个点
    #     verts.InsertCellPoint(i)  # 将点的索引添加到单元格
    # point_cloud.SetVerts(verts)  # 将这些顶点单元格设置给 vtkPolyData
    # --- 关键修改结束 ---

    # 旋转点云180°正确朝向
    transform = vtk.vtkTransform()
    transform.RotateX(180)  # 绕X轴旋转180°
    # 给点云进行变换
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(point_cloud)
    transform_filter.Update()  # 应用变换
    point_cloud = transform_filter.GetOutput()
    print(f"点云包含颜色？: {point_cloud.GetPointData().GetScalars() is not None}")
    print(f"点云数据加载成功，点数: {point_cloud.GetNumberOfPoints()}")

    return point_cloud


# 显示点云
def showPointCloud(point_cloud, vtkWidget=None, renderer=None):
    # ！！！检查点云中的组成部分，默认不显示仅仅包含点的点云
    num_points = point_cloud.GetNumberOfPoints()
    num_lines = point_cloud.GetNumberOfLines()
    num_polys = point_cloud.GetNumberOfPolys()
    num_strips = point_cloud.GetNumberOfStrips()
    # 根据检查结果选择如何显示点云
    if num_points > 0 and num_lines == 0 and num_polys == 0 and num_strips == 0:
        # 只有点，没有线或面
        glyph_filter = vtk.vtkVertexGlyphFilter()
        glyph_filter.SetInputData(point_cloud)
        glyph_filter.Update()
        data_to_display = glyph_filter.GetOutput()
    else:
        # 包含线或面
        data_to_display = point_cloud

    # 调试输出颜色信息 - 修复这里的错误
    scalar_data = point_cloud.GetPointData().GetScalars()
    print("颜色数据情况:", "有" if scalar_data else "无")
    if scalar_data:
        print(f"颜色数组名称: {scalar_data.GetName()}, 组件数: {scalar_data.GetNumberOfComponents()}")

    # 创建映射器
    if num_points < 100000:
        # ！！！这个mapper很重要，点少就这样
        mapper = vtk.vtkPointGaussianMapper()
        mapper.SetInputData(data_to_display)
        mapper.SetScaleFactor(0.0)  # 设置为0让mapper自动计算大小，或者给一个合适的小值
        mapper.SetScalarVisibility(True)  # 告诉高斯渲染器使用标量数据
        # 高斯渲染器通常能很好地自动处理颜色，无需额外设置ColorMode
    else:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data_to_display)

    point_data = data_to_display.GetPointData()
    # 尝试通过名称直接查找常见的颜色数组
    if point_data.GetArray("Colors"):
        color_array = point_data.GetArray("Colors")
    elif point_data.GetArray("RGB"):
        color_array = point_data.GetArray("RGB")
    else:
        # 如果找不到特定名称的，就获取当前的活动标量
        color_array = point_data.GetScalars()
    # `SetColorModeToDefault()` 会尝试使用查找表，这经常导致颜色错误。
    # `SetColorModeToDirectScalars()` 告诉Mapper直接使用标量数组中的数据作为颜色，
    # 前提是数据类型是 unsigned char 并且有3个(RGB)或4个(RGBA)分量。
    # 这正是PLY等格式存储颜色的方式。
    if color_array and color_array.GetDataType() == vtk.VTK_UNSIGNED_CHAR and color_array.GetNumberOfComponents() >= 3:
        # 只有在确定颜色数据是标准的UCHAR类型时，才使用DirectScalars
        print("配置Mapper: 使用 DirectScalars 模式 (直接映射颜色)。")
        mapper.SetColorModeToDirectScalars()
    else:
        # 否则回退到默认模式，它会使用查找表
        print("配置Mapper: 使用 Default 模式 (通过查找表映射颜色)。")
        mapper.SetColorModeToDefault()
    mapper.SetScalarModeToUsePointData()  # 按点的属性着色
    mapper.ScalarVisibilityOn()  # 开启标量可见性

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(1)  # 使点更大
    # actor.GetProperty().SetLighting(False)

    # 创建渲染窗口和渲染器
    if renderer is None:
        renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)  # 将actor添加到场景中
    renderer.SetBackground(0.15, 0.15, 0.16)
    # renderer.SetBackground(255, 255, 255)

    if vtkWidget is None:
        # 创建渲染窗口和渲染器
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        # 调整相机
        renderer.ResetCamera()
        renderWindow.Render()  # 渲染和交互
        renderWindowInteractor.Start()
    else:

        vtkWidget.GetRenderWindow().AddRenderer(renderer)  # 将renderer渲染器添加到vtkWidget的渲染窗口中
        vtkWidget.Initialize()  # 在添加渲染器之后，立即对交互窗口进行初始化，确保所有的设置在交互开始之前都已配置好
        renderer.ResetCamera()  # 重置相机，确保相机根据当前的场景内容调整到最佳视角
        vtkWidget.GetRenderWindow().Render()  # 进行渲染，在开始交互之前能看到完整的场景
        vtkWidget.Start()  # 开始交互


def mark_point(pos, vtkWidget, renderer, auxiliary_actors=None):
    """在场景中标记一个点的位置"""
    # 创建一个小球体表示该点
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*pos)
    sphere.SetRadius(0.05)  # 设置球的半径，可根据点云尺寸调整
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)
    sphere.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0)  # 红色点
    actor.GetProperty().SetOpacity(0.6)

    renderer.AddActor(actor)
    vtkWidget.GetRenderWindow().Render()
    if auxiliary_actors is not None:
        auxiliary_actors.append(actor)  # 存储 actor 以便后续清除


def draw_line_between_points(point1, point2, pair_index, distances, real_distances, vtkWidget, renderer,
                             auxiliary_actors):
    """在3D场景中绘制两点之间的连线和距离标签"""
    # 画线
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(point1)
    lineSource.SetPoint2(point2)
    lineSource.Update()

    lineMapper = vtk.vtkPolyDataMapper()
    lineMapper.SetInputData(lineSource.GetOutput())

    lineActor = vtk.vtkActor()
    lineActor.SetMapper(lineMapper)
    lineActor.GetProperty().SetColor(0, 1, 0)  # 绿色
    lineActor.GetProperty().SetLineWidth(3)
    renderer.AddActor(lineActor)
    auxiliary_actors.append(lineActor)

    # 添加距离文本 - 使用3D文本
    mid_point = (point1 + point2) / 2.0
    text = vtk.vtkVectorText()

    # 如果有实际距离，显示实际距离，否则显示测量距离
    dist = distances[pair_index]
    real_dist = real_distances[pair_index]

    if real_dist is not None:
        text.SetText(f"测量: {dist:.2f}\n实际: {real_dist:.2f}")
    else:
        text.SetText(f"{dist:.2f}")

    textMapper = vtk.vtkPolyDataMapper()
    textMapper.SetInputConnection(text.GetOutputPort())

    textActor = vtk.vtkFollower()  # 使用vtkFollower代替vtkActor
    textActor.SetMapper(textMapper)
    textActor.SetScale(0.3, 0.3, 0.3)
    textActor.SetPosition(mid_point[0], mid_point[1], mid_point[2] + 1)  # 在中点上方显示文本
    # textActor.SetPosition(mid_point + np.array([0, 0, 0.01]))  # 在中点上方显示文本
    textActor.GetProperty().SetColor(1, 1, 0)  # 黄色
    textActor.GetProperty().SetOpacity(0.7)  # 半透明
    textActor.SetCamera(renderer.GetActiveCamera())  # 设置摄像机，确保文本始终朝向用户
    renderer.AddActor(textActor)
    auxiliary_actors.append(textActor)
    vtkWidget.GetRenderWindow().Render()
# def getPointCloud(file_path):
#     reader = vtk.vtkPLYReader()
#     reader.SetFileName(file_path)
#     reader.Update()
#
#     point_cloud = reader.GetOutput()
#     point_data = point_cloud.GetPointData()
#
#     # Print available arrays for debugging
#     print("\nAvailable point attributes:")
#     for i in range(point_data.GetNumberOfArrays()):
#         array = point_data.GetArray(i)
#         print(
#             f"  - {array.GetName()} (components: {array.GetNumberOfComponents()}, type: {array.GetDataTypeAsString()})")
#
#     # Check if this is a 3DGS format PLY
#     has_3dgs_attributes = all(point_data.HasArray(f"f_dc_{i}") for i in range(3))
#
#     if has_3dgs_attributes:
#         print("\nDetected 3D Gaussian Splatting format")
#
#         # Get base colors (f_dc)
#         f_dc_0 = point_data.GetArray("f_dc_0")
#         f_dc_1 = point_data.GetArray("f_dc_1")
#         f_dc_2 = point_data.GetArray("f_dc_2")
#
#         # Create RGB color array
#         colors = vtk.vtkUnsignedCharArray()
#         colors.SetNumberOfComponents(3)
#         colors.SetName("Colors")
#
#         # Convert f_dc values (0-1) to RGB (0-255)
#         num_points = point_cloud.GetNumberOfPoints()
#         for i in range(num_points):
#             r = int(max(0.0, min(1.0, f_dc_0.GetValue(i))) * 255)
#             g = int(max(0.0, min(1.0, f_dc_1.GetValue(i))) * 255)
#             b = int(max(0.0, min(1.0, f_dc_2.GetValue(i))) * 255)
#             colors.InsertNextTuple3(r, g, b)
#
#         # Set as point colors
#         point_cloud.GetPointData().SetScalars(colors)
#
#         # Store other attributes for potential use
#         if point_data.HasArray("opacity"):
#             opacity = point_data.GetArray("opacity")
#             opacity_array = vtk.vtkFloatArray()
#             opacity_array.SetName("Opacity")
#             opacity_array.SetNumberOfComponents(1)
#             for i in range(num_points):
#                 opacity_array.InsertNextValue(opacity.GetValue(i))
#             point_cloud.GetPointData().AddArray(opacity_array)
#
#         # Handle scale attributes
#         if all(point_data.HasArray(f"scale_{i}") for i in range(3)):
#             scales = vtk.vtkFloatArray()
#             scales.SetName("Scales")
#             scales.SetNumberOfComponents(3)
#             for i in range(num_points):
#                 scales.InsertNextTuple3(
#                     point_data.GetArray("scale_0").GetValue(i),
#                     point_data.GetArray("scale_1").GetValue(i),
#                     point_data.GetArray("scale_2").GetValue(i)
#                 )
#             point_cloud.GetPointData().AddArray(scales)
#
#         # Handle rotation attributes
#         if all(point_data.HasArray(f"rot_{i}") for i in range(4)):
#             rotations = vtk.vtkFloatArray()
#             rotations.SetName("Rotations")
#             rotations.SetNumberOfComponents(4)
#             for i in range(num_points):
#                 rotations.InsertNextTuple4(
#                     point_data.GetArray("rot_0").GetValue(i),
#                     point_data.GetArray("rot_1").GetValue(i),
#                     point_data.GetArray("rot_2").GetValue(i),
#                     point_data.GetArray("rot_3").GetValue(i)
#                 )
#             point_cloud.GetPointData().AddArray(rotations)
#
#     else:
#         # Handle regular PLY files
#         print("\nDetected regular PLY format")
#         if point_data.GetScalars() is None:
#             # If no colors, generate random ones
#             colors = vtk.vtkUnsignedCharArray()
#             colors.SetNumberOfComponents(3)
#             colors.SetName("Colors")
#             for _ in range(point_cloud.GetNumberOfPoints()):
#                 colors.InsertNextTuple3(
#                     random.randint(0, 255),
#                     random.randint(0, 255),
#                     random.randint(0, 255)
#                 )
#             point_cloud.GetPointData().SetScalars(colors)
#
#     # Apply 180° rotation if needed
#     transform = vtk.vtkTransform()
#     transform.RotateX(180)
#     transform_filter = vtk.vtkTransformFilter()
#     transform_filter.SetTransform(transform)
#     transform_filter.SetInputData(point_cloud)
#     transform_filter.Update()
#
#     return transform_filter.GetOutput()
#
#
#
# def showPointCloud(point_cloud, vtkWidget=None):
#     num_points = point_cloud.GetNumberOfPoints()
#     if num_points > 2000000:
#         mask = vtk.vtkMaskPoints()
#         mask.SetInputData(point_cloud)
#         mask.SetOnRatio(int(num_points / 1000000))
#         mask.RandomModeOn()
#         mask.Update()
#         point_cloud = mask.GetOutput()
#     print(f"点云包含点数: {point_cloud.GetNumberOfPoints()}")
#
#     # 如果没有颜色，随机生成颜色（RGB）
#     if not point_cloud.GetPointData().GetScalars():
#         colors = vtk.vtkUnsignedCharArray()
#         colors.SetNumberOfComponents(3)
#         colors.SetName("Colors")
#         import random
#         for _ in range(point_cloud.GetNumberOfPoints()):
#             colors.InsertNextTuple3(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         point_cloud.GetPointData().SetScalars(colors)
#
#     # mapper 使用 PointGaussianMapper（关键）
#     mapper = vtk.vtkPointGaussianMapper()
#     mapper.SetInputData(point_cloud)
#     mapper.EmissiveOff()
#     mapper.SetScaleFactor(0.01)  # 控制点大小
#     mapper.SetScalarModeToUsePointFieldData()
#     mapper.SelectColorArray("Colors")
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#
#     renderer = vtk.vtkRenderer()
#     renderer.AddActor(actor)
#     renderer.SetBackground(0.2, 0.3, 0.4)
#
#     render_window = vtkWidget.GetRenderWindow()
#     render_window.AddRenderer(renderer)
#
#     interactor = render_window.GetInteractor()
#     style = vtk.vtkInteractorStyleTrackballCamera()
#     interactor.SetInteractorStyle(style)
#
#     renderer.ResetCamera()
#     render_window.Render()
#     vtkWidget.Start()
