import vtk

from plyfile import PlyData
import numpy as np
import torch


def load_splat_ply(ply_path, device="cuda"):
    """从 PLY 文件精确读取 Splat Map 属性，返回用于构造 GaussianModel 的 dict"""
    ply = PlyData.read(ply_path)
    # 顶点元素通常名为 'vertex'
    v = ply['vertex'].data  # numpy structured array

    # 读取位置
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)  # (N,3)

    # 读取 DC 颜色
    dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)  # (N,3)

    # 读取透明度
    alpha = v['opacity'].astype(np.float32)  # (N,)

    # 读取 scale, rot
    scale = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)  # (N,3)
    # rot = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'], v['rot_4']], axis=1)  # (N,5)
    rot = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1)  # (N,5)

    # 读取 SH 高频分量 rest: f_rest_0 ~ f_rest_44
    rest = np.stack([v[f'f_rest_{i}'] for i in range(45)], axis=1)  # (N,45)
    # 合并 DC + rest 按照前面管线需要的格式，例如 (N,3+(d+1)^2)
    # 假定 active_sh_degree=2, (2+1)^2=9, 3*9=27 channels。要根据你的模型调整：
    # 这里示例直接把 rest 传进去
    sh = rest.astype(np.float32)

    # 3D 协方差：若 PLY 提供，否则 None
    # 假设你没有直接提供协方差，而是用 scale+rot 算
    cov3D = None

    return {
        'xyz': torch.from_numpy(xyz).to(device),
        'dc': torch.from_numpy(dc).to(device),
        'alpha': torch.from_numpy(alpha).to(device),
        'scale': torch.from_numpy(scale).to(device),
        'rot': torch.from_numpy(rot).to(device),
        'sh': torch.from_numpy(sh).to(device),
        'cov3D': None,
        'max_sh_degree': 2,  # 根据 rest 长度或你的模型设置
    }


from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import sys
from PyQt5.QtWidgets import QApplication

# 1. 读取 Splat Map
splat_data = load_splat_ply("car.ply", device="cuda")

# 正确初始化 GaussianModel
pc = GaussianModel(sh_degree=splat_data['max_sh_degree'])
# 使用 nn.Parameter 包装张量以便正确追踪梯度
pc._xyz = nn.Parameter(splat_data['xyz'].requires_grad_(True))
pc._scaling = nn.Parameter(torch.log(splat_data['scale']).requires_grad_(True))  # 应用反激活函数
pc._rotation = nn.Parameter(splat_data['rot'].requires_grad_(True))
pc._features_dc = nn.Parameter(splat_data['dc'].unsqueeze(1).requires_grad_(True))  # 形状应为 [N, 1, 3]
pc._features_rest = nn.Parameter(splat_data['sh'].view(splat_data['sh'].shape[0], -1, 3).requires_grad_(True))
pc._opacity = nn.Parameter(pc.inverse_opacity_activation(splat_data['alpha'].unsqueeze(1)).requires_grad_(True))

# 设置必要的变量
pc.active_sh_degree = splat_data['max_sh_degree']  # 确保活动 SH 度数正确设置
pc.max_radii2D = torch.zeros((pc._xyz.shape[0]), device="cuda")  # 初始化最大半径


# 2. 构建相机（此处示例一个简单正交相机）
class DummyCam:
    image_height = 512
    image_width = 512
    FoVx = FoVy = 60 * np.pi / 180
    world_view_transform = torch.eye(4, device="cuda")
    full_proj_transform = torch.eye(4, device="cuda")
    camera_center = torch.zeros(3, device="cuda")
    image_name = "dummy"


cam = DummyCam()


# 3. 配置管线
class Pipe:
    debug = False
    antialiasing = False
    compute_cov3D_python = False
    convert_SHs_python = False


pipe = Pipe()
bg_color = torch.zeros(3, device="cuda")

# 4. 调用渲染
from Splatting_Rendering_interface import render  # render 函数

out = render(cam, pc, pipe, bg_color)

# 5. 可视化 - 提供多种可视化选项
if __name__ == "__main__":
    viz_option = "vtk"

    # 使用VTK交互式查看器
    app_vtk = QApplication.instance() or QApplication(sys.argv)
    from vtk_gaussian_viewer import show_gaussian_model_vtk

    viewer_vtk, _ = show_gaussian_model_vtk(pc)

    sys.exit(app_vtk.exec_())
