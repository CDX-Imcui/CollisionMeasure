import numpy as np


def compute_distance(p1, p2):
    """
    计算三维空间中两点的欧氏距离

    参数:
        p1: 长度为3的list或numpy数组，表示第一个三维点
        p2: 长度为3的list或numpy数组，表示第二个三维点

    返回:
        两点之间的欧氏距离
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    distance = np.linalg.norm(p1 - p2)
    return distance


# 示例
point1 = [-0.8765 , 0.3433 , 3.1975]
point2 = [-0.8665 , 0.7551 , 1.0598]

d = compute_distance(point1, point2)
print(f"Distance: {d:.4f}")
