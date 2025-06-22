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
point1 = [2.10400313 ,0.54021593, 0.95023576]
point2 = [-0.63171173 , 0.15827797 , 1.48994795]

d = compute_distance(point1, point2)
print(f"Distance: {d:.4f}")
