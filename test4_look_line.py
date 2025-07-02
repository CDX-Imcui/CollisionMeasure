import numpy as np
import matplotlib.pyplot as plt

# 你的二维点（X, Y）
points = np.array([
    (944, 321),
    (969, 337),
    (988, 348),
    (1009, 361),
    (1038, 377),
    (1063, 394),
    (1109, 417),
    (1134, 429),
    (1169, 447),
    (1197, 466)
])

# 拟合直线 y = ax + b
x = points[:, 0]
y = points[:, 1]
a, b = np.polyfit(x, y, deg=1)
print(f"拟合直线方程: y = {a:.4f}x + {b:.4f}")

# 计算所有点到直线的垂直距离
def point_to_line_dist(x0, y0, a, b):
    return abs(a * x0 - y0 + b) / np.sqrt(a**2 + 1)

distances = [point_to_line_dist(px, py, a, b) for px, py in zip(x, y)]
for i, d in enumerate(distances):
    print(f"点 {i} {points[i]} 到拟合直线的距离: {d:.2f}")

# 判断是否“近似共线”
threshold = 2.0  # 可调
if all(d < threshold for d in distances):
    print("\n✅ 所有点近似在一条直线上")
else:
    print("\n❌ 点存在明显偏离，未共线")

# 可视化点和拟合直线
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='原始点')
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit + b
plt.plot(x_fit, y_fit, color='blue', label='拟合直线')

for i, (px, py) in enumerate(points):
    plt.text(px, py, str(i), fontsize=9, color='black')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("二维点与拟合直线")
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()  # 可视化图像坐标系Y轴朝下
plt.show()
