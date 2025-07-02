# import torch
# import numpy as np
# import cv2
# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
#
# # 模型类型和权重路径
# model_type = "vit_t"
# sam_checkpoint = "./weight/mobile_sam.pt"
#
# # 使用 GPU 或 CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # 加载模型
# mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# mobile_sam.to(device=device)
# mobile_sam.eval()
#
# # 使用 OpenCV 读取图像并转换为 RGB
# image_path = "6.jpg"
# image_bgr = cv2.imread(image_path)
# if image_bgr is None:
#     raise FileNotFoundError(f"无法读取图像文件：{image_path}")
#
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#
# # 显式转换为 float32，避免 torch 无法推断 dtype 错误
# image_rgb = image_rgb.astype(np.float32) / 255.0
#
# # 构建自动掩码生成器
# mask_generator = SamAutomaticMaskGenerator(mobile_sam)
# masks = mask_generator.generate(image_rgb)
#
#
# # 输出结果
# print(f"共生成 {len(masks)} 个掩码")
# for i, mask in enumerate(masks):
#     print(f"Mask {i}: 面积 = {mask['area']}, 边界框 = {mask['bbox']}")
#
# # 可视化第一个掩码（可选）
# first_mask = masks[0]["segmentation"].astype(np.uint8) * 255
# cv2.imshow("First Mask", first_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
# for i, mask in enumerate(masks):
#     print(f"Mask {i}: area = {mask['area']}, bbox = {mask['bbox']}")
#
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = "./weight/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate('6.jpg')
