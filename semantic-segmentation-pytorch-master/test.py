# System libs
import os
import argparse
import time
from distutils.version import LooseVersion

import cv2
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate input and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '_seg.png')))


def resize_keep_aspect_ratio(img, max_size=640):
    """调整图像大小，保持横纵比，最长边不超过max_size"""
    height, width = img.shape[0], img.shape[1]

    # 如果图像已经小于等于最大尺寸，则不需要调整
    if width <= max_size and height <= max_size:
        return img

    # 计算缩放因子
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_img


def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    total_time = 0
    img_count = 0
    for batch_data in loader:
        # 处理数据
        batch_data = batch_data[0]

        # 获取原始图像并调整大小
        img_ori = batch_data['img_ori']

        # 计时开始
        start_time = time.time()

        img_ori_resized = resize_keep_aspect_ratio(img_ori, max_size=640)
        batch_data['img_ori'] = img_ori_resized

        # 设置分割尺寸
        segSize = (img_ori_resized.shape[0], img_ori_resized.shape[1])

        # 使用单一尺度
        img_data = batch_data['img_data'][0]

        with torch.no_grad():
            # 创建scores张量 - 与原始代码类似
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            # 准备输入数据
            feed_dict = batch_data.copy()
            feed_dict['img_data'] = img_data
            del feed_dict['img_ori']
            del feed_dict['info']
            feed_dict = async_copy_to(feed_dict, gpu)

            # 前向传播
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            scores = pred_tmp  # 单尺度，直接使用结果

            # 获取预测类别
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # 计时结束
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        img_count += 1
        print(f"图片 {img_count} 推理时间: {inference_time:.4f} 秒")

        # 可视化结果
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.update(1)

    # 计算平均时间
    if img_count > 0:
        avg_time = total_time / img_count
        print(f"平均每张图片推理时间: {avg_time:.4f} 秒")


# def test(segmentation_module, loader, gpu):
#     segmentation_module.eval()
#
#     pbar = tqdm(total=len(loader))
#     total_time = 0
#     img_count = 0
#     for batch_data in loader:
#         # process data
#         batch_data = batch_data[0]
#
#         segSize = (batch_data['img_ori'].shape[0],
#                    batch_data['img_ori'].shape[1])
#         img_resized_list = batch_data['img_data']
#
#         start_time = time.time()
#
#         with torch.no_grad():
#             scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
#             scores = async_copy_to(scores, gpu)
#
#             for img in img_resized_list:
#                 feed_dict = batch_data.copy()
#                 feed_dict['img_data'] = img
#                 del feed_dict['img_ori']
#                 del feed_dict['info']
#                 feed_dict = async_copy_to(feed_dict, gpu)
#
#                 # forward pass
#                 pred_tmp = segmentation_module(feed_dict, segSize=segSize)
#                 scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)
#
#             _, pred = torch.max(scores, dim=1)
#             pred = as_numpy(pred.squeeze(0).cpu())
#
#         # 记录结束时间
#         end_time = time.time()
#         inference_time = end_time - start_time
#         total_time += inference_time
#         img_count += 1
#         print(f"图片 {img_count} 推理时间: {inference_time:.4f} 秒")
#
#         # visualization
#         visualize_result(
#             (batch_data['img_ori'], batch_data['info']),
#             pred,
#             cfg
#         )
#
#         pbar.update(1)
#     # 计算平均推理时间
#     if img_count > 0:
#         avg_time = total_time / img_count
#         print(f"平均每张图片推理时间: {avg_time:.4f} 秒")


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=0,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, gpu)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)  # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    if os.path.isdir(args.imgs):
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
