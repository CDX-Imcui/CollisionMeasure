# System libs
import os
import argparse
import time
from distutils.version import LooseVersion
import sys

# 确保 mit_semseg 可以被找到
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

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
# from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

class SemanticSegmentation:
    def __init__(self, imgs_path: str):
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg_path = os.path.join(self.module_dir, "config", "ade20k-resnet50dilated-ppm_deepsup.yaml")

        self.gpu = 0
        self.imgs_path = imgs_path

        # merge config
        cfg.merge_from_file(self.cfg_path)
        cfg.merge_from_list([])
        self.cfg = cfg
        self.cfg.MODEL.arch_encoder = self.cfg.MODEL.arch_encoder.lower()
        self.cfg.MODEL.arch_decoder = self.cfg.MODEL.arch_decoder.lower()

        self.cfg.MODEL.weights_encoder = os.path.join(self.module_dir,self.cfg.DIR, 'encoder_' + self.cfg.TEST.checkpoint)
        self.cfg.MODEL.weights_decoder = os.path.join(self.module_dir,self.cfg.DIR, 'decoder_' + self.cfg.TEST.checkpoint)

        assert os.path.exists(self.cfg.MODEL.weights_encoder), "encoder weights not found!"
        assert os.path.exists(self.cfg.MODEL.weights_decoder), "decoder weights not found!"

        # generate testing image list
        if os.path.isdir(self.imgs_path):
            imgs = find_recursive(self.imgs_path)# 只寻找jpg
        else:
            imgs = [self.imgs_path]
        assert len(imgs), "imgs should be a path to image (.jpg) or directory."
        self.cfg.list_test = [{'fpath_img': x} for x in imgs]

        if not os.path.isdir(self.cfg.TEST.result):
            os.makedirs(self.cfg.TEST.result)

        self.colors = loadmat(os.path.join(self.module_dir,'data/color150.mat'))['colors']
        self.names = {}
        with open(os.path.join(self.module_dir,'data/object150_info.csv')) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

    def _resize_keep_aspect_ratio(self, img, max_size=640):
        height, width = img.shape[0], img.shape[1]
        if width <= max_size and height <= max_size:
            return img
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _visualize_result(self, data, pred):
        (img, info) = data
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        print(f"Predictions in [{info}]:")
        for idx in np.argsort(counts)[::-1]:
            name = self.names.get(uniques[idx] + 1, "unknown")
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print(f"  {name}: {ratio:.2f}%")

        pred_color = colorEncode(pred, self.colors).astype(np.uint8)
        return pred_color
        # img_name = os.path.basename(info)
        # pred_path = os.path.join(os.path.abspath(self.imgs_path), self.cfg.TEST.result, os.path.splitext(img_name)[0] + '_seg.png')
        # cv2.imwrite(str(pred_path), pred_color)

    def run(self):
        torch.cuda.set_device(self.gpu)

        net_encoder = ModelBuilder.build_encoder(
            arch=self.cfg.MODEL.arch_encoder,
            fc_dim=self.cfg.MODEL.fc_dim,
            weights=self.cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=self.cfg.MODEL.arch_decoder,
            fc_dim=self.cfg.MODEL.fc_dim,
            num_class=self.cfg.DATASET.num_class,
            weights=self.cfg.MODEL.weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).cuda()
        segmentation_module.eval()

        dataset_test = TestDataset(self.cfg.list_test, self.cfg.DATASET)
        loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=0,
            drop_last=True)

        total_time = 0
        img_count = 0
        pbar = tqdm(total=len(loader_test))

        images =[]
        for batch_data in loader_test:
            batch_data = batch_data[0]
            img_ori = batch_data['img_ori']
            start_time = time.time()
            img_ori_resized = self._resize_keep_aspect_ratio(img_ori, max_size=640)
            batch_data['img_ori'] = img_ori_resized
            segSize = (img_ori_resized.shape[0], img_ori_resized.shape[1])
            img_data = batch_data['img_data'][0]

            with torch.no_grad():
                scores = torch.zeros(1, self.cfg.DATASET.num_class, segSize[0], segSize[1])
                scores = async_copy_to(scores, self.gpu)
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img_data
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, self.gpu)
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = pred_tmp
                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())

            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            img_count += 1
            print(f"图片 {img_count} 推理时间: {inference_time:.4f} 秒")

            image=self._visualize_result((batch_data['img_ori'], batch_data['info']), pred)
            images.append(image)
            torch.cuda.empty_cache()
            pbar.update(1)

        if img_count > 0:
            avg_time = total_time / img_count
            print(f"平均每张图片推理时间: {avg_time:.4f} 秒")
        print('Inference done!')
        return images

# # 如果需要在包外直接执行，也可添加入口：
# if __name__ == '__main__':
#     import sys
#     imgs = sys.argv[1] if len(sys.argv) > 1 else 'data'
#     tester = SemanticSegmentation("image_004.jpg")
#     tester.run()