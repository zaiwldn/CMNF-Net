from .transforms import *
import os
import random
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainYTVOS(torch.utils.data.Dataset):
    def __init__(self, root, split, clip_n):
        self.root = root
        self.split = split
        with open(os.path.join(root, 'ImageSets', '{}.txt'.format(split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        # 循环直到找到有效的视频帧
        while True:
            video_name = random.choice(self.video_list)
            img_dir = os.path.join(self.root, self.split, 'JPEGImages', video_name)
            flow_dir = os.path.join(self.root, self.split, 'JPEGFlows', video_name)
            mask_dir = os.path.join(self.root, self.split, 'Annotations', video_name)

            # 获取文件列表
            img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
            flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
            mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

            # 检查是否有无效的空列表
            if not img_list:
                print(f"警告：视频 {video_name} 的 JPEGImages 目录为空，跳过该视频")
                continue  # 重新选择视频
            if not flow_list:
                print(f"警告：视频 {video_name} 的 JPEGFlows 目录为空，跳过该视频")
                continue
            if not mask_list:
                print(f"警告：视频 {video_name} 的 Annotations 目录为空，跳过该视频")
                continue
            if len(img_list) != len(flow_list) or len(img_list) != len(mask_list):
                print(f"警告：视频 {video_name} 的文件数量不匹配（图像/光流/掩码），跳过该视频")
                continue

            # 所有检查通过，选择帧并处理
            all_frames = list(range(len(img_list)))
            frame_id = random.choice(all_frames)
            img = Image.open(img_list[frame_id]).convert('RGB')
            flow = Image.open(flow_list[frame_id]).convert('RGB')
            mask = Image.open(mask_list[frame_id]).convert('P')

            # 后续的 resize、翻转等处理（保持不变）
            # img = img.resize((384, 384), Image.BICUBIC)
            # flow = flow.resize((384, 384), Image.BICUBIC)
            # mask = mask.resize((384, 384), Image.NEAREST)

            img = img.resize((512, 512), Image.BICUBIC)
            flow = flow.resize((512, 512), Image.BICUBIC)
            mask = mask.resize((512, 512), Image.NEAREST)

            if random.random() > 0.5:
                img = TF.hflip(img)
                flow = TF.hflip(flow)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                flow = TF.vflip(flow)
                mask = TF.vflip(mask)

            imgs = self.to_tensor(img).unsqueeze(0)
            flows = self.to_tensor(flow).unsqueeze(0)
            indices = torch.zeros(1, 1, 1, 1)
            masks = self.to_mask(mask).unsqueeze(0)
            masks = (masks != 0).long()
            return {'imgs': imgs, 'flows': flows, 'indices': indices, 'masks': masks}
