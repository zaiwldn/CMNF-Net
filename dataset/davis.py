from .transforms import *
import os
import random
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split, clip_n):
        self.root = root
        with open(os.path.join(root, 'ImageSets', '{}/{}.txt'.format(year, split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        flow_dir = os.path.join(self.root, 'JPEGFlows', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # select training frame
        all_frames = list(range(len(img_list)))
        frame_id = random.choice(all_frames)
        img = Image.open(img_list[frame_id]).convert('RGB')
        flow = Image.open(flow_list[frame_id]).convert('RGB')
        mask = Image.open(mask_list[frame_id]).convert('P')

        # resize to 384p
        img = img.resize((384, 384), Image.BICUBIC)
        flow = flow.resize((384, 384), Image.BICUBIC)
        mask = mask.resize((384, 384), Image.NEAREST)

        # joint flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            flow = TF.hflip(flow)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            flow = TF.vflip(flow)
            mask = TF.vflip(mask)

        # convert formats
        imgs = self.to_tensor(img).unsqueeze(0)
        flows = self.to_tensor(flow).unsqueeze(0)
        indices = torch.ones(1, 1, 1, 1)
        masks = self.to_mask(mask).unsqueeze(0)
        masks = (masks != 0).long()

        # # 打印掩码的唯一值
        # print(f"Mask unique values: {torch.unique(masks)}")
        # # 检查掩码的形状和类型
        # print(f"Mask shape: {masks.shape}")
        # print(f"Mask type: {masks.dtype}")
        # 可视化掩码
        # mask_np = masks.squeeze(0).cpu().numpy()  # 移除多余的维度
        # mask_np = mask_np.squeeze(0)  # 再次移除多余的维度
        # mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        # mask_img.save(f"mask_{idx}.png")

        # def check_masks(root_dir):
        #     mask_dir = os.path.join(root_dir, 'Annotations', '480p')
        #     video_list = os.listdir(mask_dir)
        #     for video_name in video_list:
        #         video_mask_dir = os.path.join(mask_dir, video_name)
        #         mask_files = sorted(glob(os.path.join(video_mask_dir, '*.png')))
        #         for mask_file in mask_files:
        #             mask = Image.open(mask_file).convert('P')
        #             mask_np = np.array(mask)
        #             print(f"Mask file: {mask_file}, unique values: {np.unique(mask_np)}")
        #
        # # 调用函数检查标注文件
        # check_masks('F:/dataset/DAVIS-data/DAVIS')

        return {'imgs': imgs, 'flows': flows, 'indices': indices, 'masks': masks}


class TestDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split):
        self.root = root
        self.year = year
        self.split = split
        self.init_data()

    def read_img(self, path):
        pic = Image.open(path).convert('RGB')
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path).convert('P')
        transform = LabelToLongTensor()
        return transform(pic)

    def init_data(self):
        with open(os.path.join(self.root, 'ImageSets', self.year, self.split + '.txt'), 'r') as f:
            self.video_list = sorted(f.read().splitlines())
            print('--- DAVIS {} {} loaded for testing ---'.format(self.year, self.split))

    def get_snippet(self, video_name, frame_ids):
        img_path = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        flow_path = os.path.join(self.root, 'JPEGFlows', '480p', video_name)
        mask_path = os.path.join(self.root, 'Annotations', '480p', video_name)
        imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        flows = torch.stack([self.read_img(os.path.join(flow_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        masks = torch.stack([self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))) for i in frame_ids]).unsqueeze(0)
        if self.year == '2016':
            masks = (masks != 0).long()
        files = ['{:05d}.png'.format(i) for i in frame_ids]
        return {'imgs': imgs, 'flows': flows, 'masks': masks, 'files': files}

    def get_video(self, video_name):
        frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'JPEGImages', '480p', video_name))])
        yield self.get_snippet(video_name, frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
