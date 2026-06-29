import utils
import os
import time
import numpy as np
from . import metrics


class Evaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.img_saver = utils.ImageSaver()
        self.sdm = utils.DAVISLabels()
        self.total_j_scores = []  # 新增：存储所有视频的J分数
        self.total_f_scores = []  # 新增：存储所有视频的F分数
        self.total_g_scores = []  # 新增：存储所有视频的G分数

    def evaluate_video(self, model, video_name, video_parts, output_path):
        for vos_data in video_parts:
            imgs = vos_data['imgs'].cuda()
            flows = vos_data['flows'].cuda()
            files = vos_data['files']
            masks = vos_data['masks'].cuda()  # 真实掩码

            # inference
            t0 = time.time()
            vos_out = model(imgs, flows)
            t1 = time.time()

            # save output
            for i in range(len(files)):
                fpath = os.path.join(output_path, video_name, files[i])
                data = ((vos_out['masks'][0, i, 0, :, :].cpu().byte().numpy(), fpath), self.sdm)
                self.img_saver.enqueue(data)

            # 获取预测掩码和真实掩码
            res_masks = vos_out['masks'].cpu().numpy()  # 假设输出形状为 (B, L, C, H, W)
            gt_masks = masks.cpu().numpy()  # 假设真实掩码形状为 (B, L, C, H, W)

            # 调整掩码形状
            gt_masks = gt_masks[0, :, 0, :, :]  # (L, H, W)
            res_masks = res_masks[0, :, 0, :, :]  # (L, H, W)

            # 计算J和F分数
            j_scores = []
            f_scores = []
            for frame_id in range(gt_masks.shape[0]):
                j_score = metrics.db_eval_iou(gt_masks[frame_id], res_masks[frame_id])
                f_score = metrics.db_eval_boundary(gt_masks[frame_id], res_masks[frame_id])
                j_scores.append(j_score)
                f_scores.append(f_score)
            j_score = np.mean(j_scores)
            f_score = np.mean(f_scores)
            g_score = (j_score + f_score) / 2  # 计算G分数

            print(f"{video_name} - J: {j_score:.4f}, F: {f_score:.4f}, G: {g_score:.4f}")

            # 新增：将当前视频的J、F和G分数添加到总列表
            self.total_j_scores.append(j_score)
            self.total_f_scores.append(f_score)
            self.total_g_scores.append(g_score)

        return t1 - t0, imgs.size(1)

    def evaluate(self, model, output_path):
        model.cuda()
        total_seconds, total_frames = 0, 0
        for video_name, video_parts in self.dataset.get_videos():
            os.makedirs(os.path.join(output_path, video_name), exist_ok=True)
            seconds, frames = self.evaluate_video(model, video_name, video_parts, output_path)
            total_seconds = total_seconds + seconds
            total_frames = total_frames + frames
            print('{} done, {:.1f} fps'.format(video_name, frames / seconds))

        # 新增：计算总的J、F和G指标
        mean_j = np.mean(self.total_j_scores)
        mean_f = np.mean(self.total_f_scores)
        mean_g = np.mean(self.total_g_scores)
        print(f"Total Mean J: {mean_j:.5f}, Total Mean F: {mean_f:.5f}, Total Mean G: {mean_g:.5f}")

        print('total fps: {:.1f}\n'.format(total_frames / total_seconds))
        self.img_saver.kill()



#
# class Evaluator(object):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.img_saver = utils.ImageSaver()
#         self.sdm = utils.DAVISLabels()
#
#     def evaluate_video(self, model, video_name, video_parts, output_path):
#         for vos_data in video_parts:
#             imgs = vos_data['imgs'].cuda()
#             flows = vos_data['flows'].cuda()
#             files = vos_data['files']
#
#             # inference
#             t0 = time.time()
#             vos_out = model(imgs, flows)
#             t1 = time.time()
#
#             # save output
#             for i in range(len(files)):
#                 fpath = os.path.join(output_path, video_name, files[i])
#                 data = ((vos_out['masks'][0, i, 0, :, :].cpu().byte().numpy(), fpath), self.sdm)
#                 self.img_saver.enqueue(data)
#         return t1 - t0, imgs.size(1)
#
#     def evaluate(self, model, output_path):
#         model.cuda()
#         total_seconds, total_frames = 0, 0
#         for video_name, video_parts in self.dataset.get_videos():
#             os.makedirs(os.path.join(output_path, video_name), exist_ok=True)
#             seconds, frames = self.evaluate_video(model, video_name, video_parts, output_path)
#             total_seconds = total_seconds + seconds
#             total_frames = total_frames + frames
#             print('{} done, {:.1f} fps'.format(video_name, frames / seconds))
#         print('total fps: {:.1f}\n'.format(total_frames / total_seconds))
#         self.img_saver.kill()
#
