# CMNF-Net
Official PyTorch implementation of the paper:

> **CMNF-Net: Cross-modal Multi-scale Nonlinear Fusion Network for Unsupervised Video Object Segmentation**
>
> Pengxin Xu, Yong Lin*
>
> Ningxia Key Laboratory of Artificial Intelligence and Technology Education, School of Physics and Electronic Information Engineering, Ningxia Normal University, Ningxia 756000, China
>
> *Corresponding author: linyong@nxnu.edu.cn


## Abstract
Unsupervised Video Object Segmentation (UVOS) aims to segment salient objects in video sequences without manual annotations. Existing methods still face challenges of coarse object boundaries, limited small-region accuracy, and weak robustness under dynamic background interference, which primarily arise from insufficient multi-scale representation and weak cross-modal interaction.

To address these limitations, we propose **CMNF-Net**, a dual-stream framework that effectively exploits appearance and optical flow cues. At the encoder stage, we introduce a **Hierarchical Feature Fusion (HFF)** module to enhance both semantic representations and fine-grained details via top-down semantic guidance and cross-scale collaboration. We further design a **Distance-Aware Local Nonlinear Fusion (DALNF)** module that dynamically generates fusion weights through geometric distances in latent space. Combined with a gated dual-stream structure and progressive multi-scale decoder, it effectively alleviates feature misalignment and improves foreground localization and boundary recovery.

Extensive experiments on DAVIS 2016, FBMS, and YouTube-Objects demonstrate that CMNF-Net achieves state-of-the-art performance, with a **J&F** score of **89.4%** on DAVIS 2016, a **J** score of **83.7%** on FBMS, and a **J** score of **77.4%** on YouTube-Objects. Ablation studies further confirm the effectiveness of both HFF and DALNF.

## Environment Requirements
- Python 3.8+
- PyTorch 2.4.1
- TorchVision 0.20.0
- CUDA 12.1
- Other dependencies: transformers, numpy, opencv-python, Pillow, scipy, scikit-image

## Setup
1. Download the datasets:
[DUTS](http://saliencydetection.net/duts/#org3aad434), 
[DAVIS](https://davischallenge.org/davis2017/code.html), 
[FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets), 
[YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects), 
[Long-Videos](https://www.kaggle.com/datasets/gvclsu/long-videos).

2. Estimate and save optical flow maps from the videos using [RAFT](https://github.com/princeton-vl/RAFT).

3. I also provide the pre-processed datasets:
[DUTS](https://drive.google.com/file/d/1Q-bvC1XM0cAp41a1oTSwhRsy8o6titr7/view?usp=drive_link),
[DAVIS](https://drive.google.com/file/d/1kx-Cs5qQU99dszJQJOGKNb-wD_090q6c/view?usp=drive_link),
[FBMS](https://drive.google.com/file/d/1Zgt5ouwFeTpMTemfNeEFz7uEUo77e2ml/view?usp=drive_link),
[YouTube-Objects](https://drive.google.com/file/d/1t_eeHXJ30TWBNmMzE7vfS0izEafiBfgn/view?usp=drive_link),
[Long-Videos](https://drive.google.com/file/d/1gZm1QBT_6JmHhphNrxuSztcqkm_eI6Sq/view?usp=drive_link).



##  Running 

### Training

Start CMNF training with:
```
python run.py --train
```



### Testing
Run CMNF with:
```
python run.py --test
```





## Contact
Code and models are provided for non-commercial research purposes only.\
For any questions, please contact:
```
E-mail: 1147261824xu@gmail.com
E-mail: linyong@nxnu.edu.cn
```
