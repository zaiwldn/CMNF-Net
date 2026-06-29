# CMNF-Net


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
Pengxin Xu: 1147261824xu@gmail.com
Yong Lin: linyong@nxnu.edu.cn
```
