# CSRNet-pytorch

This is a PyTorch Lightning implementation of [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062) in CVPR 2018 which is one of the most popular crowd counting methods. The original implementation is [here](https://github.com/leeyeehoo/CSRNet-pytorch).

## Datasets
ShanghaiTech Dataset: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8.13

PyTorch: 1.13.1

CUDA: 11.7

## Ground Truth

> Working on getting the code for generating ground truth work. If you have the density maps ready, you can skip this part.

## Training Process

Try `python train.py train.json val.json 0 0` to start training process.

## Validation

```bash
python test.py
```

## Results

> To be added soon.

## References

If you find this implementation useful, please give stars and cite the original paper:

```
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1091--1100},
  year={2018}
}
```
Please cite the Shanghai datasets and other works if you use them.

```
@inproceedings{zhang2016single,
  title={Single-image crowd counting via multi-column convolutional neural network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={589--597},
  year={2016}
}
```