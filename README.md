# # Focus-Longer-to-See-Better

This repo contains the pytorch implementation of paper  [Focus Longer to See Better:Recursively Refined Attention for Fine-Grained Image Classification](https://arxiv.org/abs/2005.10979)


# Requirements
* Python >= 3.6
* Pytorch > 0.4.1
* torchvision >= 0.2.1

# Training

The patches are already extracted and kept in the folder `Patches` . Use following command to train and test the results.

`python main.py -savedir <path_to_save_weights> -num_timesteps 10 -batch_size 32 -test_freq 2 -imgdir <Path_to_train_imgdir> -testimgdir <Path_to_test_imgdir>  -lr 0.001 --lr_steps 30 60 `  

# Citation
```
@InProceedings{Shroff_2020_CVPR_Workshops,
author = {Shroff, Prateek and Chen, Tianlong and Wei, Yunchao and Wang, Zhangyang},
title = {Focus Longer to See Better: Recursively Refined Attention for Fine-Grained Image Classification},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```

