# Duet of ViT and CNN: multi-scale dual-branch network for fine-grained image classification of marine organisms
Official PyTorch code for the paper: Duet of ViT and CNN: multi-scale dual-branch network for fine-grained image classification of marine organisms (IMTS)
## Overall Architecture
![img](https://github.com/Xiaosigz/MSDBN/blob/main/Model.png)
## Prerequisites
* Python >= 3.6
* Pytorch = 1.9
* Torchvision
* Apex
## Datasets
You can get the datasets from the links below:
* [ASLO-Plankton](http://ouc.ai/dataset/ASLO-Plankton.zip)
* [Sharks](https://www.kaggle.com/larusso94/shark-species)
* [WildFish](https://github.com/PeiqinZhuang/WildFish)
## Train
To train our model on datasets, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --dataset xx --split overlap --name xx
```
## Acknowledgement
Many thanks for the work and code sharing of [Pytorch Image Models(timm)](https://github.com/huggingface/pytorch-image-models)
