# MobileNetV3
Unofficial Implementation of MobileNetV3 and its experiments
## Paper
- [Searching for MobileNetV3 paper](https://arxiv.org/abs/1905.02244)
- Author: Andrew Howard(Google Research), Mark Sandler(Google Research, Grace Chu(Google Research), Liang-Chieh Chen(Google Research), Bo Chen(Google Research), Mingxing Tan(Google Brain), Weijun Wang(Google Research), Yukun Zhu(Google Research), Ruoming Pang(Google Brain), Vijay Vasudevan(Google Brain), Quoc V. Le(Google Brain), Hartwig Adam(Google Research)

## Requirements
- torch==1.0.1
- torchvision
- tqdm
Create a conda env with Python 3.6 and install the following. Activate your conda env

I used GNU/Linux 5.15.0-86-generic x86 64, which had 2 GPUs. I trained my models parallely in these 2 gpus.

## Prerequisite - Datasets: 
<br>
Step 1: Download the Datasets
Datasets Used: 
- CIFAR-10 and CIFAR-100 datasets would automatically be downloaded on running the code from torchvision. For CIFAR-100, I experimented with resize (224, 224).<br>
ImageNet Dataset: https://www.image-net.org/download-images.php, Please download the train and val datasets folders from the given website. 

## Experiments: 
1. Train the models for the following datasets: 
command to use : python main.py 

Steps to train MobileNetV3-Small:

a) CIFAR-10: python main.py --dataset-mode CIFAR10 --model-mode small

b) CIFAR-100: python main.py --dataset-mode CIFAR100 --model-mode small

c) ImageNet: python main.py --dataset-mode ImageNet --model-mode small

Steps to train MobileNetV3-Large:

a) CIFAR-10: python main.py --dataset-mode CIFAR10 --model-mode large

b) CIFAR-100: python main.py --dataset-mode CIFAR100 --model-mode large

c) ImageNet: python main.py --dataset-mode ImageNet --model-mode large

The epoch result of the Training logs can be found in reporting/best_model_large.txt and reporting/best_model_small.txt respectively. You will also find model checkpoints under the folder checkpoint/* , both these folders are created once the above code is run.

2) To evaluate the models, we have to add a suffix --evaluate True to the above

Steps to test MobileNetV3-Small:

a) CIFAR-10: python main.py --dataset-mode CIFAR10 --model-mode small --evaluate True

b) CIFAR-100: python main.py --dataset-mode CIFAR100 --model-mode small --evaluate True

c) ImageNet: python main.py --dataset-mode ImageNet --model-mode small --evaluate True

Steps to test MobileNetV3-Large:

a) CIFAR-10: python main.py --dataset-mode CIFAR10 --model-mode large --evaluate True

b) CIFAR-100: python main.py --dataset-mode CIFAR100 --model-mode large --evaluate True

c) ImageNet: python main.py --dataset-mode ImageNet --model-mode large --evaluate True

3) To evaluate other parameters like width multiplier, etc, the below parameters are set:
Options:
- `--dataset-mode` (str) - dataset (CIFAR10, CIFAR100, ImageNet).
- `--epochs` (int) - number of epochs (default: 100).
- `--batch-size` (int) - batch size, (default: 128).
- `--learning-rate` (float) - learning rate, (default: 1e-1).
- `--dropout` (float) - dropout rate, (default: 0.3).
- `--model-mode` (str) - large or small, (default islarge).
- `--load-pretrained` (bool) - (default: False).
- `--evaluate` (bool) - Used when testing. (default: False).
- `--multiplier` (float) - (default: 1.0).

You can use these hyperparameters for training the  model. 

## Implementation

- There is no official implementation given for MobileNetV3-Small and MobileNetV3-Large models. There are few unofficial models implemented by few ML enthusiasts and there exists a TensorFlow MobileNetV3 model that I can compare to. Here is the link for keras Tensor Flow model - https://github.com/keras-team/keras/blob/v2.14.0/keras/applications/mobilenet_v3.py 

I took inspiration, but did not reuse the code from Keras Tensorflow MobileNetV3 model that was available, but I implemented my MobileNetV3 model (file mobilenetV3.py) from scratch, trained it for CIFAR-10 CIFAR-100, and ImageNet datasets using Pytorch. I also re-evaluated my model using experiments written by me under main.py. I used idea of using AverageMeter and ProgressMeter using tqdm for logging from internet. It is readily available and good library for logging the model training and evaluation runs. 


