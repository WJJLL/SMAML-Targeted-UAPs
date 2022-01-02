# SMAML-Targeted-UAPs
## Setup
python=3.8.12
torch=1.4.0
torchvision=0.5.0
## Config
Edit the paths accordingly in `config.py`
## DataSets
The code supports training UAPs on ImageNet, MS COCO, PASCAL VOC
### ImageNet
The [ImageNet](http://www.image-net.org/) dataset should be preprocessed, such that the validation images are located in labeled subfolders as for the training set. You can have a look at this [bash-script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) if you did not process your data already. Set the paths in your `config.py`.
```
IMAGENET_Train_PATH = "./Data/ImageNet/train"
IMAGENET_Test_PATH = './Data/ImageNet/val'
```
### COCO
The [COCO](https://cocodataset.org/#home) 2017 images can be downloaded from here for [training](http://images.cocodataset.org/zips/train2017.zip) and [validation](http://images.cocodataset.org/zips/val2017.zip). After downloading and extracting the data update the paths in your `config.py`.
```
COCO_2017_TRAIN_IMGS = "./Data/COCO/train2017/"
COCO_2017_TRAIN_ANN = "./Data/COCO/annotations/instances_train2017.json"
COCO_2017_VAL_IMGS = "./Data/COCO/val2017/"
COCO_2017_VAL_ANN = "./Data/COCO/annotations/instances_val2017.json"
```
### PASCAL VOC
The training/validation data of the [PASCAL VOC2012 Challenge](http://host.robots.ox.ac.uk/pascal/VOC/) can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). After downloading and extracting the data update the paths in your `config.py`.
```
VOC_2012_ROOT = "./Data/VOCdevkit/"
```
## Run
Run `bash run.sh` to generate UAPs or test performance under 10-Targets setting. The bash script should be easy to adapt to perform different experiments.

## Noise models (UAPs)
We provide targeted UAPs trained against Ensemble of ImageNet Model in the folder `noise_model`
