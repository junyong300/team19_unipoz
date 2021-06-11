# [KAIST CS570] Team 19 Term Project
## Improved Human Pose Estimation with Alternative Backbone Network and Operations

## Overall Architecture

<!-- ![overall](https://user-images.githubusercontent.com/12655218/121641255-eab68700-cac9-11eb-820c-082dc3a5a7c2.png =500x200) -->

<img src = "https://user-images.githubusercontent.com/12655218/121641255-eab68700-cac9-11eb-820c-082dc3a5a7c2.png" width = 500>

## Original Baseline Model: UniPose
[Original UniPose](https://github.com/bmartacho/UniPose)

<img src = "https://camo.githubusercontent.com/40c2f795fa0e24aeabf2300a9f4bbf864f6a12162f7b5cea0a776d0b670b78d4/68747470733a2f2f70656f706c652e7269742e6564752f626d333736382f696d616765732f556e69706f73655f706970656c696e652e706e67" width = 500>

We perform experiments based on different method

* Use [You Only Look Once(Yolo)](https://pjreddie.com/darknet/yolo) as object detection module
* Change Backbone network into ResNet-50 and EfficientNet
* Use Swish Activation in encoder-decoder

<img src = "https://user-images.githubusercontent.com/12655218/121642093-053d3000-cacb-11eb-86d9-52504d5c35b9.png" width = 300>

## Experiments results
<img src = "https://user-images.githubusercontent.com/12655218/121642237-33bb0b00-cacb-11eb-9f50-ef9005617bcb.png" width = 700>

## Usage:
```
usage: unipose.py [-h] [--pretrained PRETRAINED] [--dataset DATASET]
                  [--train_dir TRAIN_DIR] [--val_dir VAL_DIR]
                  [--test_dir TEST_DIR] [--model_name MODEL_NAME]
                  [--model_arch MODEL_ARCH] [--backbone BACKBONE]
                  [--epoch EPOCH]
```
