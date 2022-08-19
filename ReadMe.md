# UNet for Weizmann Horse
The network architecture and basic training flow of semantic segmentation model UNet are implemented under the framework of PyTorch, and the indicators on verification set are reported on Weizmann Horse data set

# Installation
This code runs under:
- python 3.9.12
- pytorch 1.11.0
- numpy 1.16.6

# Model
The UNET network model weight download link is: 
https://pan.baidu.com/s/1SJh7uJIYCpO78TrZpqOitw

Extract Code is:
ai66

# Start
Download the model and put it in“Weight_ path”, then run the train.py file to train the model. The program shows some of the resulting images in“Save_ path”

# prepare your data
The data set is divided into a training set and a validation set, each including  image and segment_ image

# Train your model
1. The code for the UNET network model is in UNet.py
2. Some of the functions I customize in this project are grouped together in util.py
3. run the `train.py` file. 
4. set _data.py is used to make your own data set, be careful to change the path in it
