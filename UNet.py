import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import os
import random as rand
import numpy as np
from torch.utils.data import Dataset
from utils import *
from set_data import *
import shutil


# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')    # 输出结果保存路径
args = parser.parse_args()

# 设置超参数
EPOCH = 20
pre_epoch = 0
BATCH_SIZE = 64
LR = 0.01




#########################################  实现网络 #############################################

class Conv_Block(nn.Module):
      def __init__(self,in_channel,out_channel):   #由于卷积块的输入输出通道不同，故也要定义成变量
          super(Conv_Block,self).__init__()
          self.layer = nn.Sequential(
               nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),   #如果填充0是没有特征的；所以用reflect保证整张图都是有特征的，加强特征提取能力
               nn.BatchNorm2d(out_channel),    #归一化处理，防止在ReLU前因为数据过大而导致数据性能的不稳定
               nn.Dropout2d(0.3),
               nn.LeakyReLU(),
               nn.Conv2d(out_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
               nn.BatchNorm2d(out_channel),
               nn.Dropout2d(0.3),
               nn.LeakyReLU()
               )              
      def forward(self,x):
          return self.layer(x)
          

class DownSample(nn.Module):
      def __init__(self,channel):   #由于各卷积块之间的输入输出通道不同，也要定义成变量
          super(DownSample,self).__init__()
          self.layer = nn.Sequential(
               nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),  #同样是下采样的作用，池化层丢失了太多特征故使用步长为2的卷积层替代
               nn.BatchNorm2d(channel),
               nn.LeakyReLU()
               )               
      def forward(self,x):
          return self.layer(x)
      
          
class UpSample(nn.Module):
      def __init__(self,channel):   
          super(UpSample,self).__init__()  #上采样还可以采用转置卷积，但会出现很多空洞，效果不佳。故此处采用插值法
          self.layer = nn.Conv2d(channel,channel//2,1,1)  #1x1的卷积不是用来作特征提取的，它只是用来降通道数 
          #很无语，在这句话后面多打了一个逗号，结果反复报错'tuple' object is not callable
          #上网查了以为是调用了不可调用的对象，debug无果。结果逐层print卷积层的shape，才发现这里的结果压根没有传出去               
      def forward(self,x,feature_map):
          up = F.interpolate(x,scale_factor=2,mode='nearest')  #最近邻插值法
          out = self.layer(up)
          return torch.cat((out,feature_map),dim=1)  #在C通道上进行的，所以dim=1
        
        
class UNet(nn.Module):
      def __init__(self):   
          super(UNet,self).__init__() 
          self.c1=Conv_Block(3,64)
          self.d1=DownSample(64)
          self.c2=Conv_Block(64,128)
          self.d2=DownSample(128)
          self.c3=Conv_Block(128,256)
          self.d3=DownSample(256)
          self.c4=Conv_Block(256,512)
          self.d4=DownSample(512)
          self.c5=Conv_Block(512,1024)
          self.u1=UpSample(1024)
          self.c6=Conv_Block(1024,512)
          self.u2=UpSample(512)
          self.c7=Conv_Block(512,256)
          self.u3=UpSample(256)
          self.c8=Conv_Block(256,128)
          self.u4=UpSample(128)
          self.c9=Conv_Block(128,64)
          self.out = nn.Conv2d(64,3,3,1,1)  #种类数num_classes设置为2，表示马和背景

         
      def forward(self,x):
          R1=self.c1(x)
          R2=self.c2(self.d1(R1))
          R3=self.c3(self.d2(R2))
          R4=self.c4(self.d3(R3))
          R5=self.c5(self.d4(R4))
          #print(R5.shape)
          D1=self.c6(self.u1(R5,R4))
          #print(D1.shape)
          D2=self.c7(self.u2(D1,R3))
          D3=self.c8(self.u3(D2,R2))
          D4=self.c9(self.u4(D3,R1))          
          return self.out(D4)
      
        
        
if __name__ == '__main__' :  #检验网络是否正确
     x=torch.randn(2,3,256,256) #用批次为2，通道数为3的256x256张量输入作验证
     #从BatchNorm2d来看，输入的应该是四维数据(N,C,H,W)，即（batchsize,channel,height,width)
     #注意pytorch框架中是NCHW格式，TensorFlow中默认NHWC
     net=UNet()
     print( net(x).shape )

          




















