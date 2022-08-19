import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from utils import *
import random as rand
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


########################################## 准备数据集 ##############################################


#### 准备训练集 ####

transform=transforms.Compose([
         transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.name=os.listdir(os.path.join(path,'mask'))  #通过地址拼接获得标签数据集的地址,并通过listdir获得该文件下的所有文件名
        #这里有一个巧妙的地方，把原mask集按0.85：0.15的要求划分为子集，那么在horse集中的图片只能出现在mask训练集和测试集二者之一中
        #因此把self.name定义为mask集中数据的地址，这样才能一定在horse集中找到对应的数据
      
    def __len__(self):
        return len(self.name)   #文件名的数量就是数据集的数量
        
    def __getitem__(self,index):
        segment_name = self.name[index]   #获取对应标签（也就是分割好了的图片）的名字
        segment_path = os.path.join(self.path,'mask',segment_name)  #获取标签的地址
        image_path = os.path.join(self.path,'horse',segment_name)  #获取原图的地址
        #可以发现图片大小不一致，但网络的输入应该固定大小，故进行等比缩放（直接缩放会变形
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)
      
if __name__ == '__main__':
     data_example = MyDataset(r'C:\Users\User\Desktop\UNet\archive\weizmann_horse_db')
     print(data_example[0][0].shape)
     print(data_example[0][1].shape)


### 将mask集拆分为mask1（0.85用于训练）与mask2集（0.15用于测试）###
def creatdataset(data_path,batchsize):#创建训练验证集
    dataset = MyDataset(data_path)
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    #进行训练验证集的划分
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size = batchsize, num_workers = 0, pin_memory = True)
    train_loader = DataLoader(train_set, shuffle = True, **loader_args)
    val_loader = DataLoader(val_set, shuffle = False, **loader_args)
    return train_loader,val_loader
 