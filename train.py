from torch import nn
from torch import optim
import torch
import cv2
from torch.utils.data import DataLoader
from UNet import *
from utils import *
from set_data import *
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
import argparse

# 选择cpu或gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练权重的地址
weight_path='params/unet.pth'   
# 定义数据集地址
data_path=r'C:\Users\User\Desktop\UNet\archive\weizmann_horse_db'
# 定义训练结果的保存路径
save_path='train_image'
# 设置超参数
EPOCH = 10
pre_epoch = 0
BATCH_SIZE = 1

net = UNet().to(device)

optimizer = optim.Adam(net.parameters())
#loss_fun = nn.BCELoss()  #采用分类损失函数BCEloss，二分类交叉熵
#使用这个损失函数要注意输入值范围要在0~1之间，否则报错：all elements of input should be between 0 and 1。通常先用F.sigmoid处理一下数据
criterion = nn.BCEWithLogitsLoss(weight=None,reduction='mean',pos_weight=None)
#等价于F.sigmoid + torch.nn.BCEloss ,相当于是在BCELoss预测结果的基础上先做了个sigmoid,然后正常算loss
#但也要注意到，如果网络本身结尾也用sigmoid处理过，那么会出现问题
#mean表示获得损失后的行为，weight是每个类别的权重
     
#加载训练集和测试集
train_loader,val_loader = creatdataset(data_path,batchsize=BATCH_SIZE)
 
#训练   
if __name__ == '__main__':
     if os.path.exists(weight_path):    #加载权重
        net.load_state_dict(torch.load(weight_path))
        print('successfully load weight')
     else:
        print('not successful')
         
     for epoch in range(pre_epoch, EPOCH):    # 循环训练回合，每回合会以批量为单位训练完整个训练集，一共训练EPOCH个回合
     
          net.train()     
          print("start training...")
          for i,(image,segment_image) in enumerate(train_loader):   # 循环每次取一批量的图像与标签
                image,segment_image = image.to(device),segment_image.to(device)
                #print(image.shape)  #[1,3,256,256]
                #print(segment_image.shape)   #[1,3,256,256]
                out_image = net(image)    #得到输出图
                #print(out_image.shape)   #[1,3,256,256]
                train_loss = criterion(out_image,segment_image)
                train_miou = get_miou(out_image,segment_image) #计算评价指标MIOU
                #train_biou = get_biou(out_image,segment_image) #计算评价指标Boundary IoU

                optimizer.zero_grad()  #优化器梯度清零
                train_loss.backward()  #反向计算
                optimizer.step()   #更新梯度
                
                if i%5==0:   #每隔五次打印出训练情况
                   print(f'Epoch{epoch}-{i} train_loss =>{train_loss.item()}')
                   print(f'          train_MIOU =>{train_miou.item()}')
                   #print(f'          train_BIOU =>{train_biou.item()}')
                  
                if i%50==0:   #每隔50个批次保存一次权重
                   torch.save(net.state_dict(),weight_path)  
                
                _image=image[0]
                _segment_image=segment_image[0]
                _out_image=out_image[0]
                #以第一张图片为例，展示分割效果
                img=torch.stack([_image,_segment_image,_out_image],dim=0)
                save_image(img,f'{save_path}/{i}.png')
                
            
          #测试
          net.eval()  #使用met.eval()关掉dropout,才能使用训练好的模型          
          print("start testing...")
          with torch.no_grad():
          #在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
          #而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建。
                for i,(image,segment_image) in enumerate(val_loader):   # 循环每次取一批量的图像与标签
                    image,segment_image = image.to(device),segment_image.to(device)
                    out_image = net(image)    #得到输出图
                    test_loss = criterion(out_image,segment_image)
                    test_miou = get_miou(out_image,segment_image)
                    #test_biou = get_biou(out_image,segment_image)
                   
                    if i%5==0:   #每隔五次打印出测试情况
                       print(f'Epoch{epoch}-{i} test_loss =>{test_loss.item()}')
                       print(f'          test_MIOU =>{test_miou.item()}')
                       #print(f'          test_BIOU =>{test_biou.item()}')
           
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                