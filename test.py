import os
import torch
from utils import keep_image_size_open
from UNet import *
from utils import *
from set_data import *
from torchvision.utils import save_image


net=UNet().cpu()
weights='params/unet.pth'
if os.path.exists(weights):    #加载权重
      net.load_state_dict(torch.load(weights))
      print('successfully load') 
else:
      print('not successful')
      
_input = input(r'C:\Users\User\Desktop\UNet\archive\weizmann_horse_db\horse002.png')

img = keep_image_size_open(_input) 
img_data = transform(img).cpu()
#print(img_data.shape) #[3,256,256]
img_data = torch.unsqueeze(img.data,dim=0)  #由于数据集中只有三维，为了适配网络，升维以增加batch维度
out = net(img_data)   #out就是经过网络分割处理后的图
#save_image(out,'predict_result/result.png')
print(out)

#vis_lable(out)  #可视化分割的结果
#以分类任务举例，假如数据集包括狗，猫，马。num_classes=4,归一化后为0，1，2，3；
#令 img=np.array(img)   print(set(img.reshape(-1).tolist())) 即把数组展平后去除重复元素
#如果结果为{0，1}，说明是狗 ， 如果结果是{0，2}，说明是猫
