##############  这个文件存放各种工具函数 ################

from PIL import Image
from torch import nn
from torch import optim
import torch
import cv2
import numpy as np

###########  对图片大小进行调整  ##############
def keep_image_size_open(path,size=(256,256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask



############ 计算MIOU #############
# 由于数据进行了归一化.以0.5为界：大于0.5的为目标（马），小于0.5的为背景
def get_miou(predict, mask):
    predict = torch.sigmoid(predict).cpu().data.numpy()
    mask = torch.sigmoid(mask).cpu().data.numpy()
    predict_ = predict > 0.5
    _predict = predict <= 0.5
    mask_ = mask > 0.5
    _mask = mask <= 0.5
    # 获取交并集
    intersection = (predict_ & mask_).sum()
    union = (predict_ | mask_).sum()
    _intersection = (_predict & _mask).sum()
    _union = (_predict | _mask).sum()
    if union < 1e-5 or _union < 1e-5:
        return 0
    miou = (intersection / union) * 0.5 + 0.5 * (_intersection / _union)
    return miou


########### 进行图像边界的获取 ##############
#随着物体尺寸的增加，物体内部像素以二次方形式增加而边界像素数量线性增加，这导致边界像素占总像素的比重变小
#由此对于大物体，即使边界分割效果不好，只要物体内部像素正确分割就仍然能取得较高的MIOU。
#故采用BIOU，忽略远离边界的像素，更关注物体边界附近的分割质量。
# 通过腐蚀操作获取边界：将原图缩小一圈，再用原图减去缩小的图，就得到了边界
# 通过腐蚀将毛刺消除了，方法是在卷积核大小中对图片进行卷积，取图像中（3*3）区域内的最小值
def get_boundary(mask, dilation_ratio=0.02, sign=1):
    # 通过sign判断把mask数值置为 0、1
    if sign == 1:
        mask = torch.argmax(mask,dim=1)
        mask = torch.sigmoid(mask).data.cpu().numpy()
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = mask.astype('uint8')
        #print(mask.shape)  #[1,256,256]
    elif sign == 0: 
        mask = torch.argmax(mask,dim=1)
        mask = mask.cpu()
        mask = np.array(mask).astype('uint8')
        #mask = mask.cpu().numpy()
        #print(mask.shape)  #[1,256,256]

    b, h, w = mask.shape
    new_mask = np.zeros([b, h + 2, w + 2])
    mask_erode = np.zeros([b, h, w])
    img_diag = np.sqrt(h ** 2 + w ** 2)   # 计算图像对角线长度   
    dilation = int(round(dilation_ratio * img_diag))  # 计算腐蚀的程度dilation
    if dilation < 1:
        dilation = 1
    #腐蚀的次数与对角线的长度成正比，如果小于1则直接赋1，dolation_ratio是函数的参数

    # 对一个batch中所有图片进行腐蚀
    for i in range(b):
        new_mask[i] = cv2.copyMakeBorder(mask[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  
    #这一行用于给原图的四周添加0，这样连边界区域的目标像素也会被腐蚀掉
    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(b):
        new_mask_erode = cv2.erode(new_mask[j], kernel, iterations=dilation)        # 腐蚀操作：kernel size是(3,3),iterations是腐蚀的次数
        mask_erode[j] = new_mask_erode[1: h + 1, 1: w + 1]        # 回填：由于之前向四周填充了0，故这里不需要再向四周填

    return mask - mask_erode


############### 计算BIOU ################
#Boundary IoU就是轮廓间的IoU,对大对象的边界误差更为敏感且不会过分惩罚小对象的误差
#与MIOU相比，BIOU对物体边界分割的质量显然更加敏感，更够更好地评价不同分割算法的优劣
# 获取标签和预测的边界iou
def get_biou(pre, real, dilation_ratio=0.02):
    real_boundary = get_boundary(real, dilation_ratio ,sign=0) 
    #print(real_boundary.shape) #[1,256,256]
    pre_boundary = get_boundary(pre, dilation_ratio ,sign=1)
    #print(pre_boundary.shape) #[1,256,256]
    
    B, H, W = pre_boundary.shape
    intersection = 0
    union = 0
    # 计算交并比
    for k in range(B):
        intersection += ((pre_boundary[k] * real_boundary[k]) > 0).sum()
        union += ((pre_boundary[k] + real_boundary[k]) > 0).sum()
    #如果交集union==0时，可能出现除以0错误的问题，所以添加下面两行
    if union < 1:
        return 0
    biou = intersection / union

    return biou
    
    
##################  对网络输出的图片可视化 ###################
def vis_lable(img):
    img=Image.open(img)
    img=np.array(img)*255.0
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    