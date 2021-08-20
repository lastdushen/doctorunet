import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

#Dataset在torch.utils.data中
class ISBI_Loader(Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        #glob.glob功能 匹配符合规则的文件 ”*”匹配0个或多个字符
        #所以后面可以通过len(self.imgs_path)看图片有多少个
        #也可以通过self.imgs_path[index]来得到index图片的路径
        self.imgs_path = glob.glob(os.path.join(data_path, 'imgs_wsss4luad/*.png'))

    def augment(self, image, flipCode):
        # 进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):

        image_path = self.imgs_path[index]
        #print(image_path)
        label_path = image_path.replace('imgs_wsss4luad', 'labels_wsss4luad')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):

        return len(self.imgs_path)

