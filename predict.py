import os
import numpy as np
import torch
import time
import cv2
from unet_model import UNet


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    #W和b等参数通过下面代码调入
    net.load_state_dict(torch.load('./model/best_model.pth', map_location=device))
    net.eval()
    test_path = './data/test/[0,0,1]/'
    testImgs = os.listdir(test_path)
    for testImg in testImgs:
        save_path = './data/result/[0,0,1]/' + testImg
        img = cv2.imread(test_path + testImg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        start = time.time()
        #1×1×512×512
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        #print(img.shape)
        img_tensor = torch.from_numpy(img)
        #print(img_tensor)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        #print(pred) #输出为张量1×1×512×512
        pred = np.array(pred.data.cpu()[0])[0]
        #print(pred) #输出为数组512×512
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        print('cost time : ', time.time() - start)
        cv2.imwrite(save_path, pred)