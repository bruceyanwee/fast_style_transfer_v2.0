import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models,transforms
import torch.optim as optim
"""
测试 转置卷积恢复图像的效果
1. 用一个正向卷积对一张图像进行卷积，得到各层次的 feature map
2. 然后接着用转置卷积进行上采样，进行还原，目标是和原图像还原成一样的
3. 因为是在macbook上实验，因此 cnn 提取特征阶段，不用太深，能展示过程就行
"""
MAX_IMAGE_SIZE = 512
class Encode_Decode(nn.Module):
    def __init__(self):
        super(Encode_Decode, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(256,128,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,2,2)
        )
    def forward(self, x):
        x = self.down_sample(x)
        x = self.up_sample(x)
        return x


def show(img):
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img / 255).clip(0, 1)
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

def itot(img):
    # Rescale the image
    H, W, C = img.shape
    image_size = tuple([int((float(MAX_IMAGE_SIZE) / max([H, W])) * x//8*8) for x in [H, W]])
    itot_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    # Subtract the means
    normalize_t = transforms.Normalize([103.939, 116.779, 123.68], [1, 1, 1])
    tensor = normalize_t(itot_t(img) * 255)
    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor

def ttoi(tensor):
    # Add the means
    ttoi_t = transforms.Compose([
        transforms.Normalize([-103.939, -116.779, -123.68], [1, 1, 1])])
    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    img = ttoi_t(tensor)
    img = img.cpu().numpy()
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img
def test():
    # 读取数据
    model = Encode_Decode()
    optimizer = optim.Adam(model.parameters(),0.001)
    loss_fn = nn.MSELoss()
    img_origin = cv2.imread('/Users/yanwei/Pictures/Dream Afar/New York, U.S..jpg')
    x = itot(img_origin)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        y = model(x)
        loss = loss_fn(x,y)
        loss.backward()
        optimizer.step()
        if epoch % 5 ==0:
            print('train_epoch:{} loss:{:.4f}'.format(epoch,loss))

    # 输出结果比较
    model.eval()
    y = model(x)
    img_y = ttoi(y.clone().detach())
    show(img_origin)
    show(img_y)


if __name__ == '__main__':
    test()


