import torch.optim as optim
from PIL import Image
import os
import random
import torch
from torchvision import transforms, datasets,models
from torch.utils.data import Dataset, DataLoader
from MultiAdaption import get_encoder,MultiAdaptionNet,Decoder2,Decoder_light2
from util import IMAGE_PATH
from util import loss_disentanglement,loss_perceptual,loss_identity
from util import MODEL_SAVE_PATH
MAX_IMAGE_SIZE = 256
# hyper parameters
LEARNING_RATE = 0.00001

# 4张图片数据集，包括两张自然image，两张风格image
class Dataset_2x2(Dataset):
    def __init__(self,file_c_path,file_s_path):
        self.file_c_path = file_c_path
        self.file_s_path = file_s_path
        self.cimage_name = [os.path.join(file_c_path,f_name) for f_name in os.listdir(file_c_path)]
        self.simage_name = [os.path.join(file_s_path,f_name) for f_name in os.listdir(file_s_path)]
        self.max_length = max(len(self.cimage_name), len(self.simage_name)) # 读取一百次
        self.min_length = min(len(self.cimage_name), len(self.simage_name))
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(MAX_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return self.max_length
    def __getitem__(self, idx): # 每一次调用返回4张image
        # 读取两张 content 取余数是防止越界
        index_offset1,index_offset2 = random.randint(1,10), random.randint(1,10)
        cname1, cname2 = self.cimage_name[idx % self.min_length], self.cimage_name[(idx+index_offset1) % self.min_length]
        sname1, sname2 = self.simage_name[idx % self.min_length], self.simage_name[(idx+index_offset2) % self.min_length]
        cimg1, cimg2 = self.transform(Image.open(cname1)), self.transform(Image.open(cname1))  # img_tensor [C,H,W]
        simg1, simg2 = self.transform(Image.open(sname1)), self.transform(Image.open(sname2))
        # 组合四张
        x = torch.stack((cimg1, cimg2, simg1, simg2),dim=0)
        return x

# 获取数据集，加载图像，一次读入4张
def get_2x2_dataloader(batch_size=1): # 一次是四张图
    # 分别要从
    dataset_2x2 = Dataset_2x2(IMAGE_PATH+'/content',IMAGE_PATH+'/style')
    dataset_loader = DataLoader(dataset_2x2,batch_size = batch_size, shuffle=True)
    return dataset_loader

# 训练
def train(model,num_epoch=100,learning_rate=0.0001):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(),learning_rate)
    model = model.to(device)
    dataset_loader = get_2x2_dataloader(batch_size=1)
    model.train()
    for epoch in range(num_epoch):
        for x in dataset_loader:
            optimizer.zero_grad()
            x = x.squeeze(0).to(device) # x的形式是[4,3,H,W] 4张图像按照顺序（0，1是自然图像，2，3是风格图像）
            # 第一次经过网络forward
            Ics, features_x = model(x) # 返回的Ics 和 features_x 都是字典，包括Net输出图像和vgg中间提取的conv1_1 到 conv4_1的feature map
            # Ics[:-2]后面两个计算identity loss的不需要，因为是计算像素级别的，不是feature map -level
            features_Ics = model.forward(Ics[:-2],cal_loss=True)
            # 计算各种损失
            loss_i = loss_identity(Ics[-2],x[0],Ics[-1],x[2])
            loss_p = loss_perceptual(features_x,features_Ics)
            loss_d = loss_disentanglement(features_x,features_Ics)
            loss_all = loss_i + loss_p + loss_d
            loss_all.backward()
            optimizer.step()
            print('train_epoch:{} loss_all:{:.4f} loss_i:{:.4f} loss_p:{:.4f} loss_d:{:.4f}'.format(epoch, loss_all, loss_i, loss_p, loss_d))
    torch.save(model.state_dict(),MODEL_SAVE_PATH)
    return model

# test tansfer 的效果
def test_transfer(model,using_W = True,cimg_path='/content/in4.jpg',simg_path='/style/style3.jpg'):
    transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    tensor_c = transform(Image.open(IMAGE_PATH+cimg_path))
    tensor_s = transform(Image.open(IMAGE_PATH+simg_path))
    img_trans = model.transfer(tensor_c = tensor_c.unsqueeze(0).to('cuda'),tensor_s = tensor_s.unsqueeze(0).to('cuda'),using_Whiten = using_W)
    return img_trans.detach()

if __name__ == '__main__':
    encoder = get_encoder(pretrained=False)
    model = MultiAdaptionNet(encoder, decoder=Decoder_light2(512, 3))
    train(model,10,0.0001)