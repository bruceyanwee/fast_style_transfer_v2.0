# 因为整个模型需要训练的有三个自适应的参数，然后还有decoder的参数
# 该模块是提前对模型某些参数进行预训练的，这里是对decoder进行预训练
from util import im_show,im2tensor,tensor2im
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
from MultiAdaption import Decoder2,get_encoder
from train import IMAGE_PATH,MAX_IMAGE_SIZE
from torchvision import transforms, datasets
ENCODER_PATH = './models/encoder.pkl'

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets,models
class Decoder_Trainer(nn.Module):
    def __init__(self,encoder,decoder=None):
        super(Decoder_Trainer,self).__init__()
        self.module_encoder = encoder
        self.module_decoder = decoder or Decoder2(512,3)

    def forward(self,x):
        # print('mean:{:.4f} max:{:.4f} min:{:.4f}'.format(x.mean().item(),x.max().item(),x.min().item()))
        x = self.module_encoder(x)
        # print('mean:{:.4f} max:{:.4f} min:{:.4f}'.format(x.mean().item(),x.max().item(),x.min().item()))
        x = self.module_decoder(x)
        # print('mean:{:.4f} max:{:.4f} min:{:.4f}'.format(x.mean().item(),x.max().item(),x.min().item()))
        return x
# 获取数据集，加载图像
def get_data_loader(batch_size=2):
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(MAX_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    hymenoptera_dataset = datasets.ImageFolder(root=IMAGE_PATH,
                                               transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                                 batch_size=batch_size, shuffle=True)
    return dataset_loader

# 训练decoder，使之能够根据feature map 复原图像，之后再风格迁移中再去微调
def train_decoder(encoder,decoder,num_epoch=100,learning_rate=0.00001):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder_Trainer(encoder,decoder).to(device)
    optimizer = optim.Adam(model.module_decoder.parameters(),learning_rate)
    loss_fun = nn.MSELoss()
    dataset_loader = get_data_loader(batch_size=1)
    model.train()
    for epoch in range(num_epoch):
        for x,_ in dataset_loader:
            x = x.to(device)
            y = model(x)
            loss = loss_fun(x,y)
            loss.backward()
            optimizer.step()
            print('train_epoch:{} loss:{:.4f}'.format(epoch, loss.item()))
    return model
def train_decoder_continue(model,num_epoch=100,learning_rate=0.0001):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.module_decoder.parameters(),learning_rate)
    loss_fun = nn.MSELoss()
    dataset_loader = get_data_loader(batch_size=1)
    model.train()
    for epoch in range(num_epoch):
        for x,_ in dataset_loader:
            x = x.to(device)
            y = model(x)
            loss = loss_fun(x,y)
            loss.backward()
            optimizer.step()
            print('train_epoch:{} loss:{:.4f}'.format(epoch, loss.item()))
    return model
# 查看重建图像的效果
def test_decoder(model,img_name):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    tensor = transform(Image.open(IMAGE_PATH+img_name))
    im_show(tensor2im(tensor,un_norm=True))
    tensor_recons = model(tensor.unsqueeze(0).to('cuda'))
    im_show(tensor2im(tensor_recons.detach(),un_norm=True))
def load_encoder():
    encoder = get_encoder(pretrained=False)
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    return encoder
if __name__ == '__main__':
    encoder = load_encoder()
    train_decoder(encoder,Decoder2(),num_epoch=10,learning_rate=0.0001)
