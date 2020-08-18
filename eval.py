import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import sys
import numpy as np
# loss weights 
# 显示图片
def im_show(img):
    """
    :param img: numpy.array
    :return: None
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.show()

# 计算 feature map 的feat之间的相关性，channel-wise vector[每个位置上特征A的强度]
def cal_cov(m, y=None):
    """
    :param m: tensor [C,H*W]
    :param y:
    :return: tensor [C,C]
    """
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_mean = torch.mean(m, dim=1)
    x = m - m_mean[:, None]
    m_cov = torch.mm(x, x.t()).div(m.size(1) - 1)
    return m_cov

# 显示迁移效果的函数
def show_tensor_single(tensor_model_out,using_unorm=True):   
    img = denorm(tensor_model_out)        
    im_show(img)

# mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225]
# 另外一种denorm

def plot_loss(loss_rcd_list,label_list):
    type_n = len(loss_rcd_list)
    epoch_n = len(loss_rcd_list[0])
    for i in range(type_n):
        plt.plot(np.arange(epoch_n),loss_rcd_list[i],label=label_list[i])
    plt.title('loss record')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss_value /per')
    plt.show()
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
# SANet 的做法，其实和 whiten的目的一样，都是为了去掉 纹理信息（每个channel的 mean、std）
def normalization(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat
def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 3)
    C = size[0]
    feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)
    return feat_mean
def calc_std(feat):
    size = feat.size()
    assert (len(size) == 3)
    C = size[0]
    feat_std = feat.view(C, -1).std(dim=1).view(C, 1, 1)    
    return  feat_std

import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models,transforms
import torch.optim as optim
import torch.nn.functional as F
# 整个网络
class SANet(nn.Module):
    def __init__(self,encoder,decoder=None):
        super(SANet,self).__init__()
        self.module_encoder = encoder     
        self.module_fusion = Fusion(512) 
        self.module_decoder = decoder or Decoder2(512,3)
        self.name_layer_map = {      
                '3' : 'relu1_1',
                '10': 'relu2_1',
                '17': 'relu3_1',
                '30': 'relu4_1',
                '43': 'relu5_1',
        }
        self.layers = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1']
    def forward(self,c,s):
        # step-1 encoder using vgg19.feature[:21]        
        feat_c = self.feature_extraction(c)
        feat_s = self.feature_extraction(s)         
        # attention 进行融合
        feat_cs = self.module_fusion(feat_c['relu4_1'],feat_s['relu4_1'],feat_c['relu5_1'],feat_s['relu5_1'])
        feat_cc = self.module_fusion(feat_c['relu4_1'],feat_c['relu4_1'],feat_c['relu5_1'],feat_c['relu5_1'])
        feat_ss = self.module_fusion(feat_s['relu4_1'],feat_s['relu4_1'],feat_s['relu5_1'],feat_s['relu5_1'])
        # deocoder 图像重建
        Ics = self.module_decoder(feat_cs)
        Icc = self.module_decoder(feat_cc)
        Iss = self.module_decoder(feat_ss)   
        # 再次特征提取
        feat_cs2 = self.feature_extraction(Ics) 
        feat_cc2 = self.feature_extraction(Icc)
        feat_ss2 = self.feature_extraction(Iss)
        # 计算损失
        loss_c_4_1 = self.calc_content_loss(feat_c['relu4_1'],feat_cs2['relu4_1'],norm = False)
        loss_c_5_1 = self.calc_content_loss(feat_c['relu5_1'],feat_cs2['relu5_1'],norm = False)
        loss_c = lambda_c_weights[0]*loss_c_4_1+lambda_c_weights[1]*loss_c_5_1
        loss_s,loss_s_per_layer = self.calc_style_loss(feat_s,feat_cs2)
        loss_i1,loss_i2 = self.calc_identity_loss(Icc,c,Iss,s, feat_cc2,feat_c,feat_ss2,feat_s)
        # 为了方便调参，需要返回，content loss ，style 4层的loss ，两种identity的损失
        return loss_c,loss_c_4_1,loss_c_5_1,loss_s,loss_s_per_layer,loss_i1,loss_i2 
    # 计算损失相关函数
    def calc_mean_std(self,feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return F.mse_loss(input, target)
        else:
          return F.mse_loss(normalization(input), normalization(target))
    def calc_style_loss(self,out_feat, style_middle_features):        
        loss_s = 0.
        loss_s_per_layer = []
        for l,w in zip(self.layers,lambda_s_weights):
            c_mean, c_std = self.calc_mean_std(out_feat[l])
            s_mean, s_std = self.calc_mean_std(style_middle_features[l])
            loss_layer = F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
            loss_s_per_layer.append(loss_layer.item())
            loss_s += w*loss_layer
        return loss_s, loss_s_per_layer
    def calc_identity_loss(self,Icc,Ic,Iss,Is,feat_cc,feat_c,feat_ss,feat_s):      
        layers = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1']
        loss_i1 = F.mse_loss(Icc,Ic) + F.mse_loss(Iss,Is)
        loss_i2 = 0.
        for l in layers:
            loss_i2 += (F.mse_loss(feat_cc[l],feat_c[l]) + F.mse_loss(feat_ss[l],feat_s[l]))
        return loss_i1, loss_i2
    # 用预训练vgg 提取conv1_1~conv4_1,供 计算loss使用
    def feature_extraction(self,x): 
        features = {}
        for name, layer in self.module_encoder._modules.items():
            x = layer(x)
            if name in self.name_layer_map:
                if (name == '3'):     # conv1_1 64x256x256
                    features[self.name_layer_map[name]] = x
                elif (name == '10'):   # conv2_1 128x128x128
                    features[self.name_layer_map[name]] = x
                elif (name == '17'):  # conv3_1 256x64x64
                    features[self.name_layer_map[name]] = x
                elif (name == '30'):  # conv4_1 512x32x32
                    features[self.name_layer_map[name]] = x
                elif (name == '43'):  # conv5_1 512x32x32
                    features[self.name_layer_map[name]] = x
                    break  # 只是用到了vgg encoder的前19层，Terminate forward pass
        return features
    # test阶段用于合成图片
    def transfer(self,tensor_c,tensor_s): # 两张图片一起传进来，一张 c,一张 s ,tensor的大小是：[1,C,H,W]
        x_c,x_s = tensor_c,tensor_s        
        for name, layer in self.module_encoder._modules.items():
            x_c = layer(x_c)
            x_s = layer(x_s)
            if name == '30':
                x_c_4_1 = x_c
                x_s_4_1 = x_s
            elif name == '43':
                x_c_5_1 = x_c
                x_s_5_1 = x_s
                break         
        x_fcs = self.module_fusion(x_c_4_1, x_s_4_1,x_c_5_1,x_s_5_1) # 注意1，2，是c s ,这里style在前面        
        # step-3 Decoder   
        I_cs = self.module_decoder(x_fcs)
        return I_cs

# 获取vgg19 作为预训练模型
def get_encoder():
    vgg_normalized = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
        )
    vgg_normalized.load_state_dict(torch.load(args.vgg))
    for param in vgg_normalized.parameters():
        param.requires_grad = False
    return vgg_normalized
# vgg19 前19层的镜像解码器
# 上采样 + 正卷积
class Decoder2(nn.Module):
    def __init__(self, in_channel=512, out_channel=3):
        super(Decoder2,self).__init__()
        self.up_sample = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channel, 256, (3, 3)),            
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'), 
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),            
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),            
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),            
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),        
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),          
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),            
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, out_channel, (3, 3)),
        )
    def forward(self,x):
        x = self.up_sample(x)
        return x

"""### 自适应融合模块"""

# 多适应模块，作用为分离content和style feature 然后 fusion
# 论文中 SA和CA两模块的的代码
# style_self_Adaption
# 多适应模块，作用为分离content和style feature 然后 fusion
# 论文中 SA和CA两模块的的代码
# 这里与论文中一致，用到的是conv4_1的feature map 进行融合
# co_Adaption_fusion
# SAnet的做法是，直接对前面的content和style进行normalization，而本文是进行了自适应，并且用dis_loss去学习怎么分离
# F = self.f(mean_variance_norm(content))
# G = self.g(mean_variance_norm(style))
class SA_fusion(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(SA_fusion,self).__init__()
        self.conv_f1 = nn.Conv2d(in_channel,out_channel,(1,1))
        self.conv_f2 = nn.Conv2d(in_channel,out_channel,(1,1))
        self.conv_f3 = nn.Conv2d(in_channel,out_channel,(1,1))
        self.conv_frs = nn.Conv2d(in_channel,out_channel,(1,1))
        self.softmax = nn.Softmax(dim = -1)        
    def forward(self,x_fcc,x_fss):
        Bc,Cc,Hc,Wc = x_fcc.shape
        Bs,Cs,Hs,Ws = x_fss.shape
        x_fcc1 = self.conv_f1(normalization(x_fcc))  # [B,C,H,W]
        x_fss2 = self.conv_f2(normalization(x_fss))  # [B,C,H,W]
        x_fss3 = self.conv_f3(x_fss).view(Bc,Cs,Hs*Ws)  # shape [Cs,Ns]
        # position_wise Multiplication
        Acs = self.softmax(torch.bmm(x_fcc1.view(Bc,Cc,Hc*Wc).permute(0, 2, 1),x_fss2.view(Bs,Cs,Hs*Ws))) # softmax 按行计算的 [Nc,Ns]        
        x_frs= torch.bmm(x_fss3,Acs.permute(0, 2, 1)) # Acs的每一行代表相关性 [Cs,Ns] x [Ns,Nc] = [Cs,Nc]
        x_frs = self.conv_frs(x_frs.view(x_fcc.shape))
        x_fcs = torch.add(x_frs,x_fcc)
        return x_fcs
class Fusion(nn.Module):
    def __init__(self,in_channel = 512):
        super(Fusion,self).__init__()
        self.fusion_4_1 = SA_fusion(in_channel,in_channel)
        self.fusion_5_1 = SA_fusion(in_channel,in_channel)
        self.up_sample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_channel, in_channel, (3, 3))        
    def forward(self,content_4_1,style_4_1,content_5_1,style_5_1):
        return self.merge_conv(self.merge_conv_pad(self.fusion_4_1(content_4_1,style_4_1) 
                                + self.up_sample5_1(self.fusion_5_1(content_5_1,style_5_1))))

import numpy as np
from torch.utils import data

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(666)
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from tqdm import tqdm

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def test_transform():    
    return transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomResizedCrop(MAX_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='./mydata/content',
                    help='Directory path to a test of content images')
parser.add_argument('--style_dir', type=str, default='./mydata/style',
                    help='Directory path to a test of style images')
parser.add_argument('--model', type=str, default='./SANet_818.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--start_iter', type=float, default=0)
args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# decoder = Decoder2()
vgg = get_encoder()
network = SANet(vgg)
network.eval()
# torch.save(network.state_dict(),'SANet_818.pth')
network.load_state_dict(torch.load(args.model)
network.to(device)

content_tf = test_transform()
style_tf = test_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

image_pair = [('1.jpg','10.jpg'),
('2.jpg','1.png'),
('12.jpg','12.jpg'),
('14.jpg','14.jpg'),
('7.jpg', '6.jpg'),
('16.jpg', '16.jpg'),
('5.jpg', '4.jpg'),
('11.jpg', '11.jpg'),
('3.jpg', '2.jpg'),
('15.jpg', '15.jpg'),
('13.jpg','13.jpg'),
('10.jpg', '8.jpg'),
('17.jpg', '17.jpg'),
('6.jpg', '5.jpg'),
('6.jpg','19.jpg'),
('8.jpg', '19.jpg'),
('8.jpg', '7.jpg'),
('8.jpg','9.jpg'),
('8.jpg','20.jpg'),
('18.jpg', '18.jpg'),
('3.jpg', '19.jpg'),
('4.jpg', '3.jpg'),
('9.jpg', '9.jpg'),
('3.jpg', '3.jpg'),
('10.jpg','12.jpg'),
('2.jpg','16.jpg'),
('9.jpg','10.jpg')]    
def test_transfer(model,cimg_name,simg_name):
    with torch.no_grad():
        transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        tensor_c = transform(Image.open(args.content_dir+cimg_name).convert('RGB'))
        tensor_s = transform(Image.open(args.style_dir+simg_path).convert('RGB'))
        img_trans = model.transfer(tensor_c = tensor_c.unsqueeze(0).to('cuda'),tensor_s = tensor_s.unsqueeze(0).to('cuda'))  
        img = denorm(img_trans.detach())
        # im_show(img)
        output_name = '{:s}/{:s}_+_{:s}'.format(
                    args.output,cimg_name[1:],simg_name[1:]                  
                )
        save_image(img, output_name)
def denorm(tensor):
    tensor = tensor.detach().squeeze(0).cpu()
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    res = torch.clamp(tensor * std + mean, 0, 1).permute(1, 2, 0)
    return res

for c,s in image_pair:
    # print(c,s)
    test_transfer(network,cimg_name='/in'+c,simg_name='/style'+s)    
    













