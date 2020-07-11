import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models,transforms
import torch.optim as optim
# 整个网络
class MultiAdaptionNet(nn.Module):
    def __init__(self,encoder,decoder=None):
        super(MultiAdaptionNet,self).__init__()
        self.module_encoder = encoder
        self.module_adaption = nn.ModuleDict({
            'adaption_content_conv4_1' : SA_content(512, 512),
            'adaption_style_conv4_1' : SA_style(512, 512),
            'adaption_fusion_conv4_1'   : Co_Adaption(512, 512),
            'adaption_content_conv3_1' : SA_content(256, 256),
            'adaption_style_conv3_1' : SA_style(256, 256),
            'adaption_fusion_conv3_1'   : Co_Adaption(256, 256),
            'adaption_content_conv2_1' : SA_content(128, 128),
            'adaption_style_conv2_1' : SA_style(128, 128),
            'adaption_fusion_conv2_1'   : Co_Adaption(128, 128),
            'adaption_content_conv1_1' : SA_content(64, 64),
            'adaption_style_conv1_1' : SA_style(64, 64),
            'adaption_fusion_conv1_1'   : Co_Adaption(64, 64)
        })
        self.module_decoder = decoder or Decoder2(512,3)
        self.layers = {             # extract layers
                '0' : 'conv1_1',
                '5' : 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
        }
        self.concat_layers = {
                '3' : 'conv3_1',
                '16': 'conv2_1',
                '23': 'conv1_1'
        }
    def forward(self,x,cal_loss=False):
        # step-1 vgg feature extraction
        features = {}
        for name, layer in self.module_encoder._modules.items():
            x = layer(x)
            if name in self.layers:
                if (name == '0'):  # conv1_1 64x256x256
                    features[self.layers[name]] = x
                elif (name == '5'):  # conv2_1 128x128x128
                    features[self.layers[name]] = x
                elif (name == '10'):  # conv3_1 256x64x64
                    features[self.layers[name]] = x
                elif (name == '19'):  # conv4_1 512x32x32
                    features[self.layers[name]] = x
                    break  # 只是用到了vgg encoder的前19层，Terminate forward pass
        if cal_loss==True: # when cal_loss,只需要计算到这里,相当于损失函数中的 'fi'
            return features
        # step-2 Multi-Adaption 注意这里是对4层都需要进行融合
        layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
        fusion_feature = {}
        for l in layers:
            x_layer = features[l]
            # step-2-1 Style-Self-Adaption
            x_fss = self.module_adaption['adaption_style_'+l](x_layer)  # now x denote conv4_1 ~ features['conv4_1'] [4,C,H,W] 2自然+2风格
            # step-2-2 Content-Self-Adaption
            x_fcc = self.module_adaption['adaption_content_'+l](x_layer)  # now x denote conv4_1 ~ features['conv4_1'] [4,C,H,W]
            # step-2-3 Co-Adaption 这一步比较关键，因为涉及到计算 identity_loss 和 disentanglement_loss,需要几种不同的组合
            # 而且 module_ca 处理的[B,C,H,W]中B=1，因为whiten操作是没有实现批量化 whiten
            # 根据损失函数确定组合方式 (1,1) (1,3) (1,4) (2,3) (3,3),(2,4)
            # 感知loss (1,3),(2,3);
            x_fcs13 = self.module_adaption['adaption_fusion_'+l](x_fss[2].unsqueeze(0), x_fcc[0].unsqueeze(0)) # 注意1，2，3，4 是c c s s ,这里style在前面
            x_fcs23 = self.module_adaption['adaption_fusion_'+l](x_fss[2].unsqueeze(0), x_fcc[1].unsqueeze(0)) # x[0].unsqueeze(0) 等价于 x[0:1]
            x_fcs24 = self.module_adaption['adaption_fusion_'+l](x_fss[3].unsqueeze(0), x_fcc[1].unsqueeze(0))
            # 分离损失loss (1,3),(1,4),(2,3)
            x_fcs14 = self.module_adaption['adaption_fusion_'+l](x_fss[3].unsqueeze(0),x_fcc[0].unsqueeze(0))
            # 正定损失loss (1,1),(3,3);
            x_fcs11 = self.module_adaption['adaption_fusion_'+l](x_fss[0].unsqueeze(0), x_fcc[0].unsqueeze(0))
            x_fcs33 = self.module_adaption['adaption_fusion_'+l](x_fss[2].unsqueeze(0), x_fcc[2].unsqueeze(0))
            x_fcs = torch.cat((x_fcs13,x_fcs23,x_fcs14,x_fcs24,x_fcs11,x_fcs33),0)
            fusion_feature[l] = x_fcs
        # step-3 Decoder   输出6张图像 x_pix [6,C,H,W]
        # 同样这里也不同了，需要在一边重建，一边进行融合,  *** 记得concat后，需要更改concat后面的conv2d，把channel减半
        x_fusion = fusion_feature['conv4_1']
        for name, layer in self.module_decoder.up_sample._modules.items():
            x_fusion = layer(x_fusion)
            if name == '3': # concat fusioned_conv3_1
                x_fusion = torch.cat((x_fusion,fusion_feature[self.concat_layers[name]]), 1)
            elif name == '16': # concat fusioned_conv2_1
                x_fusion = torch.cat((x_fusion, fusion_feature[self.concat_layers[name]]), 1)
            elif name == '23': # concat fusioned_conv1_1
                x_fusion = torch.cat((x_fusion, fusion_feature[self.concat_layers[name]]), 1)
        return x_fusion,features

    def transfer(self,tensor_c,tensor_s,using_Whiten = True): # 两张图片一起传进来，一张 c,一张 s ,tensor的大小是：[1,C,H,W]
        x_c,x_s = tensor_c,tensor_s
        for name, layer in self.module_encoder._modules.items():
            x_c = layer(x_c)
            x_s = layer(x_s)
            if name == '19':
                break # 只是用到了vgg encoder的前19层，Terminate forward pass
        # step-2 Multi-Adaption
        # step-2-1 Style-Self-Adaption
        x_fss = self.module_sa_s(x_s)  # now x denote conv4_1 ~ features['conv4_1'] [1,C,H,W] +1风格
        # step-2-2 Content-Self-Adaption
        x_fcc = self.module_sa_c(x_c,using_W = using_Whiten)  # now x denote conv4_1 ~ features['conv4_1'] [1,C,H,W] +1自然
        # step-2-3 Co-Adaption
        x_fcs = self.module_ca(x_fss, x_fcc,using_W = using_Whiten) # 注意1，2，是c s ,这里style在前面
        # step-3 Decoder
        x_pix = self.module_decoder(x_fcs)
        return x_pix

# 获取vgg19 作为预训练模型
def get_encoder(pretrained=False):
    vgg = models.vgg19(pretrained=pretrained)
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg.features[:20]

# vgg19 前19层的镜像解码器
# 上采样 + 正卷积
class Decoder(nn.Module):
    def __init__(self, in_channel=512, out_channel=3):
        super(Decoder,self).__init__()
        self.deconv = nn.Sequential(                      # vgg19 前19层机构
            nn.ConvTranspose2d(in_channel, 256, 3, 1,1),  #(19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ConvTranspose2d(256, 256, 2, 2),           #(18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

            nn.ReLU(),                                    #(17): ReLU(inplace=True)
            nn.ConvTranspose2d(256, 256, 3, 1,1),         #(16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ReLU(),                                    #(15): ReLU(inplace=True)
            nn.ConvTranspose2d(256, 256, 3, 1,1),         #(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ReLU(),                                    #(13): ReLU(inplace=True)
            nn.ConvTranspose2d(256, 256, 3, 1,1),         #(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ReLU(),                                    #(11): ReLU(inplace=True)
            nn.ConvTranspose2d(256, 128, 3, 1,1),         #(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ConvTranspose2d(128, 128, 2, 2),           #(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

            nn.ReLU(),                                    #(8): ReLU(inplace=True)
            nn.ConvTranspose2d(128, 128, 3,1,1),          #(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ConvTranspose2d(128, 64, 3, 1,1),          #(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ConvTranspose2d(64, 64, 2, 2),             #(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

            nn.ReLU(),                                    #(3): ReLU(inplace=True)
            nn.ConvTranspose2d(64, 64, 3, 1,1),           #(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            nn.ReLU(),                                    #(1): ReLU(inplace=True)
            nn.ConvTranspose2d(64, 3, 3, 1,1),            #(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    def forward(self,x):
        x = self.deconv(x)
        return x

class Decoder2(nn.Module):
    def __init__(self, in_channel=512, out_channel=3):
        super(Decoder2,self).__init__()
        self.up_sample = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channel, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),      # 这里channel减半了，因为进行了concat
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
            nn.Conv2d(256, 128, (3, 3)),     # 这里channel减半了，因为进行了concat
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),       # 这里channel减半了，因为进行了concat
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, out_channel, (3, 3)),
        )
    def forward(self,x):
        x = self.up_sample(x)
        return x
# 减少一些重复channel的conv
class Decoder_light(nn.Module):
    def __init__(self, in_channel=512, out_channel=3):
        super(Decoder_light,self).__init__()
        self.up_sample = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channel, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'), #
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
# 层数更少
class Decoder_light2(nn.Module):
    def __init__(self, in_channel=512, out_channel=3):
        super(Decoder_light2,self).__init__()
        self.up_sample = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channel, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'), #
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, out_channel, (3, 3)),
        )
    def forward(self,x):
        x = self.up_sample(x)
        return x

# 多适应模块，作用为分离content和style feature 然后 fusion
# 论文中 SA和CA两模块的的代码
# style_self_Adaption
# 多适应模块，作用为分离content和style feature 然后 fusion
# 论文中 SA和CA两模块的的代码
# style_self_Adaption
class SA_style(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SA_style, self).__init__()
        self.conv_f1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv_f2 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv_f3 = nn.Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x): # 这里需要改成 四张图片一起传入，之前是一张图
        B, C, H, W = x.shape
        x_f1 = self.conv_f1(x)
        x_f2 = self.conv_f2(x)
        x_f3 = self.conv_f3(x)
        # channel-wise Multiplication
        x_fcc = x.clone()
        for i in range(B):
            As_i = torch.mm(x_f1[i].view(C, H * W), x_f2[i].view(C, H * W).t()).softmax(dim=1)  # softmax 按行计算的
            x_fcc[i] = torch.mm(As_i, x_f3[i].view(C, H * W)).view(x[i].shape)  # As_i的每一行代表相关性
            x_fcc[i] = torch.add(x_fcc[i], x[i])
        return x_fcc

# content_self_Adaption
class SA_content(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(SA_content,self).__init__()
        self.conv_f1 = nn.Conv2d(in_channel,out_channel,1,1)
        self.conv_f2 = nn.Conv2d(in_channel,out_channel,1,1)
        self.conv_f3 = nn.Conv2d(in_channel,out_channel,1,1)

    def forward(self, x, using_W = False):
        """
        :param x: tensor [B,C,H,W]
        :return: tensor [B,C,H,W]
        """
        B,C,H,W = x.shape
        if not using_W:
            x_wtn = x
        else:
            x_wtn = whiten(x)
        x_f1 = self.conv_f1(x_wtn)
        x_f2 = self.conv_f2(x_wtn)
        x_f3 = self.conv_f3(x)
        x_fcc = x.clone()
        # position_wise Multiplication
        for i in range(B):
            Ac_i = torch.mm(x_f1[i].view(C,H*W).t(),x_f2[i].view(C,H*W)).softmax(dim=1) # softmax 按行计算的
            x_fcc[i] = torch.mm(x_f3[i].view(C, H * W),Ac_i.t()).view(x[i].shape) #Ac_i.t()的每一列代表相关性
            x_fcc[i] = torch.add(x_fcc[i],x[i])
        return x_fcc

# co_Adaption_fusion
class Co_Adaption(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Co_Adaption,self).__init__()
        self.conv_f1 = nn.Conv2d(in_channel,out_channel,1,1)
        self.conv_f2 = nn.Conv2d(in_channel,out_channel,1,1)
        self.conv_f3 = nn.Conv2d(in_channel,out_channel,1,1)

    def forward(self,x_fss,x_fcc,using_W = False):
        Bc,Cc,Hc,Wc = x_fcc.shape
        Bs,Cs,Hs,Ws = x_fss.shape
        if not using_W:
            x_fcc1 = self.conv_f1(x_fcc)
            x_fss2 = self.conv_f2(x_fss)
        else:
            x_fcc1 = self.conv_f1(whiten(x_fcc))  # [1,C,H,W]
            x_fss2 = self.conv_f2(whiten(x_fss))  # [1,C,H,W]
        x_fss3 = self.conv_f3(x_fss).view(Cs,Hs*Ws)
        # position_wise Multiplication
        Acs = torch.mm(x_fcc1.view(Cc,Hc*Wc).t(),x_fss2.view(Cs,Hs*Ws)).softmax(dim=1) # softmax 按行计算的 [Ns,Nc]
        x_frs= torch.mm(x_fss3,Acs.t()) # Acs的每一行代表相关性
        x_fsc = torch.add(x_frs.view(x_fcc.shape),x_fcc)
        return x_fsc

