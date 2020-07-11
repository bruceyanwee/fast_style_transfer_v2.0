import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import sys
MAX_IMAGE_SIZE = 256
IMAGE_PATH = '/Users/yanwei/dl2020/pytorch-AdaIN/input'
MODEL_SAVE_PATH = './model.pkl'
DECODER_PATH = 'drive/My Drive/decoder.pkl'
lambda_c,lambda_s,lambda_i,lambda_dis_c,lambda_dis_s = 1,5,50,1,1
# 图片读取
def im_read(path,name):
    """
    :param path: folder_path
    :param name: picture name
    :return img: numpy.array [H,W,C] RGB
    """
    img = cv2.imread(path+name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 显示图片
def im_show(img):
    """
    :param img: numpy.array
    :return: None
    """
    plt.figure(figsize=(5, 5))
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

# 剪裁图片
def im_crop(x,new_size):
    """
    :param x:numpy.array [H,W,C]
    :param new_size: int 256
    :return out:numpy.array [new_size,new_size,C]
    """
    t_crop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop([new_size, new_size]),
        transforms.ToTensor(),
    ])
    out = t_crop(x).cpu().numpy().transpose(1, 2, 0)
    return out

# 把一张图片转换成pytorch x输入的格式 [B,C,H,W]
def im2tensor(img):
    """
    :param img: numpy.array [H,W,C]
    :return:tensor: [B,C,H,W]
    """
    t = transforms.ToTensor()
    tensor = torch.unsqueeze(t(img),dim=0)
    return tensor

# 把 pytorch中x的x的格式转换成im，供可视化
def tensor2im(tensor,un_norm=False):
    """
    :param tensor: tensor [B,C,H,W]
    :return: img: numpy.array [H,W,C]
    """
    tensor = tensor.squeeze(0)
    if un_norm == True:
        unorm = UnNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img = unorm(tensor.cpu()).clamp(0,1).numpy()
    else:
        img = tensor.cpu().numpy()
    img = img.transpose(1, 2, 0)
    return img
# 为了可视化结果，需要 inverse normalize
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# 对 vgg 提取的特征进行白化操作，这个是方法一：用的是SVD的方式，选取前K，
# 注意：这里 [B,C,H,W]，输入的B只能处理 B=1
def whiten_WCT(cF):
    """
    对 cov_m进行svd分解，取前k个分量
    :param cF: tensor [B,C,H,W]
    :return whiten_cF: tensor [B,C,H,W]
    """
    # cF = cF.double()
    B, C, H, W = cF.shape
    whiten_cF = cF.clone()
    for i in range(B):
        cF_i = cF[i].view(C, H * W)
        cFSize = cF_i.size()
        c_mean = torch.mean(cF_i, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF_i)
        cF_i = cF_i - c_mean
        # 计算协方差
        contentConv = torch.mm(cF_i, cF_i.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).to('cuda')
        try:
            c_u, c_e, c_v = torch.svd(contentConv, some=False)
        except:
            print('max_value:',contentConv.max().item(),'min_value:',contentConv.min().item())
            print("Unexpected error:", sys.exc_info()[0])
            c_u, c_e, c_v = torch.svd(contentConv+ 1e-4*contentConv.mean()*torch.rand(contentConv.shape).to('cuda'), some=False)

        # 取前K个分量
        k_c = cFSize[0]
        for j in range(cFSize[0]):
            if c_e[j] < 0.00001:
                k_c = j
                break
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF[i] = torch.mm(step2, cF_i).view(C, H, W)
    return whiten_cF.view(B, C, H, W)

# 对 vgg 提取的特征进行 白化操作，这个是方法一：用的特征值分解（正交分解） WCT论文中的方式
# 注意，这里修改了之后，可以批量处理了
def whiten(f_c):
    """
    对协方差矩阵（对称），用特征分解进行白化
    :param f_c: tensor [B,C,H,W]
    :return out: tnesor [B,C,H,W]
    """
    f_c_whiten = f_c.clone()
    B, C, H, W = f_c.shape
    for i in range(B):
        x = f_c[i].view(C, H*W)
        # 计算协方差
        cov_m = cal_cov(x)
        # 对 cov_m 正交化
        w, v = torch.symeig(cov_m, eigenvectors=True)
        U = torch.mm(torch.mm(v, torch.diag(torch.sqrt(1. / w))), v.t())
        f_c_whiten[i] = torch.mm(U, x).view(C, H, W)
    return f_c_whiten.view(B, C, H, W)

# loss function
# 注意输入的x的格式是 [4,C=3,H,W] [pic_1,pic_2,pic_3,pic_4] 1,2属于自然图像， 3，4属于风格图像
# ca模块需要组合(1,3),(2,3),(1,4),(2,4)

# 1.感知损失
def loss_perceptual(features_x,features_Ics):
    """
    pic_1+pic_3 组合 （不用算平均）
    pic_1+pic_3 组合 + pic_2+pic_4 组合 pic_2+pic_3 组合 + pic_1+pic_4 组合
    :param features_x:  dict 原始 x 经过vgg后，输出的features 包括[conv1_1 -> conv4_1]
    :param features_Ics: dict 融合后的image 经过 vgg 后，输出的features 包括[conv1_1 -> conv4_1]
    :return: loss : scalar
    """
    # loss_p_content 只需要conv4_1
    mse_loss = nn.MSELoss()
    feat_x_4_1  = features_x['conv4_1']    # [4,C,H,W] 1,2,3,4图像
    feat_Ics_4_1 = features_Ics['conv4_1'] # [4,C,H,W] 13,23,14,24
    loss_p_c = ( mse_loss(feat_Ics_4_1[0], feat_x_4_1[0])+  # 13 ~ 1
                 mse_loss(feat_Ics_4_1[2], feat_x_4_1[0])+  # 14 ~ 1
                 mse_loss(feat_Ics_4_1[1], feat_x_4_1[1])+  # 23 ~ 2
                 mse_loss(feat_Ics_4_1[3], feat_x_4_1[1])   # 24 ~ 2
                ) / 4

    # loss_p_style 只需要conv1_1 -- conv4_1
    layers = ['conv1_1','conv2_1','conv3_1','conv4_1']
    loss_p_s = 0.
    for l in layers:
        B, C, H, W = features_x[l].shape
        loss_p_s += (mse_loss(features_Ics[l][0].view(C, -1).mean(1), features_x[l][2].view(C, -1).mean(1)) +
                     mse_loss(features_Ics[l][0].view(C, -1).std(1), features_x[l][2].view(C, -1).std(1))) # 13 ~ 3
        loss_p_s += (mse_loss(features_Ics[l][2].view(C, -1).mean(1), features_x[l][3].view(C, -1).mean(1)) +
                     mse_loss(features_Ics[l][2].view(C, -1).std(1), features_x[l][3].view(C, -1).std(1))) # 14 ~ 4
        loss_p_s += (mse_loss(features_Ics[l][1].view(C, -1).mean(1), features_x[l][2].view(C, -1).mean(1)) +
                     mse_loss(features_Ics[l][1].view(C, -1).std(1), features_x[l][2].view(C, -1).std(1))) # 23 ~ 3
        loss_p_s += (mse_loss(features_Ics[l][3].view(C, -1).mean(1), features_x[l][3].view(C, -1).mean(1)) +
                     mse_loss(features_Ics[l][3].view(C, -1).std(1), features_x[l][3].view(C, -1).std(1))) # 24 ~ 4
    loss_p_s = loss_p_s / 4
    return lambda_c * loss_p_c  + lambda_s * loss_p_s

# 2.正定损失:用同样一张 自然风景图同时作为 content+style 输入，然后减去content
# ca模块需要组合(1,1),(3,3)
# Net(内容图，风格图)-return 融合图
# loss = ||Net(Ic,Ic)-Ic||2 + ||Net(Is,Is)-Is||2
def loss_identity(Icc,Ic,Iss,Is): #
    """
    :param Icc: tensor pic_1 作为 content和style 同时输入 I11
    :param Ic: tensor pic_1
    :param Iss: tensor pic_3 作为 content和style 同时输入 I33
    :param Is: tensor pic_3
    :return: scalar
    """
    mes_loss =nn.MSELoss()
    loss_i = mes_loss(Icc,Ic)+mes_loss(Iss,Is)
    return lambda_i * loss_i

# 3.分离损失
# ca模块需要组合(1,3),(1,4),(2,3)
# 核心思想是：同一张content，不同的style，得到的新的图，他们的内容是相近的 (1,3) ~content feature~ (1,4)
# 同一张style，不同的content，得到的新的图，他们的风格是相近的 (1,3) ~style feature~ (2,3)
def loss_disentanglement(features_x,features_Ics):
    """
    :param features_x:  dict 原始 x 经过vgg后，输出的features 包括[conv1_1 -> conv4_1]
    :param features_Ics: dict 融合后的image 经过 vgg 后，输出的features 包括[conv1_1 -> conv4_1]
    :return: scalar
    """
    # loss_dis_content 只需要conv4_1
    mse_loss = nn.MSELoss()
    feat_x_4_1 = features_x['conv4_1']  # [4,C,H,W] 1,2,3,4图像
    feat_Ics_4_1 = features_Ics['conv4_1']  # [4,C,H,W] 13,23,14,24
    loss_dis_c = (mse_loss(feat_Ics_4_1[0], feat_x_4_1[2]) +  # 13 ~ 14
                mse_loss(feat_Ics_4_1[1], feat_x_4_1[3])  # 23 ~ 24
                ) / 2

    # loss_dis_style 需要conv1_1 -- conv4_1
    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
    loss_dis_s = 0.
    for l in layers:
        B, C, H, W = features_x[l].shape # [4,C,H,W] 13,23,14,24
        loss_dis_s += (mse_loss(features_Ics[l][0].view(C, -1).mean(1), features_x[l][1].view(C, -1).mean(1)) +
                     mse_loss(features_Ics[l][0].view(C, -1).std(1), features_x[l][1].view(C, -1).std(1)))   # 13 ~ 23
        loss_dis_s += (mse_loss(features_Ics[l][2].view(C, -1).mean(1), features_x[l][3].view(C, -1).mean(1)) +
                     mse_loss(features_Ics[l][2].view(C, -1).std(1), features_x[l][3].view(C, -1).std(1)))   # 14 ~ 24
    loss_dis_s = loss_dis_s / 2
    return lambda_dis_c * loss_dis_c  + lambda_dis_s * loss_dis_s

if __name__ == '__main__':
    img = im_read(IMAGE_PATH,'in1.jpg')
    img_crop = im_crop(img,MAX_IMAGE_SIZE)
    im_show(img_crop)
    x = im2tensor(img_crop)
    x1 = whiten_WCT(x)
    x2 = whiten(x)
    im_show(tensor2im(x1))
    im_show(tensor2im(x2))
