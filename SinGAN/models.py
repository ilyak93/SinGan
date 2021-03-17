import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()

        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B * N * C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B * C * N
        energy = torch.bmm(proj_query, proj_key)  # batch matrix-matrix product

        attention = self.softmax(energy)  # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B * C * N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # batch matrix-matrix product
        out = out.view(m_batchsize, C, width, height)  # B * C * W * H

        # Add attention weights onto input
        out = self.gamma * out + x
        return out, attention


class MyConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(MyConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class MyWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(MyWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = Self_Attn( max(N, opt.min_nfc))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            x,_ = self.attn(x)
        x = self.tail(x)
        return x


class MyGeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(MyGeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = Self_Attn(max(N, opt.min_nfc))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            x,_ = self.attn(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y


        

from axial_attention import AxialAttention

class My2WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(My2WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = AxialAttention(
                dim = max(N, opt.min_nfc),               # embedding dimension
                dim_index = 1,         # where is the embedding dimension
                #dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads = 4,             # number of heads for multi-head attention
                num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            #x,_ = self.attn(x)
            x = self.attn(x)
        x = self.tail(x)
        return x


class My2GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(My2GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = self.attn = AxialAttention(
                dim = max(N, opt.min_nfc),               # embedding dimension
                dim_index = 1,         # where is the embedding dimension
                #dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads = 4,             # number of heads for multi-head attention
                num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            #x,_ = self.attn(x)
            x = self.attn(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        

class My21WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(My21WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = AxialAttention(
                dim = max(N, opt.min_nfc),               # embedding dimension
                dim_index = 1,         # where is the embedding dimension
                #dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads = 4,             # number of heads for multi-head attention
                num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            #x,_ = self.attn(x)
            x = x+self.attn(x)
        x = self.tail(x)
        return x


class My21GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(My21GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = self.attn = AxialAttention(
                dim = max(N, opt.min_nfc),               # embedding dimension
                dim_index = 1,         # where is the embedding dimension
                #dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads = 4,             # number of heads for multi-head attention
                num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            #x,_ = self.attn(x)
            x = x+self.attn(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        

class My22WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(My22WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.ffn = nn.Sequential(nn.Linear(max(N, opt.min_nfc), max(2 * N, opt.min_nfc), bias=True),
                                     nn.ReLU(),
                                     nn.Linear(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), bias=True))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = x + self.attn(x)
            x = x.permute([0, 2, 3, 1])
            tmp = self.ffn(x)
            x = (x + tmp).permute([0, 3, 1, 2])
        x = self.tail(x)
        return x


class My22GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(My22GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.ffn = nn.Sequential(nn.Linear(max(N, opt.min_nfc), max(2 * N, opt.min_nfc), bias=True),
                                     nn.ReLU(),
                                     nn.Linear(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), bias=True))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            # x,_ = self.attn(x)
            x = x + self.attn(x)
            x = x.permute([0,2,3,1])
            tmp = self.ffn(x)
            x = (x + tmp).permute([0,3,1,2])
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

        

class My23WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(My23WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.ffn = nn.Sequential(nn.Linear(max(N, opt.min_nfc), max(2 * N, opt.min_nfc), bias=True),
                                     nn.ReLU(),
                                     nn.Linear(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), bias=True))
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = x + self.attn(x)
            x = x.permute([0, 2, 3, 1])
            x = self.ffn(x).permute([0, 3, 1, 2])
        x = self.tail(x)
        return x


class My23GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(My23GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.attn = self.attn = AxialAttention(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True
                # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
            self.ffn = nn.Sequential(nn.Linear(max(N, opt.min_nfc), max(2 * N, opt.min_nfc), bias=True),
                                     nn.ReLU(),
                                     nn.Linear(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), bias=True))
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            # x,_ = self.attn(x)
            x = x + self.attn(x)
            x = x.permute([0,2,3,1])
            x = self.ffn(x).permute([0,3,1,2])
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        


class DecoderAxionalLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, dim, dim_index ,heads , num_dimensions, sum_axial_out):
        super().__init__()
        self.attn = AxialAttention(dim, dim_index ,heads , num_dimensions, sum_axial_out)
        self.layernorm_attn = nn.LayerNorm([dim], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([dim], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2*dim, dim, bias=True))

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        y = self.attn(X)
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(X)
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X

class My24WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(My24WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.image_transformer_layer = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.image_transformer_layer(x)
        x = self.tail(x)
        return x


class My24GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(My24GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        if opt.attn == True:
            self.image_transformer_layer = DecoderAxionalLayer(
                dim=max(N, opt.min_nfc),  # embedding dimension
                dim_index=1,  # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads=4,  # number of heads for multi-head attention
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self, 'attn'):
            x = self.image_transformer_layer(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
        

class ImageAttn(nn.Module):
    def __init__(self, in_dim, num_heads, block_length, attn_type='global'): #block length = number of columnes
        super().__init__()
        self.hidden_size = in_dim
        self.kd = in_dim // 2
        self.vd = in_dim
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.block_length = block_length
        self.q_dense = nn.Linear(self.hidden_size, self.kd, bias=False)
        self.k_dense = nn.Linear(self.hidden_size, self.kd, bias=False)
        self.v_dense = nn.Linear(self.hidden_size, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, self.hidden_size, bias=False)
        assert self.kd % self.num_heads == 0
        assert self.vd % self.num_heads == 0

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v

    def forward(self, X):
        X = X.permute([0, 2, 3, 1]).contiguous()
        orig_shape = X.shape
        #X = X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])  # Flatten channels into width
        X = X.view(X.shape[0], -1, X.shape[3])
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)
        # Split to shape [batch_size, num_heads, len, depth / num_heads]
        q = q.view(q.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.num_heads, self.vd // self.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.num_heads) ** (-0.5)

        if self.attn_type == "global":
            bias = -1e9 * torch.triu(torch.ones(X.shape[1], X.shape[1]), 1).to(X.device)
            result = self.dot_product_attention(q, k, v, bias=bias)
        elif self.attn_type == "local_1d":
            len = X.shape[1]
            blen = self.block_length
            pad = (0, 0, 0, (-len) % self.block_length) # Append to multiple of block length
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)

            bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
            first_output = self.dot_product_attention(
                q[:,:,:blen,:], k[:,:,:blen,:], v[:,:,:blen,:], bias=bias)

            if q.shape[2] > blen:
                q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3])
                k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3])
                v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3])
                local_k = torch.cat([k[:,:,:-1], k[:,:,1:]], 3) # [batch, nheads, (nblocks - 1), blen * 2, depth]
                local_v = torch.cat([v[:,:,:-1], v[:,:,1:]], 3)
                tail_q = q[:,:,1:]
                bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
                tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=bias)
                tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4])
                result = torch.cat([first_output, tail_output], 2)
                result = result[:,:,:X.shape[1],:]
            else:
                result = first_output[:,:,:X.shape[1],:]

        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        result = result.view(orig_shape[0],orig_shape[1], orig_shape[2] ,orig_shape[3])#.permute([0, 3, 1, 2])
        return result



class My31WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(My31WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
        if opt.attn == True:
            self.attn = ImageAttn(
                in_dim = max(N, opt.min_nfc),
                num_heads = 4,
                block_length = max(N, opt.min_nfc),
            )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            #x,_ = self.attn(x)
            x = x+self.attn(x).permute([0, 3, 1, 2])
        x = self.tail(x)
        return x


class My31GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(My31GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        if opt.attn == True:
            self.attn = ImageAttn(
                in_dim=max(N, opt.min_nfc),
                num_heads=4,
                block_length=max(N, opt.min_nfc),
            )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            #x,_ = self.attn(x)
            x = x+self.attn(x).permute([0, 3, 1, 2])
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

class DecoderAttnLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""
    def __init__(self, in_dim, num_heads, block_length, dropout=0.1):
        super().__init__()
        self.attn = ImageAttn(in_dim, num_heads, block_length)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm_attn = nn.LayerNorm([in_dim], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([in_dim], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(in_dim, 2*in_dim, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2*in_dim, in_dim, bias=True))


    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        y = self.attn(X)
        X = X.permute([0, 2, 3, 1])
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(X)
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X.permute([0, 3, 1, 2]).contiguous()

#exact image transformer implementation
class My32WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(My32WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
        if opt.attn == True:
            self.attn = DecoderAttnLayer(
                in_dim = max(N, opt.min_nfc),
                num_heads = 4,
                block_length = max(N, opt.min_nfc),
            )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            #x,_ = self.attn(x)
            x = self.attn(x)
        x = self.tail(x)
        return x


class My32GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(My32GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )
        if opt.attn == True:
            self.attn = DecoderAttnLayer(
                in_dim=max(N, opt.min_nfc),
                num_heads=4,
                block_length=max(N, opt.min_nfc),
            )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        if hasattr(self,'attn'):
            x = self.attn(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
		

import torch.nn as nn
import torch.nn.functional as F


class CRnnDiscriminator(nn.Module):
    ''' C-RNN-GAN discrminator
    '''

    def __init__(self,opt , hidden_units=256, drop_prob=0.1, use_cuda=False):

        super(CRnnDiscriminator, self).__init__()

        # params
        _, chns, ln, wd = opt.cur_real_shape
        self.num_feats = chns * (ln+10) * (wd+10)
        self.hidden_dim = 256
        self.num_layers = 2
        self.use_cuda = torch.cuda.is_available()

        self.dropout = nn.Dropout(p=drop_prob)

        self.lstm = nn.LSTM(input_size=self.num_feats, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=True)
        self.fc_layer = nn.Linear(in_features=(2 * self.hidden_dim), out_features=self.num_feats)

    def forward(self, note_seq, state):
        ''' Forward prop
        '''
        if self.use_cuda:
            note_seq = note_seq.cuda()

        seq_len, ch, ln, wd = note_seq.shape
        note_seq = note_seq.unsqueeze(0).view(1, seq_len, -1)
        # note_seq: (batch_size, seq_len, num_feats)
        drop_in = self.dropout(note_seq)  # input with dropout
        # (batch_size, seq_len, num_directions*hidden_size)
        lstm_out, state = self.lstm(drop_in, state)
        # (batch_size, seq_len, 1)
        out = self.fc_layer(lstm_out)
        out = torch.sigmoid(out)

        out = out.squeeze().view(seq_len, ch, ln, wd)

        #num_dims = len(out.shape)
        #reduction_dims = tuple(range(1, num_dims))
        ## (batch_size)
        #out = torch.mean(out, dim=reduction_dims)

        return out, lstm_out, state

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        layer_mult = 2  # for being bidirectional

        if self.use_cuda:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_())

        return hidden

class CRnnGenerator(nn.Module):
    ''' C-RNN-GAN generator
    '''

    def __init__(self, opt, hidden_units=256, drop_prob=0.6, use_cuda=False):
        super(CRnnGenerator, self).__init__()

        # params
        self.use_cuda = torch.cuda.is_available()
        _, chns, ln, wd = opt.cur_real_shape
        self.num_feats = chns * (ln + 10) * (wd + 10)
        self.hidden_dim = 256
        self.fc_layer1 = nn.Linear(in_features=(self.num_feats * 2), out_features=self.hidden_dim)
        self.lstm_cell1 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc_layer2 = nn.Linear(in_features=self.hidden_dim, out_features=self.num_feats)

    def forward(self, z, y, states):
        ''' Forward prop
        '''
        if self.use_cuda:
            z = z.cuda()
        # z: (batch_size, seq_len, num_feats)
        # z here is the uniformly random vector
        orig_shape = z.shape
        z = z.unsqueeze(0).view(1, z.shape[0], -1)
        batch_size, seq_len, num_feats = z.shape

        # split to seq_len * (batch_size * num_feats)
        z = torch.split(z, 1, dim=1)
        z = [z_step.squeeze(dim=1) for z_step in z]

        # create dummy-previous-output for first timestep
        prev_gen = torch.empty([batch_size, num_feats]).uniform_()
        if self.use_cuda:
            prev_gen = prev_gen.cuda()

        # manually process each timestep
        state1, state2 = states  # (h1, c1), (h2, c2)
        gen_feats = []
        for z_step in z:
            # concatenate current input features and previous timestep output features
            concat_in = torch.cat((z_step, prev_gen), dim=-1)
            out = F.relu(self.fc_layer1(concat_in))
            h1, c1 = self.lstm_cell1(out, state1)
            h1 = self.dropout(h1)  # feature dropout only (no recurrent dropout)
            h2, c2 = self.lstm_cell2(h1, state2)
            prev_gen = self.fc_layer2(h2)
            # prev_gen = F.relu(self.fc_layer2(h2)) #DEBUG
            gen_feats.append(prev_gen)

            state1 = (h1, c1)
            state2 = (h2, c2)

        # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        gen_feats = [gen_feats_step.view(orig_shape[1], orig_shape[2], orig_shape[3]) for gen_feats_step in gen_feats]
        gen_feats = torch.stack(gen_feats, dim=0)

        states = (state1, state2)
        return gen_feats+y, states

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        if (self.use_cuda):
            hidden = ((weight.new(batch_size, self.hidden_dim).zero_().cuda(),
                       weight.new(batch_size, self.hidden_dim).zero_().cuda()),
                      (weight.new(batch_size, self.hidden_dim).zero_().cuda(),
                       weight.new(batch_size, self.hidden_dim).zero_().cuda()))
        else:
            hidden = ((weight.new(batch_size, self.hidden_dim).zero_(),
                       weight.new(batch_size, self.hidden_dim).zero_()),
                      (weight.new(batch_size, self.hidden_dim).zero_(),
                       weight.new(batch_size, self.hidden_dim).zero_()))

        return hidden