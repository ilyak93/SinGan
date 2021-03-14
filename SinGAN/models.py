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