#!/usr/bin/python
'''
Defines the Encoders and Decoder that make up the
Enc+Dec/Enc+2Dec model defined in DLoc
Idea and code originally from fstin Johnson's architecture.
https://github.com/jcjohnson/fast-neural-style/
Code base expanded from pix2pix Phillip Isola's implementation
https://github.com/phillipi/pix2pix

You can add your costum network building blocks here to test various other architectures.
'''
import torch
import torch.nn as nn
import functools


# Tha base Encoder function defined for the DLoc's Encoder
class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='zero'):
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, input_nc, kernel_size=7, padding=3,
                           bias=use_bias),
                 norm_layer(input_nc),
                 nn.Tanh()]

        model += [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Tha base Decoder function defined for the DLoc's Decoders.
# Depending upon the ModelADT wraper around the decoder,
# the decoder would either be a Location Decoder or a Consistency decoder.
# For more details refer to ModelADT.py wrapper implementation and params.py
class ResnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='zero', encoder_blocks=6):
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i

        mult = 2**n_downsampling
        for i in range(n_blocks):
            if i <= encoder_blocks:
                continue
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=[3,3])]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        #print("decoder.input = ", input.shape)
        return self.model(input)
# In[3]:


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.25)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out