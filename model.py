#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:05:05 2018

@author: chocolatethunder
"""

import torch
from torch import nn
import torch.nn.functional as F

def conv(c_in, c_out, k_size = 4, stride=2, pad=1, bn=True, ReLU = True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if ReLU:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers



def deconv(c_in, c_out, k_size = 4, stride=2, pad=1, bn=True, ReLU= True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if ReLU :
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class DiscriminatorCNN(nn.Module):
    def __init__(self, input_channel = 3, first_dim = 64, num_layers = 5):
        super().__init__()
        self.layers = []
        
        self.layers += conv(input_channel, first_dim, bn= False)
        prev_dim = first_dim

        for layer in range(num_layers-2):
            out_dim = prev_dim*2
            self.layers += conv(prev_dim, out_dim, bn= True)
            prev_dim = out_dim

        self.layers += conv(prev_dim, 1, stride=1, pad=0, bn= False, ReLU = False)
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(out.size(0), -1)

    def forward(self, x):
        return self.main(x)
    
    

class GeneratorCNN(nn.Module):
    def __init__(self, input_channel = 3, first_dim = 64, num_layers = 8):
        super().__init__()
        self.layers = []
        
        self.layers += conv(input_channel, first_dim, bn= False)
        prev_dim = first_dim

        for layer in range(int(num_layers/2 -1)):
            out_dim = prev_dim*2
            self.layers += conv(prev_dim, out_dim, bn= True)
            prev_dim = out_dim
            
        for layer in range(int(num_layers/2 -1)):
            out_dim = int(prev_dim/2)
            self.layers += deconv(prev_dim, out_dim, bn= True)
            prev_dim = out_dim
            
        self.layers += deconv(prev_dim, input_channel, bn= False)
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def main(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out #.view(out.size(0), -1)

    def forward(self, x):
        return self.main(x)




class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super().__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.conv5 = conv(conv_dim*8, 1, 4, bn = False)
        
    def forward(self, x):                         # If image_size is 64, output shape is as below.
        out = self.conv1(x)   # (?, 64, 32, 32)
        out = self.conv2(out)  # (?, 128, 16, 16)
        out = self.conv3(out)  # (?, 256, 8, 8)
        out = self.conv4(out)  # (?, 512, 4, 4)
        out = self.conv5(out)  # (?, 512, 4, 4)
        return F.sigmoid(out)
    


class Generator(nn.Module):
    """Generator containing 4 convolutional and deconvolutional layers."""
    def __init__(self, conv_dim=64):
        super().__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.deconv1 = deconv(conv_dim*8,conv_dim*4,4)
        self.deconv2 = deconv(conv_dim*4,conv_dim*2,4)
        self.deconv3 = deconv(conv_dim*2,conv_dim,4)
        self.deconv4 = deconv(conv_dim,3,4, bn = False)
        
    def forward(self, x):                         # If image_size is 64, output shape is as below.
        out = self.conv1(x)   # (?, 64, 32, 32)
        out = self.conv2(out)  # (?, 128, 16, 16)
        out = self.conv3(out)  # (?, 256, 8, 8)
        out = self.conv4(out)  # (?, 512, 4, 4)
        out = self.conv5(out)  # (?, 512, 4, 4)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return F.sigmoid(out) #why is there a sigmoid here ???
    