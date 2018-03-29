#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:05:05 2018

@author: chocolatethunder
"""

import torch.nn as nn 
import torch.nn.functional as F

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return F.leaky_relu(nn.Sequential(*layers),0.02, inplace = True)



def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return F.leaky_relu(nn.Sequential(*layers),0.02, inplace = True)

class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
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
        super(Discriminator, self).__init__()
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
    