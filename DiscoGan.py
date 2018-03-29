#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 08:18:24 2018

@author: chocolatethunder
"""
import torch
import torchvision
import os
from itertools import chain
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Generator
import numpy as np


def minibatches(inputs1=None, inputs2=None, batch_size=None, shuffle=False):
    """Generate a generator that input a group of example in numpy.array and
    their labels, return the examples and labels by the given batch size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    targets : numpy.array
        The labels of inputs, every row is a example.
    batch_size : int
        The batch size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.
    """
    assert len(inputs1) == len(inputs2)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        yield inputs1[excerpt], inputs2[excerpt]




class DiscoGAN(object):
    """
    Class for building the Disco GAN model
    """
    def __init__(self, learning_rate = 0.01, betas=(0.9, 0.999), conv_dim = 64):
        
        self.learning_rate = 0.01
        self.betas = betas
        self.conv_dim = conv_dim
        
        self.builf_model()
        
    def build_model(self):
        self.D_a = Discriminator(self.conv_dim)
        self.D_b = Discriminator(self.conv_dim)
        
        self.G_ab = Generator(self.conv_dim)
        self.G_ba = Generator(self.conv_dim)
        
        self.g_paramaters = chain(G_ab.paramaters(),G_ba.paramaters)
        self.d_paramaters = chain(D_a.paramaters(),D_b.paramaters)
        
        self.g_optimizer = optim.Adam(self.g_parameters(),
                                      self.learning_rate, self.betas)
        self.d_optimizer = optim.Adam(self.d_parameters(),
                                      self.learning_rate, self.betas)
        
        self.recon_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCELoss()
    
    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def discriminator_loss(real_images, fake_images, loss):
        """ Find the discriminator loss """
        real_labels = Variable(torch.ones( [real_images.size()[0], 1] ))
        fake_labels = Variable(torch.zeros([fake_images.size()[0], 1] ))
        
        return( 0.5*loss(real_images,real_labels)+0.5*loss(fake_images,fake_labels) )
    
    def gan_loss(gen_images,loss):
        """ Find the GAN loss """
        labels = Variable(torch.ones( [gen_images.size()[0], 1] ))
        
        return(loss(gen_images,labels))
        
        
    
    def train(self, data, epochs=100, batch_size=256):
        for epoch in range(epochs):
            
            #look up what tl.iterare.minibatches does!!!!!!!!!!!!!!!!!
            for data_a, data_b in minibatches(data.a.train, data.b.train, batch_size, shuffle=True):

                x_a = self.to_variable(data_a)
                x_b = self.to_variable(data_b)
                
                self.D_a.zero_grad()
                self.D_b.zero_grad()
                
                
                x_ab = self.G_ab(x_a).detach()
                x_ba = self.G_ab(x_b).detach()
                
                discriminator_loss_b = self.discriminator_loss(self.D_a(x_a), self.D_a(x_ba), self.gan_criterion)
                discriminator_loss_a = self.discriminator_loss(self.D_b(x_b), self.D_b(x_ab), self.gan_criterion)
                
                discriminator_loss = discriminator_loss_b + discriminator_loss_a
                
                discriminator_loss.backward()
                
                self.d_optimimizer.step()
                
                
                self.G_ab.zero_grad()
                self.G_ba.zero_grad()
                
                
                x_ab = self.G_ab(x_a)
                x_ba = self.G_ab(x_b)

                x_aba = self.G_ab(x_ab)
                x_bab = self.G_ab(x_ba)
                
                
                recon_loss_a = self.recon_criteron(x_a,x_aba)
                recon_loss_b = self.recon_criteron(x_b,x_bab)
                
                recon_loss = recon_loss_a + recon_loss_b
                
                gan_loss_a = self.gan_loss(self.D_a(x_ba))
                gan_loss_b = self.gan_loss(self.D_b(x_ab))
                
                gan_loss = recon_loss+0.5*gan_loss_a+0.5*gan_loss_b
                
                gan_loss.backward()
                
                self.g_optimizer.step()
                
                
        
        
