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
from model import DiscriminatorCNN
from model import GeneratorCNN
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import torch.nn as nn
import matplotlib.pyplot as plt
import utils




class DiscoGAN(object):
    """
    Class for building the Disco GAN model
    """
    def __init__(self, learning_rate = 0.01, betas=(0.9, 0.999), conv_dim = 64):
        
        self.learning_rate = 0.01
        self.betas = betas
        self.conv_dim = conv_dim
        self.build_model()
        

    def build_model(self):
        self.D_a = DiscriminatorCNN(first_dim = self.conv_dim)
        self.D_b = DiscriminatorCNN(first_dim =self.conv_dim)
        
        self.G_ab = GeneratorCNN(first_dim =self.conv_dim)
        self.G_ba = GeneratorCNN(first_dim =self.conv_dim)
        
        self.g_parameters= chain(self.G_ab.parameters(),self.G_ba.parameters())
        self.d_parameters = chain(self.D_a.parameters(),self.D_b.parameters())
        
        self.g_optimizer = optim.Adam(self.g_parameters,
                                      self.learning_rate, self.betas)
        self.d_optimizer = optim.Adam(self.d_parameters,
                                      self.learning_rate, self.betas)
        
        self.recon_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCELoss()
    
    
    def data_loader( self, train_data_dir, batch_size = 256, img_height = 64, img_width = 64):
    
    # Initiate the train and test generators with data Augumentation 
        datagen = ImageDataGenerator(
                rescale = 1./255,
                fill_mode = "nearest")


        data_generator = datagen.flow_from_directory(
                train_data_dir,
                target_size = (img_height, img_width),
                batch_size = batch_size, 
                class_mode = None,
                shuffle = True)
    
        return(data_generator)
    
    def as_np(self, data):
        return data.cpu().data.numpy()
    
    def MSEloss(self,x,y):
        return(( x - y) * (x - y)).mean()()
    
    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def discriminator_loss(self, real_images, fake_images, loss):
        """ Find the discriminator loss """
        real_labels = Variable(torch.ones( [real_images.size()[0], 1] ))
        fake_labels = Variable(torch.zeros([fake_images.size()[0], 1] ))
        
        return( 0.5*loss(real_images,real_labels)+0.5*loss(fake_images,fake_labels) )
    
    def gan_loss(self,gen_images,loss):
        """ Find the GAN loss """
        labels = Variable(torch.ones( [gen_images.size()[0], 1] ))
        
        return(loss(gen_images,labels))
    
    def show_images(self,im_A, im_B):
        
        fig=plt.figure(figsize=(8, 8))
        columns = 3
        rows = 2
        
        for i,img in enumerate(im_A+ im_B):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img.swapaxes(0,2).swapaxes(0,1))
        plt.show()
        
        return(plt)
    
    
    def generate(self, a,b, path, save = True, epoch = None ):
            
        
        A = a.swapaxes(1,3).swapaxes(2,3)
        B = b.swapaxes(1,3).swapaxes(2,3)
        A = self.to_variable(torch.from_numpy(A))
        B = self.to_variable(torch.from_numpy(B))
        AB = self.G_ab( A )
        BA = self.G_ba( B )
        ABA = self.G_ba( AB )
        BAB = self.G_ab( BA )
                    
        num_images = 10 
        
        if epoch is None :
            f_a = path+'A_Epoch_'
            f_b = path+'B_Epoch_'
        else :
            f_a =path +'A_'
            f_b =path +'B_'
     
        for im in range(num_images):      
            if save :
                            
                A_val = A[im]
                B_val = B[im]
                BA_val = BA[im]
                ABA_val = ABA[im]
                AB_val = AB[im]
                BAB_val = BAB[im]
                            
                filename = f_a+str(epoch)+'im_'+str(im)+'.jpg'
                utils.save_image([A_val.data,AB_val.data,ABA_val.data], filename = filename, nrow=3, padding=10)
                filename = f_b+str(epoch)+'im_'+str(im)+'.jpg'
                utils.save_image([B_val.data,BA_val.data,BAB_val.data], filename = filename , nrow=3, padding=10)
            else:         
                A_val = A[im].cpu().data.numpy() 
                B_val = B[im].cpu().data.numpy() 
                BA_val = BA[im].cpu().data.numpy()
                ABA_val = ABA[im].cpu().data.numpy()
                AB_val = AB[im].cpu().data.numpy()
                BAB_val = BAB[im].cpu().data.numpy()
                self.plt.close()
                            
                self.plt = self.show_images([A_val,AB_val,ABA_val], [B_val,BA_val,BAB_val])
    
    
    def train(self, data_dir, n_epochs=100, batch_size=256, print_freq = 1, save = True, path = 'Generated Images/'):
        
        train_dir_a = data_dir + 'train_A/'
        train_dir_b = data_dir + 'train_B/'
        
        val_dir_a = data_dir + 'val_A/'
        val_dir_b = data_dir + 'val_B/'
        
        path_train = path+'train'
        path_val = path+'val'
        
        val_A_generator = self.data_loader(val_dir_a, batch_size =  10)
        val_B_generator = self.data_loader(val_dir_b, batch_size = 10)
        
        
        
        
        for epoch in range(n_epochs):
            
            A_generator = self.data_loader(train_dir_a, batch_size = batch_size)
            B_generator = self.data_loader(train_dir_b, batch_size = batch_size)
            
            ctr = 0 
            for data_a, data_b in zip(A_generator, B_generator):
                
                data_a = data_a.swapaxes(1,3).swapaxes(2,3)
                data_b = data_b.swapaxes(1,3).swapaxes(2,3)
                
                x_a = self.to_variable(torch.from_numpy(data_a))
                x_b = self.to_variable(torch.from_numpy(data_b))
                
                self.D_a.zero_grad()
                self.D_b.zero_grad()
                
                x_ab = self.G_ab(x_a).detach()
                x_ba = self.G_ba(x_b).detach()
                
                discriminator_loss_b = self.discriminator_loss(self.D_a(x_a), self.D_a(x_ba), self.gan_criterion)
                discriminator_loss_a = self.discriminator_loss(self.D_b(x_b), self.D_b(x_ab), self.gan_criterion)
                
                discriminator_loss = discriminator_loss_b + discriminator_loss_a
                
                discriminator_loss.backward()
                
                self.d_optimizer.step()
                
                self.G_ab.zero_grad()
                self.G_ba.zero_grad()
                
                
                x_ab = self.G_ab(x_a)
                x_ba = self.G_ba(x_b)

                x_aba = self.G_ba(x_ab)
                x_bab = self.G_ab(x_ba)
                
                
                recon_loss_a = self.recon_criterion(x_aba, x_a)
                recon_loss_b = self.recon_criterion(x_bab, x_b)
                
                recon_loss = recon_loss_a + recon_loss_b
                
                gan_loss_a = self.gan_loss(self.D_a(x_ba), self.gan_criterion)
                gan_loss_b = self.gan_loss(self.D_b(x_ab), self.gan_criterion)
                
                gan_loss = recon_loss+(0.5*gan_loss_a+0.5*gan_loss_b)*0.1
                
                gan_loss.backward()
                
                self.g_optimizer.step()
                
                if ctr == len(A_generator):
                    break
                ctr += 1
                
            
            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print( "---------------------")
                print ("GEN Loss:", self.as_np(gan_loss_a.mean()), self.as_np(gan_loss_b.mean()))
                print ("RECON Loss:", self.as_np(recon_loss_a.mean()), self.as_np(recon_loss_b.mean()))
                print ("DIS Loss:", self.as_np(discriminator_loss_a.mean()), self.as_np(discriminator_loss_b.mean()))
                
                #self.


                for data_a, data_b in zip(A_generator, B_generator):
                    
                    
                    self.generate(data_a, data_b, save = save, path = path_train, epoch = epoch)
                    
                    break
              
                
                for val_a, val_b in zip(val_A_generator, val_B_generator):
                    
                    
                    self.generate(val_a, val_b, save = save, path = path_val, epoch = epoch)
                    
                    break
                    
                    
                    
                    

                
        
