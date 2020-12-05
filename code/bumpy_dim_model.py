import os
import tensorflow as tf
import numpy as np
import random
import math

class Generator(tf.keras.Model):
    def __init__(self):
        """
        This model will contain code for the generator
        Set up all of the functions for batching!!!
        """
        super(Generator, self).__init__()
        #TODO Initialize Hyperparameters, linear layers, etc
        self.learning_rate=1e-5
        self.num_im_feats=2048
        self.hidden_size=1024
        self.out_size=85
        self.dropout_rate=.5 #figure out what dropout rate to use. fine tuning?
        self.num_iterations=3


        #TODO figure out args of resnet initialization, got to output right shape.
        self.ResNet=tf.keras.applications.ResNet50V2(classes=self.num_im_feats)
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        


        #3D module (as referred to in paper)
        self.layer1=tf.keras.layers.Dense(self.hidden_size,activation='relu')
        self.drop1=tf.keras.layers.Dropout(self.dropout_rate,input_shape=(self.hidden_size,))
        self.layer2=tf.keras.layers.Dense(self.hidden_size,activation='relu')
        self.drop2=tf.keras.layers.Dropout(self.dropout_rate,input_shape=(self.hidden_size,))
        self.layer3=tf.keras.layers.Dense(self.out_size)
    def IEF(self,imFeats,estimate):
        #perform one iteration of 3d regression
        #this is the current sketch. first, store the current estimate, then create our input vector,
        #then run our input vector through the network,
        #finally, add the output of our network with the current estimate as the update to our estimate
        #return the current estimate.
        curr_est=estimate
        vect=tf.concat((imFeats,estimate),1)
        reg=self.layer1(vect)
        reg=self.drop1(reg)
        reg=self.layer2(reg)
        reg=self.drop2(reg)
        reg=self.layer3(reg)
        curr_est+=reg
        return curr_est
    def extract_features(self,img):
        #use Resnet50 to extract image features
        #return image features
        #might also want to reshape images here? this function might be unnecessary.
        features=self.ResNet.call(img)
        return features
    def init_param_est(self):
        est=[]
        #returns the initial parameter estimates. 
        return est
    def call(self,x):
        #use functions previously defined to extract features, the run said features.
        #output 85 dim vector, returned by iterative regression. 
        #make sure to output in correct shape for SMPL or STAR, then to
        features=self.extract_features(x)
        curr_est=self.init_param_est()
        for i in range(self.num_iterations):
            curr_est=self.IEF(features,curr_est)
        
        return curr_est
    def loss(self,x,y,z):
        #this loss will not be the total loss, which depends on the discriminator
        #This loss will only be the reprojections loss (maybe)
        #This function might not need to exist, the structure of this class can change depending
        #on What we want, and how we want to do it. 
        pass 

class Discriminator(tf.keras.Model):
    def __init__(self):
        """
        This model will contain code for the Discriminator
        """
        super(Discriminator, self).__init__()


        #TODO Initialize Hyperparameters, linear layers, etc
        #initialize all discriminators, for 
    def call(self,x):
        # this will run the SMPL or STAR parameters through the discriminator network
        #and  output a probability. Returns two values, pose discriminator out and 
        # shape discriminator out.
        pass

    def loss(self,x,y,z):
        #TODO calculate loss, figure out what is input to loss function
        # for now, inputs are the generated SMPL/STAR params, and the real mesh params from unpaired dataset.

        pass 

