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
       
        self.hidden_size=1024
        self.out_size=85
        self.dropout_rate=.5 #figure out what dropout rate to use. fine tuning?
        self.num_iterations=3
        self.SMPLnum=85
        

        #TODO figure out args of resnet initialization, got to output right shape.
        
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
    
    def init_param_est(self,batch_size):
        est=np.zeros((1,self.SMPLnum))
        #initial scale is 0.9
        est[0,0]=0.9
        #The rest of the initializations are gotten from SMPL data, which needs to be figured out. 
        #returns the initial parameter estimates. 
        est=tf.convert_to_tensor(est,tf.float32)
        self.est_best=tf.Variable(est)
        init_est=tf.tile(self.est_best,[batch_size,1]) 
        return init_est
    def call(self,features):
        #use functions previously defined to extract features, the run said features.
        #output 85 dim vector, returned by iterative regression. 
        #make sure to output in correct shape for SMPL or STAR, then to
        batch_size=tf.shape(features)[0]
        curr_est=self.init_param_est(batch_size)
        for i in range(self.num_iterations):
            curr_est=self.IEF(features,curr_est)
        
        return curr_est
    

class Discriminator(tf.keras.Model):
    def __init__(self):
        """
        This model will contain code for the Discriminator
        """
        super(Discriminator, self).__init__()
        self.num_joints = 23
        self.poseMatrixShape=[3,3]
        self.learning_rate=1e-3
        
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #ShapeDiscriminator
        self.shapeD1=tf.keras.layers.Dense(10,activation='relu')
        self.shapeD2=tf.keras.layers.Dense(5,activation='relu')
        self.shapeOut=tf.keras.layers.Dense(1,activation='softmax')

        #the pose embedding network, common to all pose discriminators.
        #this needs to be changed, to convolution because inputs are matices
        self.pE1=tf.keras.layers.Conv2D(32,(1,1),input_shape=self.poseMatrixShape,data_format="channels_last")
        self.pE2=tf.keras.layers.Conv2D(32,(1,1),input_shape=self.poseMatrixShape,data_format="channels_last")
        #self.pe1=tf.keras.layers.Dense(32,activation='relu')
        #self.pe2=tf.keras.layers.Dense(32,activation='relu')

        #individual joint discriminators
        temp=[]
        for i in range(self.num_joints):
            temp.append(tf.keras.layers.Dense(1,activation='softmax'))
        self.jointDiscList=temp


        #ovarallPoseDiscriminator
        self.flatten=tf.keras.layers.Flatten(data_format="channels_last")
        self.poseD1=tf.keras.layers.Dense(1024,activation='relu')
        self.poseD2=tf.keras.layers.Dense(1024,activation='relu')
        self.poseOut=tf.keras.layers.Dense(1,activation='softmax')
        #TODO Initialize Hyperparameters, linear layers, etc
        #initialize all discriminators, for 
    def call(self,poses,shape):
        # this will run the SMPL or STAR parameters through the discriminator network
        #and  output a probability. Returns two values, pose discriminator out and 
        # shape discriminator out.
        """pose: Nx23x1x9
        shape:  N x 10
        
        """
        print(tf.shape(poses),tf.shape(shape))
        shapeDisc=self.shapeD1(shape)
        shapeDisc=self.shapeD2(shapeDisc)
        shapeDisc=self.shapeOut(shapeDisc)

        poseEmb=self.pE1(poses)
        poseEmb=self.pE2(poseEmb)
        print(tf.shape(poseEmb))
        poseDisc=[]
        for i in range(self.num_joints):
            #print(poseEmb[:,i,:,:])
            temp=self.jointDiscList[i](poseEmb[:,i,:,:])
            #print(tf.shape(temp))
            poseDisc.append(temp)
        #print(tf.shape(poseDisc))
        poseDisc=tf.squeeze(tf.stack(poseDisc,axis=1))
        #print(tf.shape(poseDisc))
        poseEmb=self.flatten(poseEmb)
        allPoseDisc=self.poseD1(poseEmb)
        allPoseDisc=self.poseD2(allPoseDisc)
        allPoseDisc=self.poseOut(allPoseDisc)
        allPoseDisc=tf.reshape(allPoseDisc,[-1,1,1])
        """ONce we have a tensor containing the disc output of each joint,
        we can concatenate the disc outptus (K*32 in total) """
        discs=tf.concat([poseDisc,allPoseDisc,shapeDisc],0)
        #Discs shape: Nx(23+1+1)
        return discs
   
