import os
import tensorflow as tf
import numpy as np
import random
import math
from utilities import orth_project,  lsp_STAR
from bumpy_dim_model import Generator, Discriminator
from .STAR.tf.star import STAR,tf_rodrigues




def reprojLoss(keys,predKeys):
    """keys: N x K x 3
      predKeys: N x K x 2
      """
    keys=tf.reshape(keys,(-1,3))
    predKeys=tf.reshape(predKeys,(-1,2))  
    visMask=keys[:,2]
    dif=tf.math.subtract(keys[:,:2],predKeys)
    absDif=tf.math.abs(dif)
    maskAbsDif=tf.boolean_mask(absDif,visMask)
    finloss=tf.reduce_sum(maskAbsDif)
   
    
    return finloss

#
def discLoss(disReal,disFake):
    """"inputs:
    disReal: Nx(23+1+1)
    diFake: Nx(23+1+1)
    output:
    scalar
    """
    fakeL=tf.math.pow(disFake,2)
    realL=tf.math.subtract(disReal,1)
    realL=tf.math.pow(realL,2)
    fakeLoss=tf.reduce_mean(tf.reduce_sum(fakeL,axis=1))
    realLoss=tf.reduce_mean(tf.reduce_sum(realL,axis=1))
    
    return fakeLoss+realLoss

def genLoss(disFake):
    """input: Nx(23+1+1)
    output: scalar"""
    fakeL=tf.math.subtract(disFake,1)
    fakeL=tf.math.pow(fakeL,2)
    fakeLoss=tf.reduce_mean(tf.reduce_sum(fakeL,axis=1))
    

    
    return fakeLoss

def texture_loss():
    """map function is in utilities, necessary for making texture maps. We can have this take in a batch of images,
    and have one batch be comprised of images that have consistant texture (ie videos or consecutive images)
    
    """
    
    return None 

def train(discriminator,generator,star,feats,labelBatch,meshBatch):
    
    with tf.GradientTape() as tape:
        params=generator(feats)
        pose=params[:,3:75]
        shape=params[:,75:]
        camera=params[:,:3]
        #INVESTIGATE inputs outputs of star. in particular, check camera. 
        joints=star(pose,shape,camera).Jtr
        J_lsp=lsp_STAR(joints)
        keypoints=orth_project(J_lsp)



        #19joints=reduceJoints(joints)
        #keypoints=project(19joints,camera)
         
        """Here, the discriminator takes in (pose,shape) as the parameters, and not just a singe param."""
        #These two reals coem from meshBatch How this gets indexed into depends on the Prior Data shape. 
        #note: going from Prior data to realPose involves using star's rodrigues formula. 

        #assuming meshBatch is the same shape as the params...

        realShape=meshBatch[:,75:]
        realPose=tf_rodrigues(meshBatch[:,3:75])
        realDisc=discriminator(realShape,realPose)
        fakeDisc=discriminator(pose,shape)
        advLossGen=genLoss(fakeDisc)
        advLossDisc=discLoss(realDisc,fakeDisc)
        repLoss=reprojLoss(labelBatch,keypoints)

        # make texture maps from meshes(from keypoints) 
        # make visibility mask 
        # input maps and mask into texture loss function
        texLoss=texture_loss()

        totalGenLoss=tf.concat([advLossGen,repLoss,texLoss],0)
        totalGenLoss=tf.math.reduce_sum(totalGenLoss)
    gradDisc=tape.gradient(advLossDisc,discriminator.trainable_variables)
    gradGen=tape.gradient(totalGenLoss,generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(gradGen,generator.trainable_variables)) 
    discriminator.optimizer.apply_gradients(zip(gradDisc,discriminator.trainable_variables))    
   
   
   
    return None 
 
def main():
    #todo: initilize models with batch size params
    #load data,
    #for loops for training batches and for  training epochs. 
    #
    #  bookkeeping things, like put in loss printlines 
    epochs=10
    batch_size=400
    num_batches=None
    num_im_feats=2048
    resNet=tf.keras.applications.ResNet50V2(classes=num_im_feats)

    generator=Generator(batch_size)
    discriminator=Discriminator(batch_size)
    star=STAR()
    
    for i in range(0,epochs):
        for j in range(0,num_batches):
            #batching: depends on what we do for data, I'm not sure what to do here.
            #once you have a batch, run train method on that batch. 
            #
            imBatch=None
            labelBatch=None
            priorBatch=None
            feats=resNet(imBatch)

            train(discriminator,generator,star,feats,labelBatch,priorBatch)

    return None
