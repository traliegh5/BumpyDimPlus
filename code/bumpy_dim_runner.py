import os
import tensorflow as tf
import numpy as np
import random
import math
from bumpy_dim_model import Generator, Discriminator
"""the doc where we will run all of this. we can use the 
main func to do this. I think that we can use functions outside
of the model to calculate losses. The structure of this is not
set in stone, I'm just getting the ball rolling. """
num_im_feats=2048
resNet=tf.keras.applications.ResNet50V2(classes=num_im_feats)
generator=Generator()
discriminator=Discriminator()
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
    """not sure what else needs to be done in this function. First i reshape the keys and predKeys, then i compute a visibility
    mask using the 3rd element in the 3rd dimmension of keys, then I compute keypoint loss, masking this with visibility.
    The shape of this might not be right... this doesnt return a scalar, but I dont want to edit it yet. We should probably reduce sum.
    """
    
    return maskAbsDif

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
def train(discriminator,generator,imageBatch,labelBatch,meshBatch):
    feats=resNet(imageBatch)
    with tf.GradientTape() as tape:
        params=generator(feats)
        keypoints=[params]
        #keypoints=SOMETHING
        realDisc=discriminator(meshBatch)
        fakeDisc=discriminator(keypoints)
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
 