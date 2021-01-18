import sys, os
import shutil
import h5py
import tensorflow as tf
import numpy as np
import random
import math
from utilities import orth_project,  lsp_STAR
from bumpy_dim_model import Generator, Discriminator
sys.path.append('/home/gregory_barboy/BumpyDimPlus/STAR/')
sys.path.append('/home/gregory_barboy/data')

from star.tf.star import STAR, tf_rodrigues
from data_loader import load_joints, load_and_process_image,load_cmu
import matplotlib.pyplot as plt

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

def train(discriminator,generator,star,feats,labelBatch,meshBatch,texture):
    
    with tf.GradientTape() as tape:
        params=generator(feats)
        pose=params[:,3:75]
        shape=params[:,75:]
        camera=params[:,:3]
        #INVESTIGATE inputs outputs of star. in particular, check camera. 
        
        joints=star(pose,shape,camera).Jtr
        J_lsp=lsp_STAR(joints)
        if not texture:
            keypoints=orth_project(J_lsp)



        #19joints=reduceJoints(joints)
        #keypoints=project(19joints,camera)
         
        """Here, the discriminator takes in (pose,shape) as the parameters, and not just a singe param."""
        #These two reals coem from meshBatch How this gets indexed into depends on the Prior Data shape. 
        #note: going from Prior data to realPose involves using star's rodrigues formula. 

        #assuming meshBatch is the same shape as the params...

        realShape=meshBatch[1]
        realPose=meshBatch[0]
        realDisc=discriminator(realShape,realPose)
        fakeDisc=discriminator(pose,shape)
        advLossGen=genLoss(fakeDisc)
        advLossDisc=discLoss(realDisc,fakeDisc)
        if not texture:
            repLoss=reprojLoss(labelBatch,keypoints)

        # make texture maps from meshes(from keypoints) 
        # make visibility mask 
        # input maps and mask into texture loss function
        if texture:
            texLoss=texture_loss()
            totalGenLoss=tf.concat([advLossGen,texLoss],0)
        else:
            totalGenLoss=tf.concat([advLossGen,repLoss],0)
        totalGenLoss=tf.math.reduce_sum(totalGenLoss)
    gradDisc=tape.gradient(advLossDisc,discriminator.trainable_variables)
    gradGen=tape.gradient(totalGenLoss,generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(gradGen,generator.trainable_variables)) 
    discriminator.optimizer.apply_gradients(zip(gradDisc,discriminator.trainable_variables))    
   
   
   
    return None 
def runOnSet(images,joints,poses,shapes,discriminator,generator,star,resNet,texture):
    print(type(poses))
    tf.cast(poses,tf.float32)
    tf.cast(shapes,tf.float32)
    print(type(poses))
    for i, batch in enumerate(images):
        batch_size=tf.shape(batch)[0]
        indies=tf.random.shuffle(range(batch_size))
        poseBatch=tf.gather(poses[i:i+batch_size,:],indies,axis=0)

        poseBatch=tf.reshape(poseBatch,[batch_size,24,3])
        #this reshape makes sense, but num_joints is 23... so one might have to be dropped
        
        poseBatch=tf_rodrigues(poseBatch)
        shapeBatch=tf.gather(shapes[i:i+batch_size,:],indies,axis=0)

        imBatch=tf.gather(batch,indies)
        joint_batch=tf.gather(joints[i:i+batch_size,:,:],indies,axis=0)
        
        priorBatch=[poseBatch,shapeBatch]
        feats=resNet(imBatch)
        train(discriminator,generator,star,feats,joint_batch,priorBatch,texture=False)
        # if i==batch_size:
        #     tf.keras.models.save_model(generator,runGen)
        #     tf.keras.models.save_model(discriminator,runDisc)
    
    return None 
def main():
    #todo: initilize models with batch size params
    #load data,
    #for loops for training batches and for  training epochs. 
    #
    #  bookkeeping things, like put in loss printlines 
    batch_size=10
    genFilePath=""
    discFilePath=""
    if len(sys.argv)!=2:
        generator=Generator()
        discriminator=Discriminator()
    elif sys.argv[1]=="Load":
        generator=tf.keras.models.load_model(genFilePath)
        discriminator=tf.keras.models.load_model(discFilePath)


    epochs=1
    
    num_batches=None
    num_im_feats=2048
    resNet=tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet', classes=num_im_feats,classifier_activation='softmax')

   
    star=STAR(gender='neutral')

    # Load Joint annotations
    lsp_dir = "/home/gregory_barboy/data/lsp_images"
    mpii_dir = "/home/gregory_barboy/data/mpii_images"
    #"D://Brown//Senior//CSCI_1470//FINAL//MPII//cropped_mpii"
    h36_dir = ""
    neutr_mosh="/home/gregory_barboy/data/cmu"
    lsp_joints, mpii_joints = load_joints(lsp_dir, mpii_dir, h36_dir)
    poses,shapes= load_cmu(neutr_mosh)
    shapes=tf.reshape(shapes,[-1,10])

    mpii_batch_size=100
    lsp_batch_size=100
    
    # Create Image datasets
    # Create a Dataset that contains all .png files
    # in a directory
    dir_path = lsp_dir + '/*.png'
    dataset = tf.data.Dataset.list_files(dir_path, shuffle=False)
    # Apply a function that will read the contents of
    # each file into a tensor
    dataset = dataset.map(map_func=load_and_process_image)
    # Load up data in batches
    dataset = dataset.batch(lsp_batch_size)
    # Prefetch the next batch while GPU is training
    lsp_ds = dataset

    # in a directory
    dir_path = mpii_dir + '/*.png'
    dataset = tf.data.Dataset.list_files(dir_path, shuffle=False)
    dataset = dataset.map(map_func=load_and_process_image)
    dataset = dataset.batch(mpii_batch_size)
    mpii_ds = dataset
    
    lsp_ds = lsp_ds.prefetch(1)
    mpii_ds = mpii_ds.prefetch(1)
    # Iterate over dataset
    #this is likelye not right, but eventually it should be the 
    ModelPath="/home/gregory_barboy/BumpyDimPlus/Models"

    
    for epoch_num in range(epochs):
        runOnSet(mpii_ds,mpii_joints,poses,shapes,discriminator,generator,star,resNet,False)
        runOnSet(lsp_ds,lsp_joints,poses,shapes,discriminator,generator,star,resNet,False)

        #  for i, batch in enumerate(lsp_ds):

        # for i, batch in enumerate(mpii_ds):
            
        #     #print(i)
        #     #img = batch[0]
        #     #imgplot = plt.imshow(img)
        #     #plt.show()
        
        #         #batching: depends on what we do for data, I'm not sure what to do here.
        #         #once you have a batch, run train method on that batch. 
        #         #
        #     indies=tf.random.shuffle(range(batch_size))
        #     poseBatch=tf.gather(poses[i:i+batch_size,:],indies,axis=0)
        
        #     poseBatch=tf.reshape(poseBatch,[batch_size,24,3])
        #     #this reshape makes sense, but num_joints is 23... so one might have to be dropped
        #     poseBatch=tf_rodrigues(poseBatch)
        #     shapeBatch=tf.gather(shapes[i:i+batch_size,:],indies,axis=0)

        #     imBatch=tf.gather(batch,indies)
        #     joint_batch=tf.gather(mpii_joints[i:i+batch_size,:,:],indies,axis=0)
            
        #     priorBatch=[poseBatch,shapeBatch]
        #     feats=resNet(imBatch)
        #     train(discriminator,generator,star,feats,joint_batch,priorBatch,texture=False)
        #     if i==batch_size:
        #         tf.keras.models.save_model(generator,runGen)
        #         tf.keras.models.save_model(discriminator,runDisc)
               
        
        
       

        
        tf.keras.models.save_model(generator,ModelPath)
        tf.keras.models.save_model(discriminator,ModelPath)
        
if __name__ == '__main__':
    main()
