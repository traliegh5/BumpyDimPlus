import sys, os
import shutil
import h5py
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

import numpy as np
import random
import math
import time
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
    finloss=tf.reduce_mean(maskAbsDif)
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
    
    return tf.add(fakeLoss,realLoss)

def genLoss(disFake):
    """input: Nx(23+1+1)
    output: scalar"""
    fakeL=tf.math.subtract(disFake,1)
    fakeL=tf.math.pow(fakeL,2)
    fakeLoss=tf.reduce_mean(tf.reduce_sum(fakeL,axis=1))
    

    
    return fakeLoss

def texture_loss(star_verts, star_faces, bary_map, images, camera):
    """map function is in utilities, necessary for making texture maps. We can have this take in a batch of images,
    and have one batch be comprised of images that have consistant texture (ie videos or consecutive images)
    
    """
    # N x bary_map[0] x 3
    batch_maps = map(star_verts, star_faces, bary_map, images, camera)

    num_maps = tf.shape(bach_maps)[0]
    if (num_maps % 2 == 0):
        num_maps = num_maps - 1
        batch_maps = tf.gather(batch_maps, tf.range(num_maps - 1))
    batch_maps = tf.reshape(batch_maps, [num_maps/2, 2, -1, 3])
    loss = tf.math.abs(batch_maps[:,0] - batch_maps[:,1])
    loss = tf.math.reduce_mean(loss)
    return loss

def train(discriminator,generator,star,feats,labelBatch,meshBatch, images, texture):
    
    with tf.GradientTape() as genTape,tf.GradientTape() as discTape:
        params=generator(feats)
        pose=params[:,3:75]
        shape=params[:,75:]
        camera=params[:,:3]
        #INVESTIGATE inputs outputs of star. in particular, check camera. 
        starOut, joint_out =star(pose,shape,camera)
        joints= joint_out
        
        J_lsp=lsp_STAR(joints)
        keypoints = None
        if not texture:
            keypoints=orth_project(J_lsp,camera)

        #19joints=reduceJoints(joints)
        #keypoints=project(19joints,camera)
         
        """Here, the discriminator takes in (pose,shape) as the parameters, and not just a singe param."""
        #These two reals coem from meshBatch How this gets indexed into depends on the Prior Data shape. 
        #note: going from Prior data to realPose involves using star's rodrigues formula. 

        #assuming meshBatch is the same shape as the params...

        realShape=meshBatch[1]
        realPose=meshBatch[0][:,1:,:,:]
        realPose=tf.reshape(realPose,[-1,23,1,9])
       
        realDisc=discriminator(realPose,realShape)
        
        pose=tf.reshape(pose, [-1, 24, 3])
        pose=tf_rodrigues(pose)
        pose=pose[:,1:,:,:]
        pose=tf.reshape(pose,[-1,23,1,9])
        
        fakeDisc=discriminator(pose,shape)
        
        advLossGen=genLoss(fakeDisc)
        advLossDisc=discLoss(realDisc,fakeDisc)
        if not texture:
            repLoss=reprojLoss(labelBatch,keypoints)

        # make texture maps from meshes(from keypoints) 
        # make visibility mask 
        # input maps and mask into texture loss function
        if texture:
            filename_bary = '/home/gregory_barboy/data/uv_bary.pkl'
            uv = load_obj(filename_bary)
            bary_map = tf.convert_to_tensor(np.array(uv), dtype=tf.float32)
            texLoss=texture_loss(starOut, tf.convert_to_tensor(star.f, dtype=tf.int32), bary_map, images, camera)
            totalGenLoss= 0.5 * advLossGen + 0.3 * repLoss + 0.2 * texLoss
        else:
            totalGenLoss=advLossGen
            totalGenLoss= 0.5 * advLossGen + 0.3 * repLoss
            # totalGenLoss=tf.math.reduce_sum(totalGenLoss)

    gen_loss_file = "gen_loss.txt"
    f = open(gen_loss_file, "a+")
    f.write(str(totalGenLoss.numpy()) +  '\n')
    f.close()

    dis_loss_file = "dis_loss.txt"
    f = open(dis_loss_file, "a+")
    f.write(str(advLossDisc.numpy()) +  '\n')
    f.close()

    gradGen=genTape.gradient(totalGenLoss,generator.trainable_variables)
    gradDisc=discTape.gradient(advLossDisc,discriminator.trainable_variables)
    

    
    generator.optimizer.apply_gradients(zip(gradGen,generator.trainable_variables)) 
    discriminator.optimizer.apply_gradients(zip(gradDisc,discriminator.trainable_variables))    
   
   
   
    return None 
def runOnSet(images,joints,poses,shapes,discriminator,generator,star,resNet,texture):
    
    tf.cast(poses,tf.float32)
    tf.cast(shapes,tf.float32)
    
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
        train(discriminator,generator,star,feats,joint_batch,priorBatch, batch, texture=False)
        # if i==batch_size:
        #     tf.keras.models.save_model(generator,runGen)
        #     tf.keras.models.save_model(discriminator,runDisc)
    
    return None 
## Obj download code Adapted from MPI Star/SMPL demo code
def saveMesh(params):
    pose=params[:,3:75]
    shape=params[:,75:]
    camera=params[:,:3]
    star = STAR(gender='neutral')
    m = star(pose,shape,camera)

    outmesh_path = './test_smpl.obj'
    with open( outmesh_path, 'w') as fp:
        for v in m[0][0]:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in star.f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
    ## Print message
    print('..Output mesh saved to: ', outmesh_path)

def main():
    #todo: initilize models with batch size params
    #load data,
    #for loops for training batches and for  training epochs. 
    #
    #  bookkeeping things, like put in loss printlines 
    
    
    
    batch_size=10
    genFilePath=""
    discFilePath=""
    ModelPath="/home/gregory_barboy/Models"
    genPath ='/home/gregory_barboy/Models/gen_training_checkpoints'
    discPath ='/home/gregory_barboy/Models/disc_training_checkpoints'

   
    generator=Generator()
    discriminator=Discriminator()
    generator(tf.random.uniform([10,2048]))
    discriminator(tf.random.uniform([10,23,1,9]),tf.random.uniform([10,10]))
    
    genCheck=tf.train.Checkpoint(generator)
    discCheck=tf.train.Checkpoint(discriminator)
    genMan=tf.train.CheckpointManager(genCheck,genPath,max_to_keep=3)
    discMan=tf.train.CheckpointManager(discCheck,discPath,max_to_keep=3)
    genCheck.restore(genMan.latest_checkpoint)
    discCheck.restore(discMan.latest_checkpoint)
    if genMan.latest_checkpoint:
        print("Restored generator from {}".format(genMan.latest_checkpoint))
    else:
        print("Initializing generator from scratch.")
    if discMan.latest_checkpoint:
        print("Restored discriminator from {}".format(discMan.latest_checkpoint))
    else:
        print("Initializing discriminator from scratch.")    
    save_gen=genCheck.save('/home/gregory_barboy/BumpyDimPlus/Models/gen_training_checkpoints')
    save_disc=discCheck.save('/home/gregory_barboy/BumpyDimPlus/Models/disc_training_checkpoints')
    
	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	
        # tf.saved_model.save(generator,ModelPath)
        # tf.saved_model.save(discriminator,ModelPath)
        # generator.save("my_Gen")
        # discriminator.save("my_Disc")
    
        # genCheck.read(genPath, options=options)
        # discCheck.read()
        # genCheck.restore(save_gen)
        # discCheck.restore(save_disc)


    epochs=500
    
    num_batches=None
    num_im_feats=2048
    resNet=tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet', classes=num_im_feats,classifier_activation='softmax',pooling='avg')
    
    #Save Mesh
    if len(sys.argv) == 2:
        im_path=sys.argv[1]
        image = load_and_process_image(im_path)
        image = tf.reshape(image, [1,224,224,3])
        feats=resNet(image)
        params=generator(feats)
        saveMesh(params)
        sys.exit()

    star=STAR(gender='neutral')

    # Load Joint annotations
    lsp_dir = "/home/gregory_barboy/data/lsp_images"
    mpii_dir = "/home/gregory_barboy/data/mpii_images"
    #"D://Brown//Senior//CSCI_1470//FINAL//MPII//cropped_mpii"
    h36_dir = "/home/gregory_barboy/data/S7_cropped"
    h36_actions = ['Discussion 1.54138969'  'Greeting 1.55011271'  'Photo.54138969 Posing.55011271'   'WalkDog 1.60457274']
    neutr_mosh="/home/gregory_barboy/data/cmu"
    lsp_joints, mpii_joints h36_joints = load_joints(lsp_dir, mpii_dir, h36_dir, h36_actions)
    poses,shapes= load_cmu(neutr_mosh)
    #FIX THE SHUFFLE!!!!!!
    lsp_joints=tf.convert_to_tensor(lsp_joints,dtype=tf.float32)
    mpii_joints=tf.convert_to_tensor(mpii_joints,dtype=tf.float32)

    for joint_list in h36_joints:
        joint_list = tf.convert_to_tensor(joint_list, dtype=tf.float32)
    poses=tf.convert_to_tensor(poses,dtype=tf.float32)
    shapes=tf.convert_to_tensor(shapes,dtype=tf.float32)
    
    shapes=tf.reshape(shapes,[-1,10])
    moshSize=tf.shape(shapes)[0]
    inders=tf.random.shuffle(range(moshSize))
    poses=tf.gather(poses,inders,axis=0)
    shapes=tf.gather(shapes,inders,axis=0)


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

    #import h36 S7
    h36_datasets = []
    for action in h36_actions:
        dir_path = h36_dir + '/' + action + '/*.png'
        dataset = tf.data.Dataset.list_files(dir_path, shuffle=False)
        dataset = dataset.map(map_func=load_and_process_image)
        dataset = dataset.batch(h36_batch_size)
        h36_datasets.append(dataset)
    
    lsp_ds = lsp_ds.prefetch(1)
    mpii_ds = mpii_ds.prefetch(1)
    for ds in h36_datasets:
        ds = ds.prefetch(1)
    # Iterate over dataset
    #this is likelye not right, but eventually it should be the 
    ModelPath="/home/gregory_barboy/BumpyDimPlus/Models"

    start_texture = 0
    for epoch_num in range(epochs):
        start = time.time()
        #runOnSet(mpii_ds,mpii_joints,poses,shapes,discriminator,generator,star,resNet,False)
        #runOnSet(lsp_ds,lsp_joints,poses,shapes,discriminator,generator,star,resNet,False)
        if(epoch_num > start_texture):
            for i in range(len(h36_datasets)):
                runOnSet(h36_datasets[i],h36_joints[i],poses,shapes,discriminator,generator,star,resNet,True)
        end = time.time()
        print("Epoch: ", epoch_num," took %s minutes. nice!" %((end - start)/60.0))
        
        if epoch_num%10==0:
            genSavePath=genMan.save()
            discSavePath=discMan.save()
            print("Saved checkpoint for step {}: {}".format(epoch_num, genSavePath))
            print("Saved checkpoint for step {}: {}".format(epoch_num, discSavePath))
            
        # save_gen=genCheck.save('/home/gregory_barboy/BumpyDimPlus/Models/gen_training_checkpoints')
        # save_disc=discCheck.save('/home/gregory_barboy/BumpyDimPlus/Models/disc_training_checkpoints')
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
               
        
        
       
        
        # generator.save('my_Gen')
        # discriminator.save('my_Disc')
        # genSavePath=os.path.join(ModelPath,"generator/1/")
        # discSavePath=os.path.join(ModelPath,"discriminator/1/")
        # tf.keras.models.save_model(generator,genSavePath,save_format='tf')
        # tf.keras.models.save_model(discriminator,discSavePath,save_format='tf')
        
if __name__ == '__main__':
    main()
