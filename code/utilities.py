import os
import tensorflow as tf
import numpy as np
import random
import math

"""this model can be for all intermediate processing methods, potentially. things like projection, rodriguez formula,
maybe smpl/STAR stuff? all of the math transformations we need to do can be located here. If this structure is unnecessary 
then feel free to do it another waydone
"""



def map(meshes):
    """it doesn't need to be done here, but we need to create texture maps. To do this,we need to project the mesh back into the images
    space, then convolve over it with a "bilinear sampling kernel" which estimates texture. we can look at differences 
    across texture maps of consecutive images as loss. """

    #step1: projet mesh onto image space with camera parameters
    #step2: convolve this image to output texture
    #this is done with tf.image.resize(method=ResizeMethod.BILINEAR)
    #step3: output texture map

    pass



def orth_project(PointBatch,camera):
    """
    These are the indices for extracting info straight from params
    cam=params[:,:3]
    pose=params[:,3:72]
    shape=params[:,75:]
    """
    camera=tf.reshape(camera,[-1,1,3])
    transPoints=PointBatch[:,:,:2]+camera[:,:,1:]
    scaledPoints=transPoints*tf.squeeze(camera[:,:,0],axis=1)
    
    return scaledPoints
def lsp_STAR(joints):
    #batch size x 24 x 3
    #starresults.Jtr 
    #lsp-star=[(0,7),(1,4),(2,1),(3,0),(4,3),(5,6),(6,20),(7,18),(8,16),(9,15),(10,17),(11,19),(12,11),(13,14)]
    joints=tf.transpose(joints,perm=[0,2,1])
    H=np.ones((24,14))
    #the last joint (number 13) is one we might want to leave out. 
    b=[(7,4,1,0,3,6,20,18,16,15,17,19,11,14),(0,1,2,3,4,5,6,7,8,9,10,11,12,13)]
    H[b]=1
    J_new=tf.matmul(joints,H)
    J_new=tf.transpose(J_new,perm=[0,2,1])
    #untested, as of yet. 

    return J_new


