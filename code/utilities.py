import os
import tensorflow as tf
import numpy as np
import random
import math
import tensorflow_graphics as tfg 

"""this model can be for all intermediate processing methods, potentially. things like projection, rodriguez formula,
maybe smpl/STAR stuff? all of the math transformations we need to do can be located here. If this structure is unnecessary 
then feel free to do it another waydone
"""



def map(meshes):
    """it doesn't need to be done here, but we need to create texture maps. To do this,we need to project the mesh back into the images
    space, then convolve over it with a "bilinear sampling kernel" which estimates texture. we can look at differences 
    across texture maps of consecutive images as loss. """"

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
    scaledPoints=transPoints*camera[:,:,0]
    
   



    return scaledPoints


