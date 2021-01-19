import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random
import math

"""this model can be for all intermediate processing methods, potentially. things like projection, rodriguez formula,
maybe smpl/STAR stuff? all of the math transformations we need to do can be located here. If this structure is unnecessary 
then feel free to do it another waydone
"""

def map(star_verts, star_faces, bary_map, image, camera):
    """
    Mapping function from mesh to texture
    star_verts Batch x 6890 x 3 - the coordinates of a star mesh
    star_faces num_faces x 3 - list of vertex indices that compose a face
    bary_map num_bary_pts x 3
    image Batch x 244 x 244 x 3
    """
    # Faces in mesh x 3 x 3
    faces_with_coords = tf.gather(star_verts, star_faces,axis=1) 

    # Bary points x 3 x 3 (For each uv point, the relevant 3x3 vert/coord arry for interpolation)
    num_bary_pts = bary_map.shape[0]
    bary_points = tf.gather(faces_with_coords, tf.dtypes.cast(bary_map[:,0], tf.int32), axis=1) 
    bary_weights = tf.concat([bary_map[:,1:], tf.reshape(1.0 - bary_map[:,1] - bary_map[:,2], [-1,1])],axis=1)
    bary_weights = tf.reshape(bary_weights, [-1, num_bary_pts,3,1])
    pts_interpolated = tf.math.reduce_sum(bary_points * bary_weights, axis=2)
    projected = orth_project(pts_interpolated, camera)

    image = tf.reshape(image, [-1, 224,224,3])
    sampled_pts = tfa.image.interpolate_bilinear(image, projected, "xy")
    return sampled_pts

def orth_project(PointBatch,camera):
    """
    These are the indices for extracting info straight from params
    cam=params[:,:3]
    pose=params[:,3:72]
    shape=params[:,75:]
    """
    camera=tf.reshape(camera,[-1,1,3])
    transPoints=PointBatch[:,:,:2]+camera[:,:,1:]
    scaledPoints=tf.squeeze(camera[:,:,0],axis=1)*transPoints
    
    return scaledPoints
def lsp_STAR(joints):
    
    J_new=tf.gather(joints,(7,4,1,0,3,6,20,18,16,15,17,19,11,14),axis=1) 

    return J_new
