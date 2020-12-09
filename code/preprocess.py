import os
import tensorflow as tf
import numpy as np
import random
import math
import glob

def get_data(filepath, otherparams):
    "get_data method to be used to return training and testing data"
   
    return filepath

def get_moshed(pickles_directory):

    '''
    unpickles MoShed SMPL params
    '''
    
    all_pickles = sorted(glob.glob(join(pickles_directory, '*/*.pkl'), recursive=True))
    
    poses, shapes = [], []
    
    for p in all_pickles:
        with open(pickles, 'rb') as f:
            upd = pickle.load(f, encoding="latin-1")
            
        if 'poses' in upd.keys():
            poses.append(upd['poses'])
        else:
            poses.append(upd['new_poses'])
        
        num_poses = np.shape(upd['new_poses'])[0]
        # may need to reshape betas to (10,1) (CHECK later)
        shapes.append(np.tile(upd['betas'], num_poses))  
    

"This file can be used for all of our data preprocessing needs, getting things in the right shape, and all that."
