import os
import tensorflow as tf
import numpy as np
import random
import math
import glob

'''
data pipelining based on "Scaling Deep Learning Systems" lecture ???
'''

'''
# Creates Dataset
directory_path = None

# Maps file contents => tensor
dataset = dataset.map(map_func=None)

# Loads data in batches
dataset = dataset.batch(batch_size)

# Prefetch next batch while GPU trains
dataset = dataset.prefetch(1)

# Iterate over dataset
for i, batch in enumerate(dataset)
    # process here!
'''

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
