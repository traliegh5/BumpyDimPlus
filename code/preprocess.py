import os
import tensorflow as tf
import numpy as np
import random
import math
import glob
import pickle

from os.path import join

# def get_data(filepath, otherparams):
#     "get_data method to be used to return training and testing data"
   
#     return filepath

# def get_moshed(pickles_directory):

pickles_directory = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/neutrMosh/neutrSMPL_CMU/"

'''
unpickles MoShed SMPL params
'''

all_pickles = sorted([pkl for pkl in glob.glob(join(pickles_directory, '*/*.pkl'), recursive=True)])

# print(all_pickles)

poses, shapes = [], []

for p in all_pickles:
    with open(p, 'rb') as f:
        upd = pickle.load(f, encoding="latin-1")
        # print(upd)
        
    if 'poses' in upd.keys():
        poses.append(upd['poses'])
        print(np.shape(upd['poses']))
        num_poses = np.shape(upd['poses'])[0]
        # print(poses)
    elif 'new_poses' in upd.keys():
        poses.append(upd['new_poses'])
        num_poses = np.reshape(upd['new_poses'])[0]
        # print(poses)
    print('upbs:', upd['betas'].shape)
    betas = np.reshape(upd['betas'], (10, 1))
    print('rupbs:', betas.shape) 
    betas = np.tile(upd['betas'], num_poses)
    shapes.append(betas) 
    print('tb:', betas)
    print('tb shape:', betas.shape)

    print('pkl', p)
    # print('poses', len(poses))
    # print('shapes', len(shapes))

poses = np.vstack(poses)
# shapes = np.vstack(shapes)
h_shapes = np.hstack(shapes).T

print(poses.shape)
# print(shapes.shape)
print(h_shapes.shape)



# if __name__ == '__main__':
#     main()
    

"This file can be used for all of our data preprocessing needs, getting things in the right shape, and all that."
