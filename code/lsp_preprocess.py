import os
import tensorflow as tf
import numpy as np
import random
import math
import scipy.io as sio

from os.path import join #, exists (p cut)

def get_lsp(data_path):
  
  # load annotation matrices (hopefully)
  
  ''' Note: file joints.mat is MATLAB data file with joint annotations in 3x14x10000 matrix ('joints')
  with x and y locations and binary value indicating visbility of each joint '''
  
  annotations = join(data_path, 'joints.mat')
  joints = sio.loadmat(annotations)['joints']
  
  
                         
