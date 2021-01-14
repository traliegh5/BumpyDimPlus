import os
import tensorflow as tf
import numpy as np
import random
import math
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

from os.path import join


# def get_lsp(data_directory):
  
#   # load annotation matrices (hopefully)
 
#   ''' Note: file joints.mat is MATLAB data file with joint annotations in 3x14x10000 matrix ('joints')
#   with x and y locations and binary value indicating visbility of each joint '''
  
#   annotations = join(data_directory, 'joints.mat')
#   joints = sio.loadmat(annotations)['joints']
#   return joints 
  
#   # load images
  
#   images = sorted([i for i in glob(join(data_directory, 'images/*.jpg'))])

pad = 75
bbox_pad = 30
im_size = 224

'''LSP'''

start = time.time()

data_directory = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/LSP/lsp_dataset/"

jts = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/LSP/lsp_dataset/joints.mat"

'''MPII''' 

# data_directory = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/MPII/"

# jts = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"

images = sorted([i for i in glob.glob(join(data_directory, 'images/*.jpg'))])

joints = sio.loadmat(jts)['joints']

print(np.shape(joints))
print('joints:', joints)

index = 469

for i in images:

      img = i
      img = mpimg.imread(img)
      img = np.array(img)

      # print(np.shape(img))

      padded_img = np.pad(img, ((pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))

      # print(np.shape(padded_img))

      # imgplot = plt.imshow(padded_img)
      #annotations = join(jts, 'joints.mat')


      # complete joints matrix
      ex_all = joints[:,:,index]

      ex = ex_all[ex_all[:,2]>0]

      # print('ex:', ex)

      ex += pad

      # print('ex_pad:', ex)

      x1, y1 = ex[:,0], ex[:,1]

      # print('x1:', x1)
      # print('y1:', y1)

      min_x, min_y = np.amin(x1), np.amin(y1)
      max_x, max_y = np.amax(x1), np.amax(y1)

      bbox_w = max_x - min_x
      bbox_h = max_y - min_y
      bbox_cx, bbox_cy = max_x - bbox_w/2, max_y - bbox_h/2

      # print(bbox_w)
      # print(bbox_h)

      side_length = np.max([bbox_w, bbox_h])

      # print(side_length)

      bbox_min_x, bbox_min_y = int(np.floor(bbox_cx - side_length/2)), int(np.floor(bbox_cy - side_length/2))
      bbox_max_x, bbox_max_y = int(np.ceil(bbox_cx + side_length/2)), int(np.ceil(bbox_cy + side_length/2))

      if bbox_max_x - bbox_min_x > bbox_max_y - bbox_min_y:
            bbox_max_y += 1

      if bbox_max_x - bbox_min_x < bbox_max_y - bbox_min_y:
            bbox_max_x += 1

      bbox_min_x -= bbox_pad
      bbox_min_y -= bbox_pad

      bbox_max_x += bbox_pad
      bbox_max_y += bbox_pad

      # print(x1)
      # print(y1)
      # print(np.shape(img))
            
      padded_img = padded_img[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

      # print(np.shape(padded_img))

      # plt.scatter(bbox_cx, bbox_cy, c='lime')
      # plt.scatter(x1, y1, c='gold')
      # plt.scatter(min_x, min_y, c='fuchsia')
      # plt.scatter(max_x, max_y, c='fuchsia')
      # plt.scatter(bbox_min_x, bbox_min_y, c='aquamarine')
      # plt.scatter(bbox_max_x, bbox_max_y, c='aquamarine')


      jt = np.array(joints)

      # plt.show()
      # plt.clf()

      side_length = np.shape(padded_img)[0]
      padded_img = cv2.resize(padded_img, (224, 224))
      scale_factor = float(224.0/side_length)

      # print(scale_factor)

      # print(np.shape(img))

      # imgplot = plt.imshow(padded_img)

      x1 -= bbox_min_x 
      y1 -= bbox_min_y

      x1 = x1 * scale_factor
      y1 = y1 * scale_factor

      # plt.scatter(x1, y1)
      # plt.show()

      index += 1


end = time.time()
print("processing took %s minutes. nice!" %((end - start)/60.0))

'''
4. Run!
      a. 100 images
            big problems? y/n
      b. everything!
5. Extend to MPII
'''
