import os
import tensorflow as tf
import numpy as np
import random
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from os.path import join

# def get_lsp(data_directory):
  
  # # load annotation matrices (hopefully)
 
  # ''' Note: file joints.mat is MATLAB data file with joint annotations in 3x14x10000 matrix ('joints')
  # with x and y locations and binary value indicating visbility of each joint '''
  
  # annotations = join(data_directory, 'joints.mat')
  # joints = sio.loadmat(annotations)['joints']
  # return joints 
  
  # # load images
  
  # images = sorted([i for i in glob(join(data_directory, 'images/*.jpg'))])

padding = 10
im_size = 224

index = 5

img = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/LSP/lsp_dataset/images/im09935.jpg"

jts = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/LSP/lsp_dataset/joints.mat"

img = mpimg.imread(img)

img = np.array(img)

print(np.shape(img))

imgplot = plt.imshow(img)
#annotations = join(jts, 'joints.mat')
joints = sio.loadmat(jts)['joints']

#print(joints)

ex = joints[:,:,9934]

#print(ex)

ex = ex[ex[:,2]>0]

#print(np.shape(ex))

x1, y1 = ex[:,0], ex[:,1]

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

bbox_min_x -= padding
bbox_min_y -= padding

bbox_max_x += padding
bbox_max_y += padding

# print(x1)
# print(y1)

img = img[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

print(np.shape(img))

plt.scatter(bbox_cx, bbox_cy, c='green')

plt.scatter(x1, y1)

plt.scatter(min_x, min_y, c='red')

plt.scatter(max_x, max_y, c='red')

plt.scatter(bbox_min_x, bbox_min_y, c='black')

plt.scatter(bbox_max_x, bbox_max_y, c='black')


jt = np.array(joints)

plt.show()

plt.clf()

imgplot = plt.imshow(img)

plt.show()




#print(np.shape(jt))

'''
1. Getting new joint coordinates
      translate current 
      (0,0) -> (bbox_min_x, bbox_min_y)
      add v. subtract?
2. Scaling (entire image)
      matplot function
      rescale joints
3. Edge case cropping operation
      if bbox extremes are outside image:
        fill in with zeros ?
4. Run!
      a. 100 images
            big problems? y/n
      b. everything!
5. Extend to MPII
'''

