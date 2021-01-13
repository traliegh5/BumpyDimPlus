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
import sys
import tensorlayer as tl

from os.path import join

img_train_list, ann_train_list, img_test_list, ann_test_list = tl.files.load_mpii_pose_dataset("/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/MPII/")

# print('img_train_list', ann_train_list[0])
# print('img_train_list', img_train_list[0])
# print('ann_test_list:', len(ann_train_list))

pad = 100
bbox_pad = 40
im_size = 224

start = time.time()

# '''
# MPII: 
# 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 
# 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 10 - r wrist, 12 - r shoulder, 
# 13 - l shoulder, 14 - l elbow, 15 - l wrist)" - `is_visible` - joint visibility
# '''

# ### which 10 is actually wrist ???

# '''
# LSP:
# Right ankle, Right knee, Right hip, Left hip, Left knee, Left ankle,
# Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow,
# Left wrist, Neck, Head top
# '''

lsp_to_mpii = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10, 7: 11, 8: 12, 9: 13,
    10: 14, 11: 15, 12: 8, 13: 9}

mpii_to_lsp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 8: 12, 9: 13, 10: 6, 11: 7, 
    12: 8, 13: 9, 14: 10, 15: 11}

num_joints = 14
slice = ann_train_list[350:360]
num_images = len(slice)

# print(slice)

joints = []

joints_index = 0

for i in slice:

    print(i)
    print('i[0]:', i[0])

    print('joint_pos:', i[0]['joint_pos'])
    print('vis:', i[0]['is_visible'])

    # print(i)

    i_joints = np.zeros((num_joints, 3))
    # print(joints)

    # for j in range(num_joints):
    #     if j in i[0]['joint_pos']:
    #         x, y = i[0]['joint_pos'][j]
    #         vis = i[0]['is_visible'][str(j)]
    #         if j in mpii_to_lsp:
    #             joints[mpii_to_lsp[j]] = np.array([x, y, vis])

    print(joints)

    for j in range(num_joints):
        if lsp_to_mpii[j] in i[0]['joint_pos']:
            x, y = i[0]['joint_pos'][lsp_to_mpii[j]]
            vis = i[0]['is_visible'][str(lsp_to_mpii[j])]
            i_joints[j] = np.array([x, y, vis])
            # print(j)
            # print(lsp_to_mpii[j])
            # print(i_joints[j])


    # print(i_joints)

    joints.append(i_joints)
    # joints_index += 1

# tf.convert_to_tensor(joints)

# print(tf.shape(joints))

# print(joints)

# tf.reshape(joints, [num_joints, 3, num_images])

# print(tf.shape(joints))

# print(joints)


index = 0

for i in img_train_list[350:360]:


# i = img_train_list[11]

    img = i
    img = mpimg.imread(img)
    img = np.array(img)
    # print(np.shape(img))
    padded_img = np.pad(img, ((pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))
    # print(np.shape(padded_img))
    imgplot = plt.imshow(padded_img)

    # print(joints)
    # print(np.shape(joints))

    ex_all = joints[index]

    np.reshape(ex_all, (num_joints, 3))

    # print(ex_all)


    # ex = ex_all

    ex = ex_all[ex_all[:,2]>0]

    print(ex_all[13])

    ex = np.vstack((ex, ex_all[13]))

    print(ex)

    ex += pad

    print(ex)

    #print(np.shape(ex))

    # x1, y1 = ex[:,0], ex[:,1]

    x1, y1 = ex[:,0], ex[:,1]

    print('x1:', x1)
    print('y1:', y1)


    min_x, min_y = np.amin(x1), np.amin(y1)
    max_x, max_y = np.amax(x1), np.amax(y1)

    print('max_x:', max_x)
    print('max_y:', max_y)

    print('min_x:', min_x)
    print('min_y:', min_y)  

    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    print('bbox_w:', bbox_w)
    print('bbox_h:', bbox_h)

    bbox_cx, bbox_cy = max_x - bbox_w/2, max_y - bbox_h/2

    print('bbox_cx:', bbox_cx)
    print('bbox_cy:', bbox_cy)

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

    padded_img = padded_img[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

    print(np.shape(padded_img))

    plt.scatter(bbox_cx, bbox_cy, c='lime')

    plt.scatter(x1, y1, c='gold')

    plt.scatter(min_x, min_y, c='fuchsia')

    plt.scatter(max_x, max_y, c='fuchsia')

    plt.scatter(bbox_min_x, bbox_min_y, c='aquamarine')

    plt.scatter(bbox_max_x, bbox_max_y, c='aquamarine')


    jt = np.array(joints)

    plt.show()

    plt.clf()

    side_length = np.shape(padded_img)[0]

    padded_img = cv2.resize(padded_img, (224, 224))

    scale_factor = float(224.0/side_length)

    print(scale_factor)

    # print(np.shape(img))

    imgplot = plt.imshow(padded_img)

    x1 -= bbox_min_x 
    y1 -= bbox_min_y

    x1 = x1 * scale_factor
    y1 = y1 * scale_factor

    plt.scatter(x1, y1)

    plt.show()

    index += 1


    '''
    The fallen
    '''
        
    # plt.show()

    # # mpii_to_lsp = [0,1,2,3,4,5,10,11,12,13,14,15,8,9]


        

    # potenitally loop if multiple individuals
    # for p in i[]



end = time.time()
print("processing took %s minutes. nice!" %((end - start)/60.0))









# # grab annotations (1 for now, num_imgs later)
# for i in range(1):
#     anno_list = annots.annolist[i]
#     people = annots.single_person
#     joints = anno_list.annorect
#     if not isinstance(joints, np.ndarray):
#         joints = np.array([joints])

#     print(joints)
#     print(anno_list.annorect)

#     for p in range(len(people)):
#         person_id = annots.single_person[p]
#         print(person_id)

#         # print(joints[person_id - 1])

#         joint = joints[0][person_id-1]

#         print(joint)
#         assert ('annopoints' in joint._fieldnames)

#         person_annots = joints[person_id-1].annopoints.point

#         # person_annots = joints[person_id - 1]






# # p = f_a.objpos.x
# # print(p)

# # f_a_j = ri.annopoints.point
# # print(f_a_j)

# sys.exit()


# r_id = annots[3]

# print(r_id)


# # print(annots)



# for i in images:

#       img = i

#       img = mpimg.imread(img)

#       img = np.array(img)

#       print(np.shape(img))

#       padded_img = np.pad(img, ((pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))

#       print(np.shape(padded_img))

#       imgplot = plt.imshow(padded_img)
#       #annotations = join(jts, 'joints.mat')

#       ex = joints[:,:,index]

#       #print(ex)

#       ex = ex[ex[:,2]>0]
#       ex += pad

#       #print(np.shape(ex))

#       x1, y1 = ex[:,0], ex[:,1]

#       min_x, min_y = np.amin(x1), np.amin(y1)
#       max_x, max_y = np.amax(x1), np.amax(y1)

#       bbox_w = max_x - min_x
#       bbox_h = max_y - min_y
#       bbox_cx, bbox_cy = max_x - bbox_w/2, max_y - bbox_h/2

#       # print(bbox_w)
#       # print(bbox_h)

#       side_length = np.max([bbox_w, bbox_h])

#       # print(side_length)

#       bbox_min_x, bbox_min_y = int(np.floor(bbox_cx - side_length/2)), int(np.floor(bbox_cy - side_length/2))
#       bbox_max_x, bbox_max_y = int(np.ceil(bbox_cx + side_length/2)), int(np.ceil(bbox_cy + side_length/2))

#       if bbox_max_x - bbox_min_x > bbox_max_y - bbox_min_y:
            
#             bbox_max_y += 1

#       if bbox_max_x - bbox_min_x < bbox_max_y - bbox_min_y:
            
#             bbox_max_x += 1

#       # bbox_min_x += 2*pad
#       # bbox_min_y += 2*pad

#       bbox_min_x -= bbox_pad
#       bbox_min_y -= bbox_pad

#       bbox_max_x += bbox_pad
#       bbox_max_y += bbox_pad

#       # print(x1)
#       # print(y1)



#       # print(np.shape(img))
            
#       padded_img = padded_img[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

#       print(np.shape(padded_img))

#       plt.scatter(bbox_cx, bbox_cy, c='lime')

#       plt.scatter(x1, y1, c='gold')

#       plt.scatter(min_x, min_y, c='fuchsia')

#       plt.scatter(max_x, max_y, c='fuchsia')

#       plt.scatter(bbox_min_x, bbox_min_y, c='aquamarine')

#       plt.scatter(bbox_max_x, bbox_max_y, c='aquamarine')


#       jt = np.array(joints)

#       plt.show()

#       plt.clf()

#       side_length = np.shape(padded_img)[0]

#       padded_img = cv2.resize(padded_img, (224, 224))

#       scale_factor = float(224.0/side_length)

#       print(scale_factor)

#       # print(np.shape(img))

#       imgplot = plt.imshow(padded_img)

#       x1 -= bbox_min_x 
#       y1 -= bbox_min_y

#       x1 = x1 * scale_factor
#       y1 = y1 * scale_factor

#       plt.scatter(x1, y1)

#       plt.show()

#       index += 1


# toc = time.perf_counter()

