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
from skimage import io, img_as_ubyte, img_as_float32
from os.path import join

start_index = 6468 + 41
end_index = 6468 + 42

# image_dir = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/MPII/"
image_dir = "D://Brown//Senior//CSCI_1470//FINAL//MPII"

img_train_list, ann_train_list, img_test_list, ann_test_list = tl.files.load_mpii_pose_dataset(image_dir)
print("GOT MPII FROM TENSORLAYER")
# print('img_train_list', ann_train_list[0])
# print('img_train_list', img_train_list[0])
# print('ann_test_list:', len(ann_train_list))

missing_imgs = ['040348287.jpg', '002878268.jpg']
pad = 500
bbox_pad = 40
im_size = 224

np.set_printoptions(suppress=True)

start = time.time()

'''
MPII: 
0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 
7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 10 - r wrist, 12 - r shoulder, 
13 - l shoulder, 14 - l elbow, 15 - l wrist)" - `is_visible` - joint visibility
'''

# 10 is actually wrist, 11 is elbow

'''
LSP:
Right ankle, Right knee, Right hip, Left hip, Left knee, Left ankle,
Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow,
Left wrist, Neck, Head top
'''

lsp_to_mpii = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10, 7: 11, 8: 12, 9: 13,
    10: 14, 11: 15, 12: 8, 13: 9}

mpii_to_lsp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 8: 12, 9: 13, 10: 6, 11: 7, 
    12: 8, 13: 9, 14: 10, 15: 11}

num_joints = 14
# slice = ann_train_list[5600:5610]
slice = ann_train_list[start_index:end_index]
num_images = len(slice)

joints = []
joints_index = 0
no_joints = []
joints_vis = []

for i in range(num_images):

    # print(i)

    if slice[i] != []:

        vis_joints_count = 0
        i = slice[i]

        # print('i[0]:', i[0])

        # print('joint_pos:', i[0]['joint_pos'])
        # print('vis:', i[0]['is_visible'])

        i_joints = np.zeros((num_joints, 3))

        for j in range(num_joints):
            if lsp_to_mpii[j] in i[0]['joint_pos']:
                x, y = i[0]['joint_pos'][lsp_to_mpii[j]]
                vis = i[0]['is_visible'][str(lsp_to_mpii[j])]
                if vis == 1:
                    vis_joints_count += 1
                i_joints[j] = np.array([x, y, vis])
        joints.append(i_joints)
        joints_vis.append(vis_joints_count)

    elif slice[i] == []:
        no_joints.append(i)

# print(no_joints)
# print('joints:', joints)
# # print(joints_vis)
# print(np.shape(joints))
#print(joints)

# joints = np.reshape(joints, (num_joints, 3))

# print(np.shape(joints))
# print(joints)
# # print(np.shape(joints_vis))


# img_slice = img_train_list[5600:5610]
img_slice = img_train_list[start_index:end_index]

index = 0
too_few_visible = 0
print("NUM_IMAGES: ", num_images)
for i in range(num_images):

    if i in no_joints:
        print('Passed, Unnannotated: ', i)
        pass
    elif joints_vis[index] < 5:
        print('Passed, Too Few Visible Joints: ', i)
        print('Visible Joints: ', joints_vis[index])
        index += 1
        too_few_visible += 1
        pass
    elif img_slice[i][len(img_slice[i]) - 13 : len(img_slice[i])] in missing_imgs:
        print('Passed, Missing Image: ', i)
        index += 1
        too_few_visible += 1
        pass
    else:
        img = img_slice[i]
        img = mpimg.imread(img)
        img = np.array(img)
        padded_img = np.pad(img, ((pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))

        ex_all = joints[index]

        # np.reshape(ex_all, (num_joints, 3))

        # print(ex_all)

        # if joints_vis[index] > 4:
        ex = ex_all[ex_all[:,2] > 0.5]
        if img_slice[i][len(img_slice[i]) - 13 : len(img_slice[i])] == '063800324.jpg':
            print('ex:', ex)
        ex = np.vstack((ex, ex_all[13]))
        if img_slice[i][len(img_slice[i]) - 13 : len(img_slice[i])] == '063800324.jpg':
            print('ex:', ex)
        ex += pad

        # else:
        #     ex = ex_all

        # print(ex)

        x1, y1 = ex[:,0], ex[:,1]

        min_x, min_y = np.amin(x1), np.amin(y1)
        max_x, max_y = np.amax(x1), np.amax(y1)

        # print('max_x:', max_x)
        # print('max_y:', max_y)

        # print('min_x:', min_x)
        # print('min_y:', min_y)  

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        # print('bbox_w:', bbox_w)
        # print('bbox_h:', bbox_h)

        bbox_cx, bbox_cy = max_x - bbox_w/2, max_y - bbox_h/2

        # print('bbox_cx:', bbox_cx)
        # print('bbox_cy:', bbox_cy)

        side_length = np.max([bbox_w, bbox_h])

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

        # print(np.shape(padded_img))
        # print("MIN: ", bbox_min_y, ', ',bbox_min_x)
        # print("MAX: ", bbox_max_y, ', ',bbox_max_x)
        
        #     imgplot = plt.imshow(padded_img)
        #     plt.scatter(bbox_cx, bbox_cy, c='lime')
        #     plt.scatter(x1, y1, c='gold')
        #     # plt.scatter(min_x, min_y, c='fuchsia')
        #     # plt.scatter(max_x, max_y, c='fuchsia')
        #     plt.scatter(bbox_min_x, bbox_min_y, c='aquamarine')
        #     plt.scatter(bbox_max_x, bbox_max_y, c='aquamarine')
        #     plt.show()
        padded_img = padded_img[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

        # print(np.shape(padded_img))

        # plt.show()
        # plt.clf()

        side_length = np.shape(padded_img)[0]
        padded_img = cv2.resize(padded_img, (224, 224))
        scale_factor = float(224.0/side_length)

        # print(scale_factor)

        #imgplot = plt.imshow(padded_img)
        image_path = image_dir + '/cropped_mpii/' + str(index - too_few_visible) + '.png'
        io.imsave(image_path, img_as_ubyte(padded_img.copy()))
        print("SAVED: ", image_path)
        x1 -= bbox_min_x 
        y1 -= bbox_min_y

        x1 = x1 * scale_factor
        y1 = y1 * scale_factor

        #Save 14 Joints to annotation file
        joint_file_name = image_dir + "/joints.txt"
        f = open(joint_file_name, "a+")

        ex_all[:,0] += pad
        ex_all[:,1] += pad
        ex_all[:,0] = (ex_all[:,0] - bbox_min_x) * scale_factor
        ex_all[:,1] = (ex_all[:,1] - bbox_min_y) * scale_factor
        for i in range(ex_all.shape[0]):
            f.write(str(ex_all[i][0]) + ' ' + str(ex_all[i][1]) + ' ' + str(ex_all[i][2]) + '\n')
        f.close()

        #plt.scatter(ex_all[:,0], ex_all[:,1])
        #plt.show()

        index += 1

# file = open("joints.txt", "w+")

# # Save joints in text file 
# joint_file_name = image_dir + '/annotations/' + "joints.txt"
# np.savetxt(joint_file_name, joints)

# # Displaying the contents of the text file 
# content = np.loadtxt('joints.txt') 
# print("\nContent in joints.txt:\n", content) 

end = time.time()
print("processing took %s minutes. nice!" %((end - start)/60.0))
