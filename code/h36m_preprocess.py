# from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ["CDF_LIB"] = "~/CDF/lib"
import glob
import cdflib
import sys
import matplotlib.pyplot as plt
from os import mkdir
# from spacepy import pycdf


# Human3.6M directory
h36m_dir = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/Human3.6M/"

image_dir = os.path.join(h36m_dir, 'images/')

# if first time running, make image directory:
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

'''
Why do the joint schemes not agree ???
'''

images = []
joints= []

h36m_joints = {
    0: 'Hip',
    1: 'rHip', # confirmed
    2: 'rKnee', # confirmed
    3: 'rFoot', # confirmed--replace me
    6: 'lHip', # confirmed
    7: 'lKnee', # confirmed
    8: 'lFoot', # confirmed--replace me
    12: 'Spine',
    13: 'Neck', # confirmed
    14: 'Nose',
    15: 'Head', # confirmed
    17: 'lShoulder', # confirmed
    18: 'lElbow', # confirmed
    19: 'lWrist', # confirmed
    25: 'rShoulder', # confirmed
    26: 'rElbow', # confirmed
    27: 'rWrist', # confirmed
}

'''
LSP:
Right ankle, Right knee, Right hip, Left hip, Left knee, Left ankle,
Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow,
Left wrist, Neck, Head top
'''

h36m_indices_in_lsp = [3, 2, 1, 6, 7, 8, 27, 26, 25, 17, 18, 19, 13, 15]

lsp_to_h36m = {0: 3, 1: 2, 2: 1, 3: 6, 4: 7, 5: 8, 6: 27, 7: 26, 8: 25, 9: 17,
    10: 18, 11: 19, 12: 13, 13: 15}


training_subjects=['S1']
# training_subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
test_subject =  ['S2', 'S3', 'S4']

# NOTE: S10 removed due to privacy concerns



for subject in training_subjects:
    bbox_path = os.path.join(h36m_dir, subject, 'MySegmentsMat', 'ground_truth_bb')
    pose_path = os.path.join(h36m_dir, subject, 'MyPoseFeatures', 'D2_Positions')
    video_path = os.path.join(h36m_dir, subject, 'Videos')

    sequences = glob.glob(os.path.join(pose_path, '*.cdf'))
    np.sort(sequences)

    print(len(sequences))

    # start_index = 0
    # end_index = 25000

    slice = sequences[:]
    num_frames = len(slice)

    # num_joints = 14
    joints = []
    # joints_index = 0

    # for i in range(num_frames):

    #     i = slice[i]

    #     i_joints = np.zeros((num_joints, 2))

    #     for j in range(num_joints):
    #         x, y = i[0]['joint_pos'][lsp_to_h36m[j]]
    #         lsp_to_h36m[j]
    #         i_joints[j] = np.array([x, y])
    #     joints.append(i_joints)


    for seq in sequences:
        print(seq)
        seq_name = seq.split('/')[-1]
        bbox_file = seq_name.replace('.cdf', '.mat')
        video_file = seq_name.replace('.cdf', '.mp4')
        action, camera, _ = seq_name.split('.')
        action = action.replace(' ', '_')
        print('seq:', seq_name)

        # 2d poses
        poses = cdflib.CDF(seq)['Pose'][0]
        print('poses:', poses)
        poses = np.array(poses)
        print('poses shape:', np.shape(poses))

        vid_cap = cv2.VideoCapture(os.path.join(video_path, video_file))

        all_frames = np.shape(poses)[0]
        frame_curr = 0

        print('all frames:', all_frames)

        for fr in range(all_frames):

            print('fr:', fr)

            while(True):
                cont, frame = vid_cap.read()

                if cont:
                    # cv2.imshow('frame', frame)
                    imgplot = plt.imshow(frame)

                    if fr % 5 == 0:
                        annots = poses[frame_curr]

                        # Why does this only work with frame_curr ???

                    # print('annots shape:', annots.shape)
                    # print('annots:', annots)

                        annots = annots.reshape((-1,2))

                        # print('annots_reshaped:', annots)

                        x1 = annots[:,0]
                        y1 = annots[:,1]

                        lsp_x1 = x1[h36m_indices_in_lsp]
                        lsp_y1 = y1[h36m_indices_in_lsp] 

                        # print('lsp_x1:', lsp_x1)
                        # print('lsp_y1:', lsp_y1)

                        lsp_annots = annots[h36m_indices_in_lsp]

                        print('lsp_annots:', lsp_annots)

                        # hip_x = x1[0]
                        # hip_y = y1[0]

                        # find ankles or use feet?

                        # i = 17

                        # x_x = x1[i]
                        # y_x = y1[i]

                        # # plt.scatter(x1, y1, c='lime')
                        # plt.scatter(lsp_x1, lsp_y1, c='lime')
                        # # plt.scatter(x_x, y_x, c='lime')
                        # plt.show()

                       # sys.exit()

                        print('frame_curr:', frame_curr)
                        img_name = '%s_%s.%s_%06d.png' % (subject, action, camera, frame_curr+1)
                        img_file = os.path.join(image_dir, img_name)
                        print ('Creating...' + img_file) 

                        cv2.imwrite(img_file, frame)

                        #Save 14 Joints to annotation file
                        joint_file_name = image_dir + "/joints.txt"
                        f = open(joint_file_name, "a+")

                        for i in range(lsp_annots.shape[0]):
                            f.write(str(lsp_annots[i][0]) + ' ' + str(lsp_annots[i][1]) + '\n')
                        f.close()

                        images.append(img_name)
                        joints.append(lsp_annots)

                    frame_curr += 1

                else:
                    break

            vid_cap.release() 
            cv2.destroyAllWindows()

  
# Function which take path as input and extract images of the video 
def img_extract(path): 
      
    # Path to video file --- capture_image is the object which calls read
    capture_image = cv2.VideoCapture(path) 
    #keeping a count for each frame captured  
    frame_count = 0
  
    while (True): 
        #Reading each frame
        con,frames = capture_image.read() 
        #con will test until last frame is extracted
        if con:
            #giving names to each frame and printing while extracting
            name = str(frame_count)+'.jpg'
            print('Capturing --- '+ name)
  
            # Extracting images and saving with name 
            cv2.imwrite(name, frames) 
            frame_count = frame_count + 1
        else:
            break
  
# path = ""

# img_extract(path)