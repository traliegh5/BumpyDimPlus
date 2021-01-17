# from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ["CDF_LIB"] = "~/CDF/lib"
import glob
import cdflib
from os import mkdir
# from spacepy import pycdf

h36m_dir = "/Users/annaswanson/Desktop/Deep Learning/Final Project/Data/Human3.6M/"
image_dir = os.path.join(h36m_dir, 'images/')


'''
Why do the joint schemes not agree ???
'''

h36m_joints = {
    0: 'Hip',
    1: 'rHip',
    2: 'rKnee',
    3: 'rFoot',
    6: 'lHip',
    7: 'lKnee',
    8: 'lFoot',
    12: 'Spine',
    13: 'Neck',
    14: 'Nose',
    15: 'Head',
    17: 'lShoulder',
    18: 'lElbow',
    19: 'lWrist',
    25: 'rShoulder',
    26: 'rElbow',
    27: 'rWrist',
}

'''
LSP:
Right ankle, Right knee, Right hip, Left hip, Left knee, Left ankle,
Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow,
Left wrist, Neck, Head top
'''

h36m_in_lsp = [3, 2, 1, 6, 7, 8, 27, 26, 18, 19, 13, 15]


training_subjects=['S1']
# training_subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
test_subject =  ['S2', 'S3', 'S4']

# NOTE: S10 removed due to privacy concerns

for subject in training_subjects:
    bbox_path = os.path.join(h36m_dir, subject, 'MySegmentsMat', 'ground_truth_bb')
    pose_path = os.path.join(h36m_dir, subject, 'MyPoseFeatures', 'D3_Positions_mono')
    video_path = os.path.join(h36m_dir, subject, 'Videos')

    sequences = glob.glob(os.path.join(pose_path, '*.cdf'))
    np.sort(sequences)

    for seq in sequences:
        print(seq)
        seq_name = seq.split('/')[-1]
        bbox_file = seq_name.replace('.cdf', '.mat')
        video_file = seq_name.replace('.cdf', '.mp4')
        action, camera, _ = seq_name.split('.')
        action = action.replace(' ', '.')
        print('seq:', seq_name)

        # 3d poses (from ???)
        poses = cdflib.CDF(seq)['Pose'][0]
        # print('poses:', poses)

        vid_cap = cv2.VideoCapture(os.path.join(video_path, video_file))

        all_frames = np.shape(poses)[0]
        frame_curr = 0

        for fr in range(all_frames):

            while(True):
                cont, frame = vid_cap.read()

                if cont:
                    cv2.imshow('frame', frame)

                    if fr % 5 == 0:
                        print('frame_curr:', frame_curr)
                        img_name = '%s_%s.%s_%06d.png' % (subject, action, camera, frame_curr+1)
                        img_file = os.path.join(image_dir, img_name)
                        print ('Creating...' + img_file) 

                        cv2.imwrite(img_file, frame)

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