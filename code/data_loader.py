import numpy as np
import tensorflow as tf

def load_joints(lsp_dir, mpii_dir, h36_dir, h36_actions):
    # Load LSP Data
    lsp_joints = read_joints(lsp_dir, False)
    # Load MPII Data
    mpii_joints = read_joints(mpii_dir, False)

    h36_joints = []
    for action in h36_actions:
        path = h36_dir + '/' + action
        action_joints = read_joints(path, True)
        h36_joints.append(action_joints)
    return lsp_joints, mpii_joints, h36_joints

def read_joints(dir, h36):
    f = open(dir + '/joints.txt', "r")
    contents = f.readlines()
    num_imgs = int(len(contents)/14)
    annots = np.zeros((num_imgs, 14, 3))

    if not h36:
        for i in range(num_imgs):
            for j in range(14):
                string = contents[i * 14 + j]
                split_str = np.array(string.split())
                split_str = split_str.astype(np.float)
                annots[i][j] = split_str
    else:
        for i in range(num_imgs):
            for j in range(14):
                string = contents[i * 14 + j]
                out = np.zeros(3)
                split_str = np.array(string.split())
                split_str = split_str.astype(np.float)
                out[0] = split_str[0]
                out[1] = split_str[1]
                out[2] = 1.0
                annots[i][j] = out
    return annots

def load_and_process_image(file_path):
    # Load image
    image = tf.io.decode_png(tf.io.read_file(file_path),channels=3)
    # Convert image to normalized float [0, 1]
    image = tf.cast(image, tf.float32)
    image=tf.keras.applications.resnet_v2.preprocess_input(image, data_format='channels_last')
    return image

def load_cmu(file_path):
    poses = np.load(file_path + '/poses_netruSMPL_CMU.npy')
    shapes = np.load(file_path + '/shapes_netruSMPL_CMU.npy')
    return poses, shapes

