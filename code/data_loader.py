import numpy as np
import tensorflow as tf

def load_joints(lsp_dir, mpii_dir, h36_dir):
    # Load LSP Data
    lsp_joints = read_joints(lsp_dir)
    # Load MPII Data
    mpii_joints = read_joints(mpii_dir)
    return lsp_joints, mpii_joints

def read_joints(dir):
    f = open(dir + '/joints.txt', "r")
    contents = f.readlines()
    num_imgs = int(len(contents)/14)
    annots = np.zeros((num_imgs, 14, 3))

    for i in range(num_imgs):
        for j in range(14):
            string = contents[i * 14 + j]
            split_str = np.array(string.split())
            split_str = split_str.astype(np.float)
            annots[i][j] = split_str
    return annots

def load_and_process_image(file_path):
    # Load image
    image = tf.io.decode_png(
    tf.io.read_file(file_path),
    channels=3)
    # Convert image to normalized float [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image=tf.keras.applications.resnet_v2.preprocess_input(image, data_format='channels_last')
    return image

def load_cmu(file_path):
    poses = np.load(file_path + '/poses_netruSMPL_CMU.npy')
    shapes = np.load(file_path + '/shapes_netruSMPL_CMU.npy')
    return poses, shapes

