# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman
import sys, os
sys.path.append('D:\\Brown\\Senior\\CSCI_1470\\sampl\\STAR')

from star.tf.star import STAR
import tensorflow as tf
import numpy as np
batch_size = 10
gender = 'male'
star = STAR()

trans = tf.constant(np.zeros((1,3)),dtype=tf.float32)
pose = tf.constant(np.zeros((1,72)),dtype=tf.float32)
betas = tf.constant(np.zeros((1,10)),dtype=tf.float32)
m = star(pose,betas,trans)

outmesh_path = './test_smpl.obj'
with open( outmesh_path, 'w') as fp:
    for v in m[0]:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in star.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print('..Output mesh saved to: ', outmesh_path)