import os
import tensorflow as tf
import numpy as np
import random
import math
from bumpy_dim_model import Generator, Discriminator
"""the doc where we will run all of this. we can use the 
main func to do this. I think that we can use functions outside
of the model to calculate losses. The structure of this is not
set in stone, I'm just getting the ball rolling. """
num_im_feats=2048
resNet=tf.keras.applications.ResNet50V2(classes=num_im_feats)
generator=Generator()
discriminator=Discriminator()
def reprojLoss():
    pass
def disLoss(disOut,):
    pass

def genLoss(disOut,):
    pass
def train(discriminator,generator,imageBatch,labelBatch):
    
    
    pass
 