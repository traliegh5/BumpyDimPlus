import tensorflow as tf
import numpy as np
import cv2
import os

h36m_dir = ""



vid_cap = cv2.VideoCapture(h36m_dir)

frame_count = 0

while(True):

    cont, img = vid_cap.read()

    if cont:
        pass
    
    else:
        break


  
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
            print('Capturing --- '+name)
  
            # Extracting images and saving with name 
            cv2.imwrite(name, frames) 
            frame_count = frame_count + 1
        else:
            break
  
path = r"C:\Users\KIRA\Desktop\case study\sample-mp4-file.mp4"

img_extract(path)