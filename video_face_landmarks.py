# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:45:11 2021

@author: chris
"""

import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageDraw

#Get the webcam #0 (the default one, 1, 2, etc. menas additional attached web cams)
webcam_video_stream = cv2.VideoCapture('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\modi.mp4')
#webcam_video_stream = cv2.VideoCapture(0)

#Initialize array variable to hold all images in the stream
all_face_locations  = []



while (True):
    
    #Get current frame
    ret, current_frame = webcam_video_stream.read()
    
    #Resize current frame so that the computer can process it faster
    #current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    #Find face landmarks
    face_landmarks_list = face_recognition.face_landmarks(current_frame)
    
    #print(face_landmarks_list)
    
    #Convert numpy array in PIL Object and create a draw object 
    pil_image = Image.fromarray(current_frame)
    
    #convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)
    
    #loop through all face landmarks in face landmarks list
    for index in range(len(face_landmarks_list)):
        for face_landmarks in face_landmarks_list:
            
            
            #join each face landmark points
            d.line(face_landmarks['chin'],fill=(255,255,255), width=2)
            d.line(face_landmarks['left_eyebrow'],fill=(255,255,255), width=2)
            d.line(face_landmarks['right_eyebrow'],fill=(255,255,255), width=2)
            d.line(face_landmarks['nose_bridge'],fill=(255,255,255), width=2)
            d.line(face_landmarks['nose_tip'],fill=(255,255,255), width=2)
            d.line(face_landmarks['left_eye'],fill=(255,255,255), width=2)
            d.line(face_landmarks['right_eye'],fill=(255,255,255), width=2)
            d.line(face_landmarks['top_lip'],fill=(255,255,255), width=2)
            d.line(face_landmarks['bottom_lip'],fill=(255,255,255), width=2)
    
    #convert PIL image to RGB to show in opencv window    
    rgb_image = pil_image.convert('RGB') 
    rgb_open_cv_image = np.array(pil_image)
    
    # Convert RGB to BGR 
    bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
    bgr_open_cv_image = bgr_open_cv_image[:, :, ::-1].copy()

    #showing the current face with rectangle drawn
    cv2.imshow("Webcam Video",bgr_open_cv_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the stream and cam
#Close all OpenCV windows open
webcam_video_stream.release()
cv2.destroyAllWindows()
