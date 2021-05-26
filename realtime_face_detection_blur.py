# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:01:30 2021

@author: chris
"""

#Importing libraries
import cv2
import face_recognition


#Get the webcam #0 (the default one, 1, 2, etc. menas additional attached web cams)
#webcam_video_stream = cv2.VideoCapture('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\modi.mp4')
webcam_video_stream = cv2.VideoCapture(0)

#Initialize array variable to hold all images in the stream
all_face_locations  = []



while (True):
    
    #Get current frame
    ret, current_frame = webcam_video_stream.read()
    
    #Resize current frame so that the computer can process it faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    
    #Detect faces
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')
    
    #Find face positions
    for index, current_face_location in enumerate(all_face_locations):
        #Split tuple
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        right_pos = right_pos*4
        
        print("Found face {} at location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(index + 1, top_pos, left_pos, bottom_pos, right_pos))
        
        #Slice faces from image
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        
        
        #Blur
        current_face_image = cv2.GaussianBlur(current_face_image, (99,99), 30)
        
        #Paste blur into the actual image
        current_frame[top_pos:bottom_pos, left_pos:right_pos] = current_face_image
        
        #Draw rectangle around image
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
        
    #Show webcam video
    cv2.imshow("Webcam Video", current_frame)
    
    #Press q to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#Release the stream and cam
#Close all OpenCV windows open
webcam_video_stream.release()
cv2.destroyAllWindows()