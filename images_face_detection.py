# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:26:08 2021

@author: chris
"""

#Importing libraries
import cv2
import face_recognition


#Load image to detect
image_to_detect = cv2.imread('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\trump-modi.jpg')


#Show image
cv2.imshow('test', image_to_detect)

#Detect number of faces
all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')
print("There are {} face(s) in this image".format(len(all_face_locations)))


#Find face positions
for index, current_face_location in enumerate(all_face_locations):
    #Split tuple
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    
    print("Found face {} at location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(index + 1, top_pos, left_pos, bottom_pos, right_pos))
    
    
    #Slice faces from image
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow('Face No:' + str(index+1), current_face_image)