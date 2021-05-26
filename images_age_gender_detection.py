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
#cv2.imshow('test', image_to_detect)

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
    
    
    #The 'AGE_GENDER_MODEL_MEAN_VALUES' calculated by using numpy.mean()
    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    #Create blob of current face slice
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
    
    #Declare gender labels and model path files
    gender_label_list = ['Male', 'Female']
    gender_protext = 'C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\dataset\\gender_deploy.prototxt'
    gender_caffemodel = 'C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\dataset\\gender_net.caffemodel'
    #Create model from files and provide blob as input
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    #Get gender predictions
    gender_predictions = gender_cov_net.forward()
    gender = gender_label_list[gender_predictions[0].argmax()]
    
    #Declare age labels and model path files
    age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age_protext = 'C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\dataset\\age_deploy.prototxt'
    age_caffemodel = 'C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\dataset\\age_net.caffemodel'
    #Create model from files and provide blob as input
    age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
    age_cov_net.setInput(current_face_image_blob)
    #Get age predictions
    age_predictions = age_cov_net.forward()
    age = age_label_list[age_predictions[0].argmax()]
    
    
    #Draw rectangle around image
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender +' '+ age+'yrs', (left_pos,bottom_pos), font, 0.5, (0,255,0),1)
   
#Show webcam video
cv2.imshow("Age and Gender", image_to_detect)