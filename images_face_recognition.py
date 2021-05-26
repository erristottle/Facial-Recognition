# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:26:08 2021

@author: chris
"""

#Importing libraries
import cv2
import face_recognition

#Load image to detect
original_image = cv2.imread('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\trump-modi-unknown.jpg')

#Load sample image and extract face encoding 
modi_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\modi.jpg')
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\trump.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]


#Create array to save encodings
known_face_encodings = [modi_face_encoding, trump_face_encoding]

#Create another array to hold labels
known_face_names = ['Narendra Modi', 'Donald Trump']

#Load unknown image to identify faces
image_to_recognize = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\trump-modi-unknown.jpg')


#Find all faces and face encodings in the unknown image
all_face_locations = face_recognition.face_locations(image_to_recognize, model='hog')
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)


#Loop through each face location and face encoding found in the image
for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
    #Split the tuple
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("Found face at location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top_pos, left_pos, bottom_pos, right_pos))


    #See if the face matches any known faces
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    #Initialize face name to unknown
    name_of_person = 'Unknown Face'
    
    #If match was found in face encodings, use the first one
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
        
    #Draw rectangle around image (use original_image because image_to_detect was turned blue in the load_image_file)
    cv2.rectangle(original_image, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)

#Show image
cv2.imshow('Identified Faces', original_image)
