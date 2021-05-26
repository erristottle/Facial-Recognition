# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:26:08 2021

@author: chris
"""

#Importing libraries
import cv2
import face_recognition

image_to_recognize_path = 'C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\trump.jpg'

#Load image to detect
original_image = cv2.imread(image_to_recognize_path)

#Load sample image and extract face encoding 
modi_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\modi.jpg')
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\trump-2.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]


#Create array to save encodings
known_face_encodings = [modi_face_encoding, trump_face_encoding]

#Create another array to hold labels
known_face_names = ['Narendra Modi', 'Donald Trump']

#Load unknown image to identify faces
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encoding = face_recognition.face_encodings(image_to_recognize)[0]


#Find face distances
face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encoding)


for i, face_distance in enumerate(face_distances):
    print("The calculated face distance is {:.2} from sample image {}".format(face_distance, known_face_names[i]))
    print("The matching percentage is {} from sample image {}".format(round((1-float(face_distance))*100, 2), known_face_names[i]))

