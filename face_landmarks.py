# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:45:11 2021

@author: chris
"""

import face_recognition
from PIL import Image, ImageDraw

#Load image
face_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\chris.jpg')

#Find face landmarks
face_landmarks_list = face_recognition.face_landmarks(face_image)

print(face_landmarks_list)


#loop through all face landmarks in face landmarks list
for face_landmarks in face_landmarks_list:
    #Convert numpy array in PIL Object and create a draw object 
    pil_image = Image.fromarray(face_image)
    
    #convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)
    
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

#show the final image    
pil_image.show()

#save the image
pil_image.save('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\chris_landmarks.jpg')