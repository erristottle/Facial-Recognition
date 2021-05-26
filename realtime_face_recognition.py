# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:01:30 2021

@author: chris
"""

#Importing libraries
import cv2
import face_recognition


#Get the webcam #0 (the default one, 1, 2, etc. menas additional attached web cams)
webcam_video_stream = cv2.VideoCapture(0)


#Load sample image and extract face encoding 
modi_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\modi.jpg')
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\trump.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

chris_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\chris.jpg')
chris_face_encoding = face_recognition.face_encodings(chris_image)[0]

corey_image = face_recognition.load_image_file('C:\\Users\\chris\\Documents\\Learning\\Udemy\\Computer Vision - Face Recognition Quick Starter in Python\\code\\images\\corey2.jpg')
corey_face_encoding = face_recognition.face_encodings(corey_image)[0]


#Create array to save encodings
known_face_encodings = [modi_face_encoding, trump_face_encoding, chris_face_encoding, corey_face_encoding]

#Create another array to hold labels
known_face_names = ['Narendra Modi', 'Donald Trump', 'Chris Smith', 'Corey Smith']


#Initialize array variable to hold all images in the stream
all_face_locations  = []
all_face_encodings = []
all_face_names = []


while (True):
    
    #Get current frame
    ret, current_frame = webcam_video_stream.read()
    
    #Resize current frame so that the computer can process it faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    
    #Detect faces
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=1, model='hog')
    
    
    
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)


    #Loop through each face location and face encoding found in the image
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        #Split the tuple
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        right_pos = right_pos*4
        print("Found face at location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top_pos, left_pos, bottom_pos, right_pos))
    
    
        #See if the face matches any known faces
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        #Initialize face name to unknown
        name_of_person = 'Unknown Face'
        
        #If match was found in face encodings, use the first one
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
            
        #Draw rectangle around image (use current_frame because image_to_detect was turned blue in the load_image_file)
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0), 2)
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
    #Show image
    cv2.imshow('Identified Faces', current_frame)
    
    #Press q to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

#Release the stream and cam
#Close all OpenCV windows open
webcam_video_stream.release()
cv2.destroyAllWindows()