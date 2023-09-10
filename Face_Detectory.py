import cv2
import os
import urllib3 as urllib
import numpy as np

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in
#img = cv2.imread('RDJ.png')
#0 for the  default camera
webcam = cv2.VideoCapture(0)


#for video file
# webcam = cv2.VideoCapture('(the same directory) file name')

### Iterate forever over frames
# try:
      
#     # creating a folder named data
#     if not os.path.exists('data'):
#         os.makedirs('data')
  
# # if not created then raise error
# except OSError:
#     print ('Error: Creating directory of data')
current = 0
while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x, y, w,  h) in face_coordinates:
        cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 255, 0),2)
        cv2.imwrite(f".\data\{current}.jpg",frame)
        current = current+1

    
    cv2.imshow('Clever Programmer Detector',frame)
    
    # writing the extracted images
    
    key= cv2.waitKey(0)
    if key==81 or key==113:
        break
    
print(current)
webcam.release()
cv2.destroyAllWindows()
# key = cv2.waitKey()



#Must convert to grayscale(make it black and white)
# grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw recrangles around the faces
# (x, y, w,  h) in face_coordinates[0] -- face one in the photo
# (x, y, w,  h) in face_coordinates[1] -- face two in the face
#for to detect all faces
# for (x, y, w,  h) in face_coordinates:
#     cv2.rectangle(img,(x, y),(x+w, y+h),(255, 255, 0),2)



# print(face_coordinates)


#open the photo
# cv2.imshow('Clever Programmer Detector',img)

#wait till we press a key and image can be shown
# cv2.waitKey()








print("Code Completed")

