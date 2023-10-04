# import cv2
# import numpy as np

# # Create a CascadeClassifier Object
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# # Reading the image as it is
# # img = cv2.imread("pic.jpeg")
# # assert not isinstance(img, type(None)), 'img not found'
# #img = cv2.imdecode(np.fromfile('/home/swasty/Desktop/Face rec/pic.jpeg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
# img = cv2.imread('pic.jpeg'.encode('utf-8', 'surrogateescape').decode('utf-8', 'surrogateescape'))
# print img.shape
# print img
# # Reading the image as gray scale image
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Search the co-ordintes of the image
# faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)
# for x,y,w,h in faces:
#     img = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)
# resized = cv2.resize(img, (int(img.shape[1]/7),int(img.shape[0]/7)))
# cv2.imshow("Gray", resized)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

import cv2
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('pic.jpeg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()