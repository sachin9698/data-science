import numpy as np
import getch
import cv2

face_cascade=cv2.CascadeClassifier('/home/papa/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

vid=cv2.VideoCapture(0)
while True:
    ret,img=vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        simg=img[y:y+h,x:x+w]
        cv2.imwrite('pic1.png',simg)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(19,139,245),2)


    cv2.imshow('img',img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
