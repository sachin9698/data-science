import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/papa/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)



i=1
while (i<100):
    name = str(i)
    ret, frame = video.read()
    print (ret,type(frame))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        simg = frame[y:y + h, x:x + w]
        cv2.imwrite('/home/papa/Data Science/sachin/sachin_'+name+'.jpg', simg)
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(154,124,10),2)


    cv2.imshow('video',frame)
    cv2.waitKey(1)
    i+=1


cv2.destroyAllWindows()
