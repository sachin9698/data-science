import pandas as pd
import numpy as np
import cv2
img=cv2.imread('pixel2.jpeg')
# print(img.shape)
img2=np.ones([400,400,3])
img3=np.zeros([400,400,3])
cv2.imshow('hello',img)
cv2.waitKey(0)
