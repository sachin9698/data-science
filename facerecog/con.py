import cv2
import numpy as np

img = cv2.imread('sachin_1.jpg')
img2 = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imwrite('sachin_gray.jpg',gray)
# cv2.imshow('img',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(img2)
