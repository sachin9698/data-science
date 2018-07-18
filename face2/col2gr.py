import os
import cv2
import numpy as np
import pickle
from PIL import Image
path = os.path.abspath(os.path.curdir)

for root,subdir,files in os.walk(path):
    for file in files:
        fpath=os.path.join(root,file)
        if fpath.find('sachin')!=-1:
            img=cv2.imread(fpath)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(fpath,gray)            
