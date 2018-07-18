# import os
# import cv2
# import pickle
#
# # path=os.path.abspath(os.path.curdir)
# # labels={}
# # x=[]
# # y=[]
# # id=0
# # for(root,subdir,files) in os.walk(path):
# #     # print(subdir)
# #     for file in files:
# #         # print(file)
# #         pathf=os.path.join(root, file)
# #         pname=file.split('_',0)
# #         if pname in labels.keys():
#
# rec=cv2.face.LBPHFaceRecognizer_create()
import pickle
f=open('labeldata.pkl','rb')
l=pickle.load(f)
print(l)
