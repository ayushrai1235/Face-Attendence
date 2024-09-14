import pickle

import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgbg = cv2.imread('Resources/background.png')


foldermodepath = 'Resources/Modes'
modepathlist = os.listdir(foldermodepath)
imgModelist = []
for path in modepathlist:
    imgModelist.append(cv2.imread(os.path.join(foldermodepath,path)))

# loading encodings
file = open("Encodefile.p","rb")
encodeListKnownwithname = pickle.load(file)
encodeListKnown, studentName = encodeListKnownwithname
# print(studentName)

liveencodingknown

while True:
    success, img = cap.read()

    imgs = cv2.resize(img,(0,0),None,0.25, 0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    imgbg[162:162+480,55:55+640] = img
    imgbg[44:44+633,808:808+414] = imgModelist[0]


    cv2.imshow("face attendance",imgbg)
    cv2.waitKey(1)