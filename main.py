import pickle

import cv2
import os
import cvzone
import numpy as np
import face_recognition

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

while True:
    success, img = cap.read()
    imgbg[162:162+480,55:55+640] = img
    imgbg[44:44+633,808:808+414] = imgModelist[3]

    imgs = cv2.resize(img,(0,0),None,0.25, 0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facecur = face_recognition.face_locations(imgs)
    encodecur = face_recognition.face_encodings(imgs,facecur)


    for encface, faceloc in zip(encodecur,facecur):
        matches = face_recognition.compare_faces(encodeListKnown,encface)
        facename = face_recognition.face_distance(encodeListKnown, encface)
        # print("Matches",matches)
        # print(facename)

        matchindex = np.argmin(facename)
        # print("match name :",matchindex)

        if matches[matchindex]:
            # print("FACE DETECTED")
            # print(studentName[matchindex])

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            w = x2-x1
            h = y2-y1
            x1=55 + x1
            y1 = 162 + y1
            bbox = (x1, y1, w , h)
            imgbg = cvzone.cornerRect(imgbg, bbox, rt=0)







    cv2.imshow("face attendance", imgbg)
    cv2.waitKey(1)