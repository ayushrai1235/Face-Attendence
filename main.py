import pickle
import cv2
import os
import cvzone
import numpy as np
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
from datetime import datetime

cred = credentials.Certificate("serviceAccountkey (2).json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-attendance-realtime-bc190-default-rtdb.firebaseio.com/",
    'storageBucket': "face-attendance-realtime-bc190.appspot.com"
})

bucket = storage.bucket()


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgbg = cv2.imread('Resources/background.png')


foldermodepath = 'Resources/Modes'
modepathlist = os.listdir(foldermodepath)
imgModelist = []
imgstud = []

for path in modepathlist:
    imgModelist.append(cv2.imread(os.path.join(foldermodepath,path)))

# loading encodings
file = open("Encodefile.p","rb")
encodeListKnownwithname = pickle.load(file)
encodeListKnown, studentName = encodeListKnownwithname
# print(studentName)
modeType = 0
count = 0
name = -1
while True:
    success, img = cap.read()


    imgs = cv2.resize(img,(0,0),None,0.25, 0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facecur = face_recognition.face_locations(imgs)
    encodecur = face_recognition.face_encodings(imgs , facecur)

    imgbg[162:162+480,55:55+640] = img
    imgbg[44:44+633,808:808+414] = imgModelist[modeType]

    for encface, (y1,x2,y2,x1) in zip(encodecur,facecur):
        matches = face_recognition.compare_faces(encodeListKnown,encface)
        facename = face_recognition.face_distance(encodeListKnown, encface)
        # print("Matches",matches)
        # print(facename)


        matchindex = np.argmin(facename)
        # print("match name :",matchindex)
        # print(f"Face Location:{(x,y,w,h)}")

        if matches[matchindex]:
            # print("FACE DETECTED")
            # print(studentName[matchindex])

            w = x2-x1
            h = y2-y1
            x1= 55 + x1
            y1 = 162 + y1
            bbox = (x1, y1, w , h)
            imgbg = cvzone.cornerRect(imgbg, bbox, rt=0)

            name = studentName[matchindex]

            if count == 0:
                count = 1
                modeType = 1

    if count!= 0:

        if count == 1:

            studentInfo = db.reference(f'Students/{name}').get()
            print(studentInfo)
            #getting images from database
            blob = bucket.get_blob(f'Images/{name}.png')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgstud = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            #Update data for attendence
            datetimeobj = datetime.strptime(studentInfo['last_attendence_time'],
                                           "%Y-%m-%d %H:%M:%S")
            secondselaspes = (datetime.now()-datetimeobj).total_seconds()
            print(secondselaspes)
            if secondselaspes > 60:
                ref = db.reference(f'Students/{name}')
                studentInfo['total_attendance'] += 1
                ref.child('total_attendance').set(studentInfo['total_attendance'])
                ref.child('last_attendence_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                modeType = 3
                count = 0
                imgbg[44:44 + 633, 808:808 + 414] = imgModelist[modeType]

        if modeType != 3:

            if 10<count<20:
                modeType = 2

            imgbg[44:44 + 633, 808:808 + 414] = imgModelist[modeType]

            if count<=10:

                cv2.putText(imgbg,str(studentInfo['total_attendance']),(861,125),
                            cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
                cv2.putText(imgbg, str(studentInfo['major']), (1006,550),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgbg, str(studentInfo['id']), (1006, 493),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgbg, str(studentInfo['year']), (1025, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100,100,100), 1)
                cv2.putText(imgbg, str(studentInfo['starting_year']), (1125, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100,100,100), 1)

                (w,h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1,1)
                offset = (414 - w) // 2
                cv2.putText(imgbg, str(studentInfo['name']), (808 + offset, 445),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (55, 55, 55), 1)

                imgbg[175:175+216,909:909+216] = imgstud

        count+=1

        if count>=20:
            count = 0
            modeType = 0
            studentInfo = []
            imgstud = []
            imgbg[44:44 + 633, 808:808 + 414] = imgModelist[modeType]


    cv2.imshow("face attendance", imgbg)
    cv2.waitKey(1)
