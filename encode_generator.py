import cv2
import pickle
import os
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountkey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-attendance-realtime-bc190-default-rtdb.firebaseio.com/",
    'storageBucket': "face-attendance-realtime-bc190.appspot.com"
})



folderpath = 'Images'
pathlist = os.listdir(folderpath)
imglist = []
studentName = []
for path in pathlist:
    imglist.append(cv2.imread(os.path.join(folderpath,path)))
    studentName.append((os.path.splitext(path)[0]))

#storage ke liye
    filename = (f'{folderpath}/{path}')
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)



print(studentName)

def findEncoding(imageslist):
    encodelist = []
    for img in imageslist:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

print("Encoding Started")
encodeListKnown = findEncoding(imglist)
encodeListKnownwithname = [encodeListKnown, studentName]
print("Encoding Complete")
# print(encodeListKnown)

file = open("Encodefile.p","wb")
pickle.dump(encodeListKnownwithname,file)
file.close()
print("File saved")