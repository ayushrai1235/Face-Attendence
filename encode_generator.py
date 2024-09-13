import cv2
import pickle
import os
import face_recognition

folderpath = 'Images'
pathlist = os.listdir(folderpath)
imglist = []
studentName = []
for path in pathlist:
    imglist.append(cv2.imread(os.path.join(folderpath,path)))
    studentName.append((os.path.splitext(path)[0]))

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