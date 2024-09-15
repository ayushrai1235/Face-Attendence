import sys
import os
import cv2
import pickle
import numpy as np
import face_recognition
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, \
    QMessageBox, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class FaceAttendanceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)
        
        self.start_button = QPushButton("Start Attendance", self)
        self.start_button.clicked.connect(self.start_attendance)
        self.layout.addWidget(self.start_button)
        
        self.add_student_button = QPushButton("Add New Student", self)
        self.add_student_button.clicked.connect(self.add_new_student)
        self.layout.addWidget(self.add_student_button)
        
        self.generate_encodings_button = QPushButton("Generate Encodings", self)
        self.generate_encodings_button.clicked.connect(self.generate_encodings)
        self.layout.addWidget(self.generate_encodings_button)
        
        self.status_label = QLabel("Status: Ready", self)
        self.layout.addWidget(self.status_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.initialize_firebase()
        self.load_encodings()

        self.imgBackground = cv2.imread('Resources/background.png')
        self.folderModePath = 'Resources/Modes'
        self.modePathList = os.listdir(self.folderModePath)
        self.imgModeList = [cv2.imread(os.path.join(self.folderModePath, path)) for path in self.modePathList]

        self.modeType = 0
        self.counter = 0
        self.id = -1
        self.imgStudent = []

    def initialize_firebase(self):
        cred = credentials.Certificate("serviceAccountkey (2).json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://face-attendance-realtime-bc190-default-rtdb.firebaseio.com/",
            'storageBucket': "face-attendance-realtime-bc190.appspot.com"
        })
        self.bucket = storage.bucket()

    def load_encodings(self):
        try:
            file = open("Encodefile.p", "rb")
            self.encodeListKnownWithNames = pickle.load(file)
            file.close()
            self.encodeListKnown, self.studentNames = self.encodeListKnownWithNames
            print("Encodings Loaded.")
        except FileNotFoundError:
            print("Encodings file not found. Please generate encodings first.")
            self.encodeListKnown, self.studentNames = [], []

    def start_attendance(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.timer.start(20)
        self.status_label.setText("Status: Attendance Started")

    def update_frame(self):
        success, img = self.cap.read()
        if success:
            imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            faceCurFrame = face_recognition.face_locations(imgs)
            encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

            imgBackground = self.imgBackground.copy()
            imgBackground[162:162 + 480, 55:55 + 640] = img
            imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[self.modeType]

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = self.cornerRect(imgBackground, bbox, rt=0)
                    id = self.studentNames[matchIndex]

                    if self.counter == 0:
                        self.counter = 1
                        self.modeType = 1

            if self.counter != 0:
                if self.counter == 1:
                    studentInfo = db.reference(f'Students/{id}').get()
                    print(studentInfo)
                    blob = self.bucket.get_blob(f'Images/{id}.png')
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    self.imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                    datetimeObject = datetime.strptime(studentInfo['last_attendence_time'], "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                    print(secondsElapsed)
                    if secondsElapsed > 30:
                        ref = db.reference(f'Students/{id}')
                        studentInfo['total_attendance'] += 1
                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                        ref.child('last_attendence_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        self.modeType = 3
                        self.counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[self.modeType]

                if self.modeType != 3:
                    if 10 < self.counter < 20:
                        self.modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[self.modeType]

                    if self.counter <= 10:
                        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        imgBackground[175:175 + 216, 909:909 + 216] = self.imgStudent

                    self.counter += 1

                if self.counter >= 20:
                    self.counter = 0
                    self.modeType = 0
                    studentInfo = []
                    self.imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[self.modeType]

            self.display_image(imgBackground)

    def cornerRect(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        if rt != 0:
            cv2.rectangle(img, bbox, (0, 255, 0), rt)
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

    def display_image(self, img):
        qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img = img.rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def add_new_student(self):
        name, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter student name:')
        if ok and name:
            major, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter student major:')
            if ok and major:
                year, ok = QInputDialog.getInt(self, 'Input Dialog', 'Enter student year:', 1, 1, 5)
                if ok:
                    data = {
                        "id": len(self.studentNames) + 1,
                        "name": name,
                        "major": major,
                        "starting_year": datetime.now().year,
                        "total_attendance": 0,
                        "year": year,
                        "last_attendence_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    ref = db.reference('Students')
                    ref.child(name.replace(" ", "_")).set(data)
                    self.status_label.setText(f"Status: Added new student {name}")

                    options = QFileDialog.Options()
                    fileName, _ = QFileDialog.getOpenFileName(self, "Select Student Image", "",
                                                              "Image Files (*.png *.jpg *.bmp)", options=options)
                    if fileName:
                        destination = f'Images/{name.replace(" ", "_")}.png'
                        cv2.imwrite(destination, cv2.imread(fileName))

                        blob = self.bucket.blob(destination)
                        blob.upload_from_filename(destination)

                        self.status_label.setText(f"Status: Added new student {name} with image")
                    else:
                        self.status_label.setText(f"Status: Added new student {name} without image")

    def generate_encodings(self):
        folderPath = 'Images'
        pathList = os.listdir(folderPath)
        imgList = []
        studentNames = []
        for path in pathList:
            imgList.append(cv2.imread(os.path.join(folderPath, path)))
            studentNames.append(os.path.splitext(path)[0])
            fileName = f'{folderPath}/{path}'
            bucket = storage.bucket()
            blob = bucket.blob(fileName)
            blob.upload_from_filename(fileName)

        def findEncodings(imagesList):
            encodeList = []
            for img in imagesList:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

        print("Encoding Started ...")
        encodeListKnown = findEncodings(imgList)
        encodeListKnownWithNames = [encodeListKnown, studentNames]
        print("Encoding Complete")
        file = open("Encodefile.p", "wb")
        pickle.dump(encodeListKnownWithNames, file)
        file.close()
        print("File Saved")

        self.status_label.setText("Status: Encodings generated and saved")
        self.load_encodings()

    def closeEvent(self, event):
        if hasattr(self, 'cap'):
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceAttendanceApp()
    window.show()
    sys.exit(app.exec_())