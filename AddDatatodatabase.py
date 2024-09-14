import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountkey (2).json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-attendance-realtime-bc190-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "Ayush_rai":
        {
            "id": 7,
            "name": "Ayush rai",
            "major": "cse",
            "starting_year": 2024,
            "total_attendance": 16,
            "year":1,
            "last_attendence_time": "2024-09-12 00:54:34"
        },
    "Elon_Musk":
        {
            "id": 1,
            "name": "Elon Musk",
            "major": "AI",
            "starting_year": 2020,
            "total_attendance": 20,
            "year":4,
            "last_attendence_time": "2024-09-12 00:59:34"
        },
    "Emily":
        {
            "id": 10,
            "name": "Emily ben",
            "major": "cse",
            "starting_year": 2021,
            "total_attendance": 11,
            "year":3,
            "last_attendence_time": "2024-09-12  00:34:34"
        },
    "Hrithik_roshan":
        {
            "id": 4,
            "name": "Hrithik roshan",
            "major": "mechanical",
            "starting_year": 2021,
            "total_attendance": 21,
            "year": 3,
            "last_attendence_time": "2024-09-12  00:34:34"
        },

}

for key,value in data.items():
    ref.child(key).set(value)