# pip install cmake
# pip install face_recognition
# pip install opencv
# pip install numpy
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# load known faces
abhi_image = face_recognition.load_image_file("faces/abhi.jpg")
abhi_encoding = face_recognition.face_encodings(abhi_image)[0]

bince_image = face_recognition.load_image_file("faces/bince.jpg")
bince_encoding = face_recognition.face_encodings(bince_image)[0]

known_face_encodings = [abhi_encoding, bince_encoding]
known_face_names = ["abhi", "bince"]

# list of expected students
students = known_face_names.copy()

face_location = []
face_encodings = []

# Get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    frame: object
    name1: object
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # RECOGNIZE FACES
    face_location = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_location)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name1 = known_face_names[best_match_index]

        # add test if the person is present
        if name1 in known_face_names:
            org = (10, 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name1 + " present ", org, font, fontScale, fontColor, thickness, lineType)

            if name1 in students:
                students.remove(name1)
                current_time = now.strftime("%H:%M-min-%Ssec")
                lnwriter.writerow([name1, current_time])

    cv2.imshow("attendance", frame)
    if cv2.waitKey(1) & 0xff == ord("d"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
