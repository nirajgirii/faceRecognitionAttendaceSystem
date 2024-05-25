import os  # For opening and using camera
import cv2  # For image processing
import numpy as np  # For numerical operations on images
import pickle  # For saving the face data in a pickle file
import time  # For time operations
import csv  # For reading the CSV file
from sklearn.neighbors import KNeighborsClassifier  # For KNN algorithm i.e training dataset
from datetime import datetime, timedelta  # For date and time operations

# Ensure the 'Attendance' directory exists
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

# Load Dataset
video = cv2.VideoCapture(0)
faceDetection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Using KNN algorithm for training the dataset
with open('data/fullNames.pkl', 'rb') as file:
    LABELS = pickle.load(file)

with open('data/faceData.pkl', 'rb') as file:
    FACES = pickle.load(file)

knnAlgorithm = KNeighborsClassifier(n_neighbors=5)
knnAlgorithm.fit(FACES, LABELS)

# Create the CSV file columns
COLUMN_NAMES = ['Full Name', 'Date', 'Time']

# Dictionary to track the last attendance time
last_attendance = {}

# Load the last attendance times from the existing CSV file if it exists
def load_last_attendance():
    date = datetime.now().strftime('%Y-%m-%d')
    csv_file = f'Attendance/Attendance_{date}.csv'
    if os.path.isfile(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                name, date_str, time_str = row
                datetime_str = f"{date_str} {time_str}"
                last_attendance[name] = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

load_last_attendance()

# Using KNN algorithm for recognizing the face
while True:
    ret, frame = video.read()
    if not ret:
        continue
    
    grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Scaling 
    facesAvailable = faceDetection.detectMultiScale(grayScaleImage, 1.3, 5)

    for (x, y, w, h) in facesAvailable:
        cropImage = frame[y:y+h, x:x+w]
        resizedImage = cv2.resize(cropImage, (50, 50)).flatten().reshape(1, -1)
        
        output = knnAlgorithm.predict(resizedImage)
        student_name = output[0]
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        exist = os.path.isfile(f'Attendance/Attendance_{date}.csv')

        current_time = datetime.fromtimestamp(ts)

        # Check if the student has already been marked present in the last hour
        if student_name in last_attendance:
            last_time = last_attendance[student_name]
            if current_time - last_time < timedelta(hours=1):
                remaining_time = timedelta(hours=1) - (current_time - last_time)
                cv2.putText(frame, f"{student_name}: Try again after {str(remaining_time).split('.')[0]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue
        
        # Record attendance only if 'p' is pressed
        if cv2.waitKey(1) == ord('p'):
            cv2.putText(frame, student_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            attendance = [student_name, date, timestamp]
            last_attendance[student_name] = current_time

            # Write attendance to CSV
            if exist:
                with open(f'Attendance/Attendance_{date}.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow(attendance)
            else:
                with open(f'Attendance/Attendance_{date}.csv', 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(COLUMN_NAMES)
                    writer.writerow(attendance)
        else:
            cv2.putText(frame, student_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
