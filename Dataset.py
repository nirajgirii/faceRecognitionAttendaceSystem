import cv2
import numpy as np
import os
import pickle

# '0' is for the webcam
video = cv2.VideoCapture(0)

# Load the Cascade Classifier Algorithm
faceDetection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create an array to store data
faceData = []
value = 0

# Ask the user for their name
fullName = input("Enter your Full Name: ")

# Create the 'data' directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Convert color image to gray image for CCA
while True:
    ret, frame = video.read()
    grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Scaling 
    facesAvailable = faceDetection.detectMultiScale(grayScaleImage, 1.3, 5)

    for (x, y, w, h) in facesAvailable:
        cropImage = frame[y:y+h, x:x+w]
        resizedImage = cv2.resize(cropImage, (50, 50))

        # Append the resized image to faceData
        faceData.append(resizedImage)

        # Write text in the output frame
        if len(faceData) <= 100 and value % 10 == 0:
            cv2.putText(frame, str(len(faceData)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)

        cv2.imshow('Frame', frame)
        
        value += 1

    k = cv2.waitKey(1)
    if len(faceData) >= 100 or k == ord('q'):
        break

# Close all open windows
video.release()
cv2.destroyAllWindows()

# Save the face data in a pickle file
faceData = np.array(faceData)
faceData = faceData.reshape(100, -1)

if 'fullNames.pkl' not in os.listdir('data/'):
    fullNames = [fullName] * 100
    with open('data/fullNames.pkl', 'wb') as file:
        pickle.dump(fullNames, file)
else:
    with open('data/fullNames.pkl', 'rb') as file:
        fullNames = pickle.load(file)
    fullNames = fullNames + [fullName] * 100

    with open('data/fullNames.pkl', 'wb') as file:
        pickle.dump(fullNames, file)

# Save all the face data detected in the camera in a pickle file
if 'faceData.pkl' not in os.listdir('data/'):
    with open('data/faceData.pkl', 'wb') as file:
        pickle.dump(faceData, file)
else:
    with open('data/faceData.pkl', 'rb') as file:
        data = pickle.load(file)
    faces = np.concatenate((data, faceData), axis=0)
    with open('data/faceData.pkl', 'wb') as file:
        pickle.dump(faces, file)
