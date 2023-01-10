import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'

image = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    currentImg = cv2.imread(f'{path}/{cls}')
    image.append(currentImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)    


def findEncodings(images):
    encodeList = []
    
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attandance.csv','r+') as f :
        myDetaiList = f.readlines()
        nameList = []
        for line in myDetaiList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList :
            now = datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')            
     
        
    
    

encodeListKnown = findEncodings(image)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True :
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurrentFram = face_recognition.face_locations(imgS)
    encodeCurrentFram = face_recognition.face_encodings(imgS, faceCurrentFram)
    
    for encodeFace, faceLoc in zip(encodeCurrentFram, faceCurrentFram):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            x1,x2,y1,y2 = faceLoc
            x1,x2,y1,y2 = x1*4,x2*4,y1*4,y2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,225,0),2)
            cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,225,0))
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(225, 255,255),2)
            markAttendance(name)   
            
    cv2.imshow('webCam', img)
    cv2.waitKey(1)        



# faceLocation = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(225,0,225),2)

# faceLocationTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(225,0,225),2)

# result = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)

