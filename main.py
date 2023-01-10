import cv2
import numpy as np
import face_recognition

imgElon= face_recognition.load_image_file('ImagesAttendance/Elon.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
imgTest= face_recognition.load_image_file('ImagesAttendance/Elon.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

faceLocation = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(225,0,225),2)

faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(225,0,225),2)

result = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
cv2.putText(imgTest,f'{result}{round(faceDis[0],2)}',(0,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225),2)

print(result,faceDis)

cv2.imshow('Elon Mask', imgElon)
cv2.imshow('TEST', imgTest)
cv2.waitKey(0)