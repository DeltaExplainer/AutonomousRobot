import cv2
import os
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#each person will now get an id
face_id  = input('\n enter user id and press return ')
print('\n Initializing Face Capture. Look at the Camera & Please Wait :)')

count = 0
while(True):
      ret, img = cam.read()

      #if(not ret):
          #continue
      #img = cv2.flip(img, -1)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_detector.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
          cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

          count += 1

          cv2.imwrite("/home/pi/RealTimeFaceDetection/FacialRecognition/dataset/User." +str(face_id) + '.' + str(count) + ".jpg", gray [y:y+h,x:x+w])
          cv2.imshow('image', img)
      k = cv2.waitKey(100) & 0xff
      if k == 27:
          break
      elif count >= 30:
          break

print("\n Exiting Program")
cam.release()
cv2.destroyAllWindows()
