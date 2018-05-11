import numpy as np
import cv2
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture('rtmp://192.168.20.144:1935/flash/12:admin:EYE3inapp')
ret, img = cam.read()
id = raw_input("Enter user name : ").lower()
grab_frame = True
sampleNum = 0
while True:
	ret, img = cam.read()
	if grab_frame:
		img_copy = img
		faces = faceDetect.detectMultiScale(img_copy,scaleFactor=1.3,minNeighbors=5)
		folder = 'images/train' 
		if not os.path.exists(folder):
			original_umask = os.umask(0)
			os.makedirs(folder)
		for (x,y,w,h) in faces:
			sampleNum = sampleNum+1
			img_copy = img_copy[y:y+h,x:x+w]
			cv2.imwrite(folder+"/"+str(id)+'.'+str(sampleNum)+".jpg",cv2.resize(img_copy,(50,50)))
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.waitKey(1)
		cv2.imshow('face',img)
		cv2.waitKey(1)
		if sampleNum > 20000:
			break

print 'Sampling Done'
cam.release()
cv2.destroyAllWindows()








