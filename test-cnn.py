import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import os
from random import shuffle 
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression  
import tensorflow as tf
import time 

IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'persondetector-{}-{}.model'.format(LR, '2conv-basic')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

fig=plt.figure()


def process_frame(img):
	output_data = []
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        output_data.append([np.array(img), 0])
	return output_data

cam = cv2.VideoCapture('rtmp://192.168.20.144:1935/flash/12:admin:EYE3inapp')
stop_time = time.time()
print 'Camera Warming Up...'

while True:
	ret, img = cam.read()
	if time.time() - stop_time >= 5:
		break

print 'Capture Started...'
while True:
	ret, img = cam.read()
	faces = faceDetect.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)
	for (x,y,w,h) in faces:
		img_copy = img[y:y+h,x:x+w]
		output_data = process_frame(img_copy)
		data = output_data[0]
		img_num = data[1]
    		img_data = data[0]
		data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
		model_out = model.predict([data])[0]
		print model_out
		if np.argmax(model_out) == 1: 
			str_label='Arvind'
		else: 
			str_label='Joann' 
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(1)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,str_label,(x,y), font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Output', img)
	key = cv2.waitKey(1)
	if key & 0xff == ord('q'):
		break
