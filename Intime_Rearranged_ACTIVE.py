#!/usr/bin/env python
# coding: utf-8

# ### This jupyter notebook is to recognize faces on live camera

# In[6]:


# importing libraries for voice
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
#import pyaudio
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import time
#from plyer import notification
import calendar
import datetime
# importing lib for FR
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import playsound
import datetime
import time
import os
from gtts import gTTS


# In[7]:


# initializing MTCNN and InceptionResnetV1 
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')

def talk(text):
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()


# in_time = {}
break_intime = {}
lunch_intime = {}

def true():
    talk('hello' +name+ ', In time is noted')
    in_time = str(datetime.datetime.now())
    print(name + ' : ' + in_time)
    # print(min_dist)
    # global pers
    # global timein
    # pers = []
    # timein = []
    # pers.append(name)
    # timein.append(in_time)
    intime()
    
def intime():
    mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=False
    mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval() 


    # loading data.pt file

    
    load_data = torch.load('rudram_SERVER_13july2.pt')
    # load_data = torch.load('rudram6000_9July.pt') 
    embedding_list = load_data[0] 
    name_list = load_data[1] 
    cam = cv2.VideoCapture(0) 
    # cam = cv2.VideoCapture('rtsp://admin:rudra22test***@192.168.1.205')  ## For IP camera
    # cam = cv2.VideoCapture('https:/192.168.106.212:8080/video')


    while True:
        ret, frame = cam.read()
        if not ret:
            print("fail to grab frame, try again")
            break
            
        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
        
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
                    
            for i, prob in enumerate(prob_list):
                if prob>0.95:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                    
                    dist_list = [] # list of matched distances, minimum distance is used to identify the person
                    
                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list) # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                    global name
                    name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                    
                    box = boxes[i] 
                    
                    original_frame = frame.copy() # storing copy of frame before drawing on it
                    
                    if min_dist>0.65:
                        frame = cv2.putText(frame, 'Unknown', (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
                    elif min_dist <= 0.65 and min_dist >= 0.5:
                        frame = cv2.putText(frame, 'Recognizing..', (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2, cv2.LINE_AA)
                    elif min_dist<0.5:
                        #frame = cv2.putText(frame, name+' '+str(min_dist), (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA) ## With Minimum distance
                        frame = cv2.putText(frame, name, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA) ## Only name                     
                        #print(name,' : ',round(1-min_dist,2)*100,'%')
                        print(name)
                        print(min_dist)
                        #true()
                    frame = cv2.rectangle(frame, (box[0],box[1]) , (box[2],box[3]), (255,255,0), 1)
                    print(min_dist)

        cv2.imshow("IMG", frame)
            
        
        k = cv2.waitKey(1)
        if k%256==27: # ESC
            print('Esc pressed, closing...')
            break
        
        elif k%256==32: # space to save image
            print('Enter your name :')
            name = input()
            
            # create directory if not exists
            if not os.path.exists('photos/'+name):
                os.mkdir('photos/'+name)
                
            img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
            cv2.imwrite(img_name, original_frame)
            print(" saved: {}".format(img_name))

    # print(pers)        
    # print(timein)        
    cam.release()
    cv2.destroyAllWindows()




intime()

# In[ ]:




