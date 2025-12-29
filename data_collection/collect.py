import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
offset=20
imgSize=300
current_file_path = os.path.abspath(__file__)
last_save_time = time.time()  # Track the last time a picture was saved

folder = "../data/NETERO"

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=2)

counter=0
while True:
    succes,img=cap.read()
    hands,img=detector.findHands(img)
    copy=np.ones((imgSize,imgSize,3),np.uint8)*255
    if hands:
        x1=0
        y1=0
        w1=0
        h1=0
        hand1=hands[0]
        x,y,w,h=hand1['bbox']

        if(len(hands)>1):
            hand2=hands[1]
            x1,y1,w1,h1=hand2['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        if(x<x1 and y<y1 ):
            imgCrop=img[y-offset:y+h1+abs(y1-y)+offset,x-offset:x+w1+abs(x1-x)+offset]
        if(x>x1 and y>y1 ):
             imgCrop=img[y1-offset:y1+h+abs(y-y1)+offset,x1-offset:x1+w+abs(x-x1)+offset]
        if(x<x1 and y>y1 ):
            imgCrop=img[y1-offset:y1+h+abs(y-y1)+offset,x-offset:x+w1+abs(x1-x)+offset]
        if(x>x1 and y<y1 ):
             imgCrop=img[y-offset:y+h1+abs(y1-y)+offset,x1-offset:x1+w+abs(x-x1)+offset]
        
        if(x>0 and x1==0):
            imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]

        
        ratio=(h1+h)/(w+w1)
        if(ratio>1):
            k=imgSize/(h1+h)
            wCal=math.ceil(k*(w+w1))
            if( imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0):
                imgResize=cv2.resize(imgCrop,(wCal,imgSize))
                wGap=math.ceil((imgSize-wCal)/2)
                imgWhite[:,wGap:wCal+wGap]=imgResize
            
        else:
            k=imgSize/(w+w1)
            hCal=math.ceil(k*(h1+h))
            if( imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0):
                imgResize=cv2.resize(imgCrop,(imgSize,hCal))
                hGap=math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap,:]=imgResize
        
        if(imgCrop.shape[0]>0 and imgCrop.shape[1]>0):    
            cv2.imshow("imageCrop",imgCrop)
            cv2.imshow("imgWhite",imgWhite)
            copy=imgWhite        
    cv2.imshow("Image",img)
    current_time = time.time()
    if (current_time - last_save_time >= 0.1) and (len(hands)>=2):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', copy)
        print(f'Saving image number {counter}')
        last_save_time = current_time  # Update the last save time
    if( cv2.waitKey(1) & 0xFF==ord('q')):
        break
    # if(cv2.waitKey(1)==ord('l')):
    #     counter+=1
    #     cv2.imwrite(f'{folder}/image_{time.time()}.jpg', imgWhite)
    #     print(f'saving image number {counter}')
    


