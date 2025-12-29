from time import sleep
import numpy as np
import math
import cv2
import pygame
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import threading

offset = 30
imgSize = 300
pygame.mixer.init()

sound_playing = False


def play_sound(sound_path):
    global sound_playing
    pygame.mixer.music.load(f'../sound/{sound_path}.mp3')
    pygame.mixer.music.play()

    # Continuously check if the music is still playing
    while pygame.mixer.music.get_busy():
        sleep(0.1)  # Sleep a bit to prevent high CPU usage

    # When music finishes, set sound_playing to False
    sound_playing = False


def animate(image_path, sound_path, imgOutput, x, y):
    global sound_playing

    # Load the animation image
    animation = cv2.imread(
        f'../filter/{image_path}', cv2.IMREAD_UNCHANGED)[:, :, :3]

    if animation is None:
        print("Error: Animation image not found.")
        return

    if not sound_playing:
        sound_thread = threading.Thread(target=play_sound, args=(sound_path,))
        sound_thread.start()
        sound_playing = True

    hh, ww = animation.shape[:2]
    if (y-100 >= 0) and (x-100 >= 0) and (y + hh-100 <= imgOutput.shape[0]) and (x + ww-100 <= imgOutput.shape[1]):
        imgOutput[y-100:y+hh-100, x-100:x-100+ww] = animation
    else:
        print("Error: Overlay image exceeds the bounds of the main image.")


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("../model/keras_model.h5", "../model/labels.txt")
counter = 0
labels = ["MALEVOLENT SHRINE", "INFINITE VOID",
          "KONO DIO DA!", "KAMEHAMEHA", "RIZZING YOU", "NETERO 99"]
while True:
    succes, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        x1 = 0
        y1 = 0
        w1 = 0
        h1 = 0
        hand1 = hands[0]
        x, y, w, h = hand1['bbox']
        if (len(hands) > 1):
            hand2 = hands[1]
            x1, y1, w1, h1 = hand2['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        if (x < x1 and y < y1):
            imgCrop = img[y-offset:y+h1 +
                          abs(y1-y)+offset, x-offset:x+w1+abs(x1-x)+offset]
        if (x > x1 and y > y1):
            imgCrop = img[y1-offset:y1+h +
                          abs(y-y1)+offset, x1-offset:x1+w+abs(x-x1)+offset]
        if (x < x1 and y > y1):
            imgCrop = img[y1-offset:y1+h +
                          abs(y-y1)+offset, x-offset:x+w1+abs(x1-x)+offset]
        if (x > x1 and y < y1):
            imgCrop = img[y-offset:y+h1 +
                          abs(y1-y)+offset, x1-offset:x1+w+abs(x-x1)+offset]

        if (x > 0 and x1 == 0):
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        ratio = (h1+h)/(w+w1)

        if (ratio > 1):
            k = imgSize/(h1+h)
            wCal = math.ceil(k*(w+w1))
            if (imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0):
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)

        else:
            k = imgSize/(w+w1)
            hCal = math.ceil(k*(h1+h))
            if (imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0):
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
        if prediction[index] > 0.92:
            if (index == 0 or len(hands) < 1):
                animate("sukuna.jpg", "domain-expansion-sukuna", imgOutput, x, y)
            if (index == 1 or len(hands) < 1):
                animate(
                    "gojo.jpg", "Gojo domain expansion sound effects", imgOutput, x, y)
            if (index == 2 or len(hands) < 1):
                animate("dio.jpg", "kono-dio-da99", imgOutput, x, y)
            if (index == 3 or len(hands) < 1):
                animate("kameha.jpg", "kamehameha", imgOutput, x, y)
            if (index == 4 or len(hands) < 1):
                animate("rizzler.jpg", "WHAT DA HELLLLL", imgOutput, x, y)
            if (index == 5 or len(hands) < 1):
                animate("99hand.jpg", "99hand", imgOutput, x, y)

            if (imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0):
                cv2.putText(
                    imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
            if (imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0):
                cv2.putText(imgOutput, "UNKOWN", (x, y-20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("imageCrop", imgCrop)
            cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
