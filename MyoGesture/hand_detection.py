import cv2
import numpy as np
import time  # borrar al terminar el test
from matplotlib import pyplot as plt
import random
'''
    Constants
'''
# Values
GUIDE_UPPER = (320, 115)
GUIDE_LOWER = (0, 95)
IMG_WIDTH = 320
IMG_HEIGHT = 240

RECORD_TIME = 4  # Segundos que vamos a grabar


# FINGER_MIN = np.array([115, 20, 20], np.uint8)
# FINGER_MAX = np.array([200, 150, 150], np.uint8)

# Lab
FINGER_MIN = np.array([78, 70, 70], np.uint8)
FINGER_MAX = np.array([255, 180, 180], np.uint8)



'''
    Code
'''


def set_up(camera):
    camera.set(3, IMG_WIDTH)
    camera.set(4, IMG_HEIGHT)


# Adds a transparent *color* rectangle as a guide for a
# correct hand/finger position
# (0, 255, 0) green
# ( 255, 0, 0) red
def add_guide(img, color):
    overlay = img.copy()
    # -1 es relleno
    rectangle = cv2.rectangle(img, GUIDE_UPPER, GUIDE_LOWER, color, -1)
    alpha = 0.7
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img


# When recording, this method prints a more vivid red to know how much
# time is elapsed since the recording started
def add_progress(img, color, actualProgress):
    overlay = img.copy()
    xAxis = int(actualProgress*IMG_WIDTH/RECORD_TIME)
    progress = (xAxis, GUIDE_UPPER[1])
    rectangle = cv2.rectangle(overlay, progress, GUIDE_LOWER, color, -1)
    alpha = 0.5
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img


# Given an x (width) value return the mean H value
# between upper height and lower height
def mean_h_value_for_x_point(img, x):
    meanH = 0
    for i in range(GUIDE_LOWER[1], GUIDE_UPPER[1]):
        meanH += img[i][x]
    return round(meanH/(GUIDE_UPPER[1] - GUIDE_LOWER[1]), 2)


# Given a image, return the mean of H value
# for each width pixel in a list
def mean_h_value_for_img(img):
    hValues = []
    for i in range(0, IMG_WIDTH):
        hValues.append(mean_h_value_for_x_point(img, i))
    return hValues


# Determines when the finger begins and returns the finger lenght in pixels
def finger_length(hValues):
    threshold = sum(hValues[0:40])/40

    for i in range(40, len(hValues)):
        # Find the value that is higher than the threshold
        if hValues[i] < threshold:
            return IMG_WIDTH - i

    # In case of error return -1
    return -1


# Returns de h values
def get_finger_strech_and_H_value_from_img(raw_frame):
    # Convert the frame into HSV image
    # hsvImg = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
    # Apply a blur to clean up the frame
    blurImg = cv2.GaussianBlur(raw_frame, (3, 7), 0)
    # Apply a threshold to try to binarizing the image
    threshImg = cv2.inRange(blurImg, FINGER_MIN, FINGER_MAX)
    # Get the H values for the image
    hValues = mean_h_value_for_img(threshImg)
    # Get the finger lenght
    fingerLen = finger_length(hValues)
    return hValues, fingerLen


def get_finger_strech_value_from_img(raw_frame):
    hValues, fingerLen = get_finger_strech_and_H_value_from_img(raw_frame)
    return fingerLen


def plot_graph_and_img_results(hValues, img, fingerLength, cont):
    plt.figure(figsize=(16, 12), dpi=200)
    # plt.suptitle('Media de los valores H situados en la franja verde a lo largo de la anchura de la imagen.')
    plt.suptitle('Valores medios de la vertical encerrada en la guia por cada pixel del ancho de la imagen.')

    # Plotting the data of H values
    plt.subplot(211)
    plt.ylabel('Valor de la media.')
    plt.xlabel('Pixel del ancho de la imagen.')
    plt.xlim(0, len(hValues))
    plt.xticks(range(0, len(hValues)+10, 10))
    plt.grid(True)
    plt.plot(hValues, label='Menor valor de media es más oscuro en la imagen.')
    plt.axvline(x=320-fingerLength, color='r')
    plt.legend(loc='best', framealpha=0.5, prop={'size':'small'})

    # Plotting de image from wich the data was taken
    plt.subplot(212)
    plt.xlabel('Imagen correspondiente a los datos.')
    plt.xlim(0, IMG_WIDTH)
    plt.xticks(range(0, IMG_WIDTH+20, 20))
    plt.axvline(x=IMG_WIDTH-fingerLength, color='r')
    plt.imshow(img)

    # plt.show()
    plt.savefig('detection'+str(cont)+'test.png', dpi=200)
    plt.close()



def is_record_finished(startTime):
    elapsedTime = time.monotonic() - startTime
    return elapsedTime, elapsedTime >= RECORD_TIME

'''
    This method is desinged for the purpose of testing.
    Set everything and take a single frame with some debbuging
    when pressing the key r (record)
'''
def take_a_frame(num):
    cam = cv2.VideoCapture(num)
    set_up(cam)
    num_frame = 1
    ok, frame = cam.read()
    lens  = []
    recording = False
    COLOR_RED = (0, 0, 255)
    while True:
        ok, frame = cam.read()

        # hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsvImg = cv2.GaussianBlur(frame, (3, 7), 0)
        # hsvImg = cv2.inRange(hsvImg, FINGER_MIN, FINGER_MAX)
        # show = add_guide(hsvImg.copy(), (255, 0, 0))
        show = add_guide(frame.copy(), (0, 255, 0))
        # cv2.imshow('hsv filter', show)

        if cv2.waitKey(1) & 0xFF == ord('r'):
            # hValues = mean_h_value_for_img(hsvImg)
            # fingerLength = finger_length(hValues)
            # print(fingerLength)
            # plot_graph_and_img_results(hValues, add_guide(frame, (0, 255 ,0)), fingerLength, num_frame)
            # num_frame +=1
            # print('\n\n\n')
            recording = True
            startRecording = time.monotonic()
            cont = 0

        if recording:
            # Stop recording after hd.RECORD_TIME seconds
            elapsedTime, finished = is_record_finished(startRecording)
            # The guide turns red when recording
            userImg = add_guide(frame.copy(), COLOR_RED)
            # Shows the remaining time until the record ends
            userImgProgress = add_progress(userImg, COLOR_RED,  elapsedTime)
            cv2.imshow('Capture finger lenght', userImgProgress)
            # cv2.imwrite('Recording/recording-'+str(cont)+'.png', userImgProgress)
            cont+=1
            if finished:
                recording = False
        else:
            userImg = add_guide(frame.copy(), (0, 255, 0))
            cv2.imshow('Capture finger lenght', userImg)

if __name__ == '__main__':
    take_a_frame(1)
