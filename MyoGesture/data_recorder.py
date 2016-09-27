import cv2
import time
from matplotlib import pyplot as plt

import sys

import myo_raw as myo
import hand_detection as hd
from common import *

from collections import deque

import threading
import my_io

import random


from multiprocessing import Process, Queue

'''
    Constants
'''
COLOR_RED = (0, 0, 255)


'''
# Thread class for recording emg values:
class EmgRecorder(multiprocessing.process):
    # Override Thread’s __init__ method to accept the parameters needed:
    def __init__(self, my_myo):
        self.m = my_myo
        self.begin = time.monotonic()
        print('begin: ', self.begin)
        threading.Thread.__init__(self)

    def run(self):
        cont = 0
        elapsedTime = time.monotonic() - self.begin
        print('elapsed: ', elapsedTime)
        self.m.clear_emg()
        self.m.mc_start_collection()
        while elapsedTime <= hd.RECORD_TIME:
            cont+=1
            recv = self.m.run(1)
            if recv is None:
                print('Connection lost')
                m.connect()
            # self.m.run(1)
            print(recv)
            elapsedTime = time.monotonic() - self.begin
        self.m.mc_end_collection()
        print('cont: ', cont)



class FingerLengthRecorder(threading.Thread):
    def __init__(self, raw_frame, fingerLengths, elapsedTime):
        self.frame = raw_frame
        self.fl = fingerLengths  # deque
        self.elapsedTime = elapsedTime
        threading.Thread.__init__(self)

    def run(self):
        # The guide turns red when recording
        userImg = hd.add_guide(self.frame.copy(), COLOR_RED)
        # Shows the remaining time until the record ends
        userImgProgress = hd.add_progress(userImg, COLOR_RED,  self.elapsedTime)
        ## cv2.imwrite('theFuckHappened'+str(cont)+'.png', frame) # NOTE: delete after testing
        cv2.imshow('Capture finger lenght', userImgProgress)
        # Determine where is the finger and save the length
        l = hd.get_finger_strech_value_from_img(self.frame)

        self.fl.append(l)
'''

def is_record_finished(startTime):
    elapsedTime = time.monotonic() - startTime
    return elapsedTime, elapsedTime >= hd.RECORD_TIME


def process_finger_length(raw_frame, elapsedTime, finger_queue):
    # Determine where is the finger and save the length
    l = hd.get_finger_strech_value_from_img(raw_frame)
    # Return the length calculated
    finger_queue.put(l)


def process_run_myo(m, emg_queue):
    m.bt.buff_cont = 0
    m.bt.ret_cont = 0
    oks = []
    cont = time.monotonic()
    while time.monotonic() - cont < 4:
        ok = m.run(0.2)
        oks.append(ok)
        print(ok)

    cont_none = 0
    for k in oks:
        if k is None:
            cont_none+=1

    if cont_none > 10:
        myo_disconnected()

    if is_myo_connected():
        emg_queue.put(m.get_emg().copy())

    m.clear_emg()


def myo_reconnect(m):
    m.disconnect()
    m2 = myo.MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
    return m2


myo_connected = True


def myo_disconnected():
    global myo_disconnected    # Needed to modify global copy of globvar
    myo_connected = False


def is_myo_connected():
    if myo_connected is True:
        return True
    else:
        return False


def run(cam_port):
    # Camera set up
    cam = cv2.VideoCapture(cam_port)
    hd.set_up(cam)

    # Myo set up
    m = myo.MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
    m.connect()
    is_disconnected = False

    # Recording set up
    recording = False
    name_data_file = 'data_'
    num_records = 0
    queue_length = Queue() # Store finger lenghts
    aux = deque()

    # Start
    while(True):
        # Take a frame form the webcam
        ok, frame = cam.read()

        # Start collecting data when r (recording) is pressed
        if cv2.waitKey(1) & 0xFF == ord('r'):
            # Flag to start the recording's state
            recording = True
            # Set up a thread for recording emg values
            startRecording = time.monotonic()
            # Set up the emg recoerder thread
            # hilo_emg = EmgRecorder(m)
            # hilo_emg.start()
            emg_queue = Queue()
            p = Process(target=process_run_myo, args=(m, emg_queue, ))
            p.start()

        if recording:
            # Stop recording after hd.RECORD_TIME seconds
            elapsedTime, finished = is_record_finished(startRecording)
            # The guide turns red when recording
            userImg = hd.add_guide(frame.copy(), COLOR_RED)
            # Shows the remaining time until the record ends
            userImgProgress = hd.add_progress(userImg, COLOR_RED,  elapsedTime)
            cv2.imshow('Capture finger lenght', userImgProgress)

            if finished:
                if not is_myo_connected():
                    print('Error interno de Myo. Reconectando...')
                    m = myo_reconnect()
                    queue_length = Queue()
                    recording = False
                else:
                    # Save emg values
                    p.join()
                    emg = emg_queue.get()  # emg is a queue
                    # Save finger lenghts
                    fingerList = [queue_length.get() for i in range(0, queue_length.qsize())]
                    finger = deque(normalize(fingerList))
                    queue_length = Queue()
                    # Label each emg value with a finger length [0 - 10]
                    dictLabeledData = label_data(emg, finger)
                    # Save in a file the data
                    my_io.save_dict_to(dictLabeledData, name_data_file+str(num_records))
                    num_records += 1

                    recording = False

            else:
                # Set up the thread that calcules the finger length for every frame
                p2 = Process(target=process_finger_length, args=(frame, elapsedTime, queue_length, ))
                p2.start()

        # not recording, show the user-friendly image
        else:
            userImg = hd.add_guide(frame.copy(), (0, 255, 0))
            cv2.imshow('Capture finger lenght', userImg)

    cam.release()
    cv2.destroyAllWindows()


def print_ordered_dict(myDict):
    for key in sorted(myDict.keys()):
        print("%s: %s" % (key, len(myDict[key])))


# EMG and finger values are deques
def label_data(emgValues, fingerValues):
    labeledData = {l: [] for l in fingerValues}
    # As long as we have labels left
    while fingerValues:  # while is not empty
        # How many emg values are we going to label
        numElements = int(round(len(emgValues)/len(fingerValues), 0))
        begin = fingerValues.popleft()
        if fingerValues:
            end = fingerValues.pop()
            for i in range(0, numElements):
                if emgValues:
                    labeledData[begin].append(emgValues.popleft())
                if emgValues:
                    labeledData[end].append(emgValues.pop())
            #print(print_ordered_dict(labeledData))

        # when there are only one label left, add the rest to that label
        else:
            print('else label_data')
            while emgValues:
                labeledData[begin].append(emgValues.popleft())

    #for key in allDataFixed[0].keys():
    #    print('key: ', key, ' - len: ',  len(allDataFixed[0][key]))
    return labeledData


# Alternative
# http://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
def normalize(listData):
    oldMax = max(listData)
    oldMin = min(listData)
    oldRange = (oldMax - oldMin)

    if oldRange is 0:
        oldRange = 1

    newMax = 5
    newMin = 0
    newRange = (newMax - newMin)

    normalizedValues = []

    for d in listData:
        if d is not 320:
            normValue = (((d - oldMin) * newRange) / oldRange) + newMin
            normValue = int(round(normValue, 0))
            normalizedValues.append(int(normValue))

    return normalizedValues


if __name__ == '__main__':
    run(0)
