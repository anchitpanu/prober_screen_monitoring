import cv2 as cv
import numpy as np
import os
import keyboard
from time import time
from windowcapture import WindowCapture
from notification import WindowNotify


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
# wincap = WindowCapture('window name')
wincap = WindowCapture('[program name]')    # program's name is focused on

notify = WindowNotify("Prober Screen")      # topic of notification

notify.started()


loop_time = time()
while(True):

    # if keyboard.is_pressed('esc'):
    #     notify.stopped()
    #     break

    # get an updated image of the program
    screenshot = wincap.get_screenshot()

    # display the image
    cv.imshow('Computer Vision', screenshot)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()


    # # press 'q' with the output window focused to exit.
    # # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)

    if key == ord('q'):
        cv.destroyAllWindows()
        notify.stopped()
        break

print('Done.')