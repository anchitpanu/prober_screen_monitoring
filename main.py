import cv2 as cv
import numpy as np
import os
from time import time

from windowcapture import WindowCapture
from circledetector import CircleDetector
from notification import WindowNotify

os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap   = WindowCapture('wftb33')
detector = CircleDetector()
notify   = WindowNotify("Prober Screen")
notify.started()

detector.debug = True

loop_time = time()
while True:
    try:
        screenshot = wincap.get_screenshot()

        # wafer_screenshot, wafer_pos, circle_info = detector.find_wafer(screenshot)

        if screenshot is not None:

            # detect biggest circle
            annotated, cropped, circle_info = detector.process(screenshot)

            if circle_info is not None:
                cx, cy, r = circle_info
                print(f"[CircleDetector] centre = ({cx}, {cy}) radius = {r}px")

                # show circle only window
                if cropped is not None:
                    cv.imshow('Circle Only', cropped)

            else:
                print("[CircleDetector] No circle found.")

            # cv.imshow('Wafer Map', screenshot)
            # cv.imshow('Biggest Circle (Wafer)', annotated)

        else:
            print("Wafer not found!")
        
        cv.imshow('Computer Vision', screenshot)

        fps = 1 / (time() - loop_time)
        print(f'FPS: {fps:.1f}')
        loop_time = time()

    except Exception as e:
        print(f'Error: {e}')
        cv.waitKey(500)

    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        notify.stopped()
        break

print('Done.')