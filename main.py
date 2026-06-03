import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from wafer_detector import WaferDetector
from notification import WindowNotify

os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap   = WindowCapture('RealVNC')
detector = WaferDetector()
notify   = WindowNotify("Prober Screen")
notify.started()

loop_time = time()
while True:
    try:
        # Get screenshot once per frame
        screenshot = wincap.get_screenshot()

        # Find black area with circle shape inside
        wafer_screenshot, wafer_pos, circle_info = detector.find_wafer(screenshot)

        if wafer_screenshot is not None:
            # Draw box on full screenshot
            screenshot = detector.draw_wafer_box(screenshot, wafer_pos)

            # Draw circle on wafer image
            wafer_with_circle = detector.draw_circle(
                                    wafer_screenshot.copy(), 
                                    circle_info)

            # Show wafer map with circle drawn
            cv.imshow('Wafer Map', wafer_with_circle)
        else:
            print("Wafer not found!")

        cv.imshow('Computer Vision', screenshot)

        fps = 1 / (time() - loop_time)
        print(f'FPS: {fps:.1f}')
        loop_time = time()

    except Exception as e:
        print(f'Error: {e}')
        break

    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        notify.stopped()
        break

print('Done.')
