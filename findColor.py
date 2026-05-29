import cv2 as cv
import numpy as np
import os
from windowcapture import WindowCapture

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*50)
print("HSV Color Finder")
print("Click on wafer BORDER to get HSV values")
print("="*50)

# Take screenshot
wincap = WindowCapture('realVNC.jpg')
img = wincap.get_screenshot()
cv.imwrite('full_screenshot.png', img)

hsv_values = []

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Get BGR color at clicked point
        bgr = img[y, x]
        b, g, r = bgr

        # Convert to HSV
        hsv = cv.cvtColor(np.uint8([[bgr]]), cv.COLOR_BGR2HSV)
        h, s, v = hsv[0][0]

        hsv_values.append((h, s, v))

        print(f'Point {len(hsv_values)}:')
        print(f'  BGR : ({b}, {g}, {r})')
        print(f'  HSV : ({h}, {s}, {v})')

        # After 3+ clicks suggest color range
        if len(hsv_values) >= 3:
            h_vals = [val[0] for val in hsv_values]
            s_vals = [val[1] for val in hsv_values]
            v_vals = [val[2] for val in hsv_values]

            margin = 15
            lower = [
                max(0,   min(h_vals) - margin),
                max(0,   min(s_vals) - margin),
                max(0,   min(v_vals) - margin)
            ]
            upper = [
                min(179, max(h_vals) + margin),
                min(255, max(s_vals) + margin),
                min(255, max(v_vals) + margin)
            ]

            print(f'\n--- Suggested Color Range ---')
            print(f'lower_purple = {lower}')
            print(f'upper_purple = {upper}')
            print(f'\nCopy to wafer_detector.py or use set_color_range()')

cv.imshow('Click on wafer BORDER color', img)
cv.setMouseCallback('Click on wafer BORDER color', mouse_callback)
cv.waitKey(0)
cv.destroyAllWindows()