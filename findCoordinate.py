import cv2 as cv
import os
from windowcapture import WindowCapture


os.chdir(os.path.dirname(os.path.abspath(__file__)))


wincap = WindowCapture('realVNC.jpg')

# take one screenshot
full_img = wincap.get_screenshot()

# save it to check
cv.imwrite('full_screenshot.png', full_img)
print(f'Window size: width = {wincap.w}, height = {wincap.h}')

# click to find coordinate
click_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f'Point {len(click_points)}: x={x}, y={y}')

        # after 2 clicks calaulate region
        if len(click_points) == 2:
            x1, y1 = click_points[0]    # top-left
            x2, y2 = click_points[1]    # bottom-right
            width = x2 - x1
            height = y2 - y1
            print(f'\nwafer_region = ({x1}, {y1}, {width}, {height})')

cv.imshow('Click top-left then bottom-right of wafer map', full_img)
cv.setMouseCallback('Click top-left then bottom-right of wafer map', mouse_callback)
cv.waitKey(0)
cv.destroyAllWindows()

