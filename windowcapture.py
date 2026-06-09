import numpy as np
import win32gui, win32ui, win32con
import ctypes
from threading import Thread, Lock


class WindowCapture:

    # threading properties
    stopped = True
    lock = None
    screenshot = None

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name):
        # create a thread lock object
        self.lock = Lock()

        # set DPI awareness at the start of initialization 
        ctypes.windll.user32.SetProcessDPIAware()

        # find the handle for the window we want to capture
        # self.hwnd = win32gui.FindWindow(None, window_name)
        self.hwnd = self._find_window_partial(window_name)

        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # print actual window name found
        actual_name = win32gui.GetWindowText(self.hwnd)
        print(f'Window found: "{actual_name}')

        # update window dimensions
        self._update_window_dimensions()

    def _update_window_dimensions(self):
        """Update window dimensions to handle window resizing"""
        # get the client area (actual content area without borders/titlebar)
        client_rect = win32gui.GetClientRect(self.hwnd)
        self.w = client_rect[2] - client_rect[0]
        self.h = client_rect[3] - client_rect[1]
        
        # Get window position for coordinate translation
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.offset_x = window_rect[0]
        self.offset_y = window_rect[1]

    def _find_window_partial(self, partial_name):
        result = []

        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                 
                # case insensitive partial match
                if partial_name.lower() in window_title.lower():
                    result.append(hwnd)
                    print(f'Matched: "{window_title}')

        win32gui.EnumWindows(winEnumHandler, None)

        if not result:
            return None
        
        # If multiple windows found, use first one
        if len(result) > 1:
            print(f'Warning: {len(result)} windows found, using first one')
            for i, hwnd in enumerate(result):
                print(f' {i+1}. "{win32gui.GetWindowText(hwnd)}')

        return result[0]

    def get_screenshot(self):
        # update window dimensions to handle resizing
        self._update_window_dimensions()
        
        # Get full window rect for capture
        window_rect = win32gui.GetWindowRect(self.hwnd)
        full_w = window_rect[2] - window_rect[0]
        full_h = window_rect[3] - window_rect[1]
        
        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, full_w, full_h)
        cDC.SelectObject(dataBitMap)
        
        # PrintWindow with flag 2 captures the full window
        result = ctypes.windll.user32.PrintWindow(self.hwnd, cDC.GetSafeHdc(), 2)

        # convert the raw data into a format opencv can read
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (full_h, full_w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel
        img = img[...,:3]

        # Calculate actual borders from the window and client rects
        client_rect = win32gui.GetClientRect(self.hwnd)
        client_w = client_rect[2] - client_rect[0]
        client_h = client_rect[3] - client_rect[1]
        
        # Crop to client area (remove borders and titlebar)
        border_left = (full_w - client_w) // 2
        border_top = full_h - client_h - border_left
        img = img[border_top:border_top+client_h, border_left:border_left+client_w]

        # make image C_CONTIGUOUS
        img = np.ascontiguousarray(img)

        if result == 0:
            print("Warning: PrintWindow failed")

        return img

    
    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)
