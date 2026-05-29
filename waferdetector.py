import cv2 as cv
import numpy as np

class WaferDetector:
    
    def __init__(self):
        """
        Initialize WaferDetector with default purple/magenta color range
        You can adjust HSV values using find_hsv_color.py
        """
        # Purple/Magenta color range (wafer border)
        # Adjust these values using find_hsv_color.py
        self.lower_purple = np.array([120, 50, 50])
        self.upper_purple = np.array([180, 255, 255])
        self.padding = 10

    def set_color_range(self, lower, upper):
        """
        Set custom HSV color range for wafer border
        
        Parameters:
        -----------
        lower : list [H, S, V] lower bound
        upper : list [H, S, V] upper bound
        
        Example:
        --------
        detector.set_color_range([130, 50, 50], [170, 255, 255])
        """
        self.lower_purple = np.array(lower)
        self.upper_purple = np.array(upper)
        print(f'Color range set: lower={lower}, upper={upper}')

    def find_wafer(self, img):
        """
        Find wafer map by detecting purple/magenta border circle
        
        Parameters:
        -----------
        img : full screenshot (BGR format from get_screenshot())
        
        Returns:
        --------
        wafer_img : cropped wafer image or None if not found
        wafer_pos : tuple (x, y, w, h)  or None if not found
        """
        if img is None:
            return None, None

        # Convert to HSV for better color detection
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Create mask for purple/magenta color
        mask = cv.inRange(hsv, self.lower_purple, self.upper_purple)

        # Remove noise
        # kernel = np.ones((5, 5), np.uint8)
        kernel = np.ones((3,3), np.uint8)   # smaller kernel
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)  # remove small noise
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # fill small holes

        # Find contours
        contours, _ = cv.findContours(mask, 
                                       cv.RETR_EXTERNAL, 
                                       cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found!")
            return None, None
        
        areas = [cv.contourArea(c) for c in contours]
        print(f'Found {len(contours)} contours, areas: {sorted(areas, reverse=True)[:5]}')

        # Find largest contour (should be wafer circle)
        largest = max(contours, key=cv.contourArea)
        area = cv.contourArea(largest)
        print(f'Largest contour area: {area}')

        # Lower minimum area
        if area < 100:
            print(f"Contour too small: {area}")
            return None, None

        # Filter too small contours
        # if cv.contourArea(largest) < 1000:
        #     print("Contour too small - wafer not found!")
        #     return None, None

        # Get bounding box
        x, y, w, h = cv.boundingRect(largest)

        # Add padding
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(img.shape[1] - x, w + self.padding * 2)
        h = min(img.shape[0] - y, h + self.padding * 2)

        # Crop wafer map
        wafer_img = img[y:y+h, x:x+w]
        wafer_img = np.ascontiguousarray(wafer_img)

        return wafer_img, (x, y, w, h)

    def draw_wafer_box(self, img, wafer_pos):
        """
        Draw bounding box around detected wafer on full screenshot
        
        Parameters:
        -----------
        img       : full screenshot
        wafer_pos : tuple (x, y, w, h) from find_wafer()
        
        Returns:
        --------
        img : image with drawn bounding box
        """
        if wafer_pos is None:
            return img

        x, y, w, h = wafer_pos

        # Draw rectangle
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw label
        cv.putText(img, 
                   f'Wafer Map ({w}x{h})',
                   (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        return img

    def show_mask(self, img):
        """
        Show color mask for debugging
        Useful to check if color range is correct
        
        Parameters:
        -----------
        img : full screenshot
        """
        hsv  = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_purple, self.upper_purple)
        cv.imshow('Color Mask (white = detected)', mask)