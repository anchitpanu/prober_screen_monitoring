import cv2 as cv
import numpy as np

class WaferDetector:
    
    def __init__(self):
        """
        Find black area that contains wafer circle by SHAPE detection
        """
        # Black background color range
        self.lower_black = np.array([0,   0,   0 ])
        self.upper_black = np.array([180, 255, 50])

        self.min_area = 5000
        self.padding  = 10
        self.debug    = True

    def _has_wafer_circle(self, region_img):
        """
        Check if this region contains circle shape
        using HoughCircles
        
        Parameters:
        -----------
        region_img : cropped region image to check
        
        Returns:
        --------
        has_circle : True if circle found inside
        score      : radius of circle (larger = more likely wafer)
        circle_info: (cx, cy, r) or None
        """
        # Convert to grayscale
        gray    = cv.cvtColor(region_img, cv.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv.GaussianBlur(gray, (9, 9), 2)

        # Get region size
        h, w  = region_img.shape[:2]
        min_r = int(min(w, h) * 0.2)   # min radius = 20% of region
        max_r = int(min(w, h) * 0.6)   # max radius = 60% of region

        if self.debug:
            print(f'  Region size: {w}x{h} | '
                  f'Circle radius range: {min_r} - {max_r}')

        # Detect circles
        circles = cv.HoughCircles(
            blurred,
            cv.HOUGH_GRADIENT,
            dp       = 1,
            minDist  = min(w, h) // 2,  # only one circle expected
            param1   = 50,              # edge detection threshold
            param2   = 30,              # circle detection threshold
            minRadius= min_r,
            maxRadius= max_r
        )

        if circles is None:
            if self.debug:
                print(f'  No circle found')
            return False, 0, None

        # Get largest circle
        circles        = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda c: c[2])
        cx, cy, r      = largest_circle

        if self.debug:
            print(f'  Circle found: center=({cx},{cy}), radius={r}')

        return True, r, (cx, cy, r)

    def find_wafer(self, img):
        """
        Find black area that contains wafer circle by shape
        
        Parameters:
        -----------
        img : full screenshot (BGR format)
        
        Returns:
        --------
        wafer_img  : cropped wafer image or None if not found
        wafer_pos  : tuple (x, y, w, h) or None if not found
        circle_info: tuple (cx, cy, r) relative to wafer_img
        """
        if img is None:
            return None, None, None

        # Convert to HSV
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Step 1: Find all black areas
        black_mask = cv.inRange(hsv, self.lower_black, self.upper_black)

        # Clean up mask
        kernel     = np.ones((5, 5), np.uint8)
        black_mask = cv.morphologyEx(black_mask, cv.MORPH_OPEN,  kernel)
        black_mask = cv.morphologyEx(black_mask, cv.MORPH_CLOSE, kernel)

        if self.debug:
            cv.imshow('Black Mask', black_mask)

        # Step 2: Find all black contours
        contours, _ = cv.findContours(black_mask,
                                       cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            if self.debug:
                print("No black areas found!")
            return None, None, None

        if self.debug:
            print(f'Found {len(contours)} black areas')

        # Step 3: For each black area check if it has circle shape inside
        best_region      = None
        best_pos         = None
        best_score       = 0
        best_circle_info = None

        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)

            # Skip too small areas
            if area < self.min_area:
                continue

            # Get bounding box
            x, y, w, h = cv.boundingRect(contour)

            # Check aspect ratio (wafer panel is roughly square)
            aspect_ratio = w / h
            if not (0.5 <= aspect_ratio <= 2.0):
                continue

            if self.debug:
                print(f'Checking region {i+1}: '
                      f'area={area:.0f}, '
                      f'size={w}x{h}, '
                      f'ratio={aspect_ratio:.2f}')

            # Crop this region
            region_img = img[y:y+h, x:x+w]

            # Step 4: Check if this region has circle shape
            has_circle, score, circle_info = self._has_wafer_circle(region_img)

            if has_circle and score > best_score:
                best_score       = score
                best_region      = region_img
                best_pos         = (x, y, w, h)
                best_circle_info = circle_info
                if self.debug:
                    print(f'  ✅ Circle found! radius={score}')
            else:
                if self.debug:
                    print(f'  ❌ No circle shape')

        if best_region is None:
            if self.debug:
                print("No wafer circle found in any black area!")
            return None, None, None

        # Add padding to best region
        x, y, w, h = best_pos
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(img.shape[1] - x, w + self.padding * 2)
        h = min(img.shape[0] - y, h + self.padding * 2)

        wafer_img = img[y:y+h, x:x+w]
        wafer_img = np.ascontiguousarray(wafer_img)

        if self.debug:
            print(f'Final wafer position: x={x}, y={y}, w={w}, h={h}')

        return wafer_img, (x, y, w, h), best_circle_info

    def draw_wafer_box(self, img, wafer_pos, color=(0, 255, 0)):
        """Draw bounding box around detected wafer panel"""
        if wafer_pos is None:
            return img

        x, y, w, h = wafer_pos
        cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv.putText(img,
                   f'Wafer Map ({w}x{h})',
                   (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)
        return img

    def draw_circle(self, img, circle_info, color=(0, 255, 255)):
        """
        Draw detected circle on wafer image
        
        Parameters:
        -----------
        img         : wafer image
        circle_info : tuple (cx, cy, r) from find_wafer()
        color       : circle color BGR default=yellow
        """
        if circle_info is None:
            return img

        cx, cy, r = circle_info
        # Draw circle outline
        cv.circle(img, (cx, cy), r,     color, 2)
        # Draw center point
        cv.circle(img, (cx, cy), 3,     color, -1)
        # Draw label
        cv.putText(img,
                   f'Circle r={r}',
                   (cx - 50, cy - r - 10),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.6, color, 2)
        return img

    def show_mask(self, img):
        """Show black color mask for debugging"""
        hsv  = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_black, self.upper_black)
        cv.imshow('Color Mask (white = black areas)', mask)
