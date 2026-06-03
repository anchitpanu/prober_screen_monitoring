import cv2 as cv
import numpy as np


class CircleDetector:

    def __init__(self,
                 min_circularity = 0.7,
                 min_area        = 1000,
                 blur_kernel     = 5,
                 padding         = 5):

        self.min_circularity = min_circularity
        self.min_area        = min_area
        self.blur_kernel     = blur_kernel
        self.padding         = padding
        self.debug           = False

    # -------------------------------------------------------------------------
    def _preprocess(self, image: np.ndarray):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        mask_green = cv.inRange(hsv,
                                np.array([35,  40,  40]),
                                np.array([90, 255, 255]))
        mask_teal  = cv.inRange(hsv,
                                np.array([80,  40,  40]),
                                np.array([100, 255, 255]))

        mask   = cv.bitwise_or(mask_green, mask_teal)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        mask   = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask   = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)

        blurred = cv.GaussianBlur(mask,
                                  (self.blur_kernel, self.blur_kernel), 1)
        edges   = cv.Canny(blurred, 30, 150)

        return edges, mask

    # -------------------------------------------------------------------------
    def _get_circularity(self, contour) -> float:
        area      = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter ** 2)

    # -------------------------------------------------------------------------
    def _find_circle_contours(self, edges: np.ndarray, image_shape: tuple):
        contours, _ = cv.findContours(edges,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
        h, w        = image_shape[:2]
        circle_list = []
        margin      = 5

        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.min_area:
                continue

            circularity = self._get_circularity(contour)
            if circularity < self.min_circularity:
                continue

            (cx, cy), radius = cv.minEnclosingCircle(contour)
            cx, cy, radius   = int(cx), int(cy), int(radius)

            if (cx - radius) < margin:       continue
            if (cy - radius) < margin:       continue
            if (cx + radius) > (w - margin): continue
            if (cy + radius) > (h - margin): continue

            circle_list.append((cx, cy, radius))

        return circle_list

    # -------------------------------------------------------------------------
    def find_biggest_circle(self, image: np.ndarray):
        """
        Returns (cx, cy, radius) or None
        """
        if image is None:
            return None

        edges, mask = self._preprocess(image)
        circle_list = self._find_circle_contours(edges, image.shape)

        if not circle_list:
            return None

        biggest        = max(circle_list, key=lambda c: c[2])
        cx, cy, radius = biggest

        if self.debug:
            print(f"[ShapeDetect] circles found  : {len(circle_list)}")
            print(f"[ShapeDetect] biggest centre : ({cx}, {cy})")
            print(f"[ShapeDetect] biggest radius : {radius}px")

        return cx, cy, radius

    # -------------------------------------------------------------------------
    def draw_biggest_circle(self,
                            image:  np.ndarray,
                            circle: tuple) -> np.ndarray:
        """
        circle = (cx, cy, radius)
        """
        output    = image.copy()
        cx, cy, r = circle

        # green boundary ring
        cv.circle(output, (cx, cy), r,  (0, 255,   0), 3)

        # red centre dot
        cv.circle(output, (cx, cy), 6,  (0,   0, 255), -1)

        # yellow cross-hair
        cv.line(output, (cx - r, cy), (cx + r, cy), (0, 255, 255), 1)
        cv.line(output, (cx, cy - r), (cx, cy + r), (0, 255, 255), 1)

        # labels
        cv.putText(output,
                   f"Wafer Circle  r={r}px",
                   (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   (0, 255, 0), 2)

        cv.putText(output,
                   f"Centre = ({cx}, {cy})",
                   (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65,
                   (255, 255, 255), 2)

        return output

    # -------------------------------------------------------------------------
    def crop_circle(self,
                    image:  np.ndarray,
                    circle: tuple) -> np.ndarray:
        """
        Crop and show ONLY the circle area from image.
        Outside the circle = black (masked).

        Parameters
        ----------
        image  : original screenshot
        circle : (cx, cy, radius)

        Returns
        -------
        cropped : np.ndarray - square crop containing only the circle
                               outside area is black
        """
        if image is None or circle is None:
            return None

        cx, cy, r = circle
        h, w      = image.shape[:2]
        padding = self.padding

        # expand radius by padding
        r_padded = r + padding

        # create circular mask with padded radius
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.circle(mask, (cx, cy), r_padded, 255, -1)       # white filled circle

        # apply mask - black outside circle
        masked = cv.bitwise_and(image, image, mask=mask)

        # crop bounding box around circle
        x1 = max(cx - r_padded, 0)
        y1 = max(cy - r_padded, 0)
        x2 = min(cx + r_padded, w)
        y2 = min(cy + r_padded, h)

        cropped = masked[y1:y2, x1:x2]

        return cropped

    # -------------------------------------------------------------------------
    def show_debug_windows(self, image: np.ndarray):
        edges, mask = self._preprocess(image)
        debug_img   = image.copy()

        contours, _ = cv.findContours(edges,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.min_area:
                continue

            circularity = self._get_circularity(contour)
            color       = (0, 255, 0) if circularity >= self.min_circularity \
                                      else (0, 0, 255)

            cv.drawContours(debug_img, [contour], -1, color, 2)

            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.putText(debug_img,
                           f"{circularity:.2f}",
                           (cx, cy),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           color, 1)

        # cv.imshow('[Debug] Color Mask',   mask)
        # cv.imshow('[Debug] Edges',        edges)
        # cv.imshow('[Debug] All Contours', debug_img)

    # -------------------------------------------------------------------------
    def process(self, image: np.ndarray):
        """
        Returns
        -------
        annotated_image : np.ndarray  - full image with circle drawn
        circle_cropped  : np.ndarray  - circle area only (black outside)
        circle_info     : (cx, cy, radius) or None

        Usage
        -----
        annotated, cropped, circle_info = detector.process(screenshot)
        if circle_info is not None:
            cx, cy, r = circle_info
        """
        circle = self.find_biggest_circle(image)

        if circle is not None:
            cx, cy, r   = circle
            annotated   = self.draw_biggest_circle(image, circle)
            cropped     = self.crop_circle(image, circle)           # NEW
        else:
            annotated   = image.copy()
            cropped     = None                                       # NEW
            cv.putText(annotated,
                       "No wafer circle detected",
                       (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75,
                       (0, 0, 255), 2)

        if self.debug:
            self.show_debug_windows(image)

        return annotated, cropped, circle                            # NEW - returns 3 things