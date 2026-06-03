import cv2 as cv
import numpy as np


class CircleDetector:
    """
    Detects the biggest circle inside a given image/frame
    using OpenCV Hough Circle Transform.
    """

    def __init__(self,
                 dp         = 1.2,
                 min_dist   = 100,
                 param1     = 80,
                 param2     = 40,
                 min_radius = 80,
                 max_radius = 500):
        """
        Parameters
        ----------
        dp         : inverse ratio of accumulator resolution to image resolution
        min_dist   : minimum distance between detected circle centres (px)
        param1     : upper threshold for the Canny edge detector
        param2     : accumulator threshold — lower = more circles detected
        min_radius : minimum circle radius to search for (px)
        max_radius : maximum circle radius to search for (px), 0 = unlimited
        """
        self.dp         = dp
        self.min_dist   = min_dist
        self.param1     = param1
        self.param2     = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply Gaussian blur."""
        gray    = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (15, 15), 3)
        return blurred

    def _run_hough(self, blurred: np.ndarray):
        """Run HoughCircles on a pre-processed (blurred) image."""
        circles = cv.HoughCircles(
            blurred,
            cv.HOUGH_GRADIENT,
            dp        = self.dp,
            minDist   = self.min_dist,
            param1    = self.param1,
            param2    = self.param2,
            minRadius = self.min_radius,
            maxRadius = self.max_radius
        )
        return circles

  
    def find_biggest_circle(self, image: np.ndarray):
        """
        Find the biggest circle in *image*.

        Returns
        -------
        (cx, cy, radius)  – integers  → biggest circle found
        None              → no circle detected
        """
        if image is None:
            return None

        blurred = self._preprocess(image)
        circles = self._run_hough(blurred)

        if circles is None:
            return None

        # pick the circle with the largest radius
        circles = np.round(circles[0, :]).astype("int")
        biggest = max(circles, key=lambda c: c[2])
        return tuple(biggest)   # (cx, cy, r)

    def draw_biggest_circle(self,
                            image:  np.ndarray,
                            circle: tuple,
                            label:  str = "") -> np.ndarray:
        """
        Draw the detected circle on a copy of *image*.

        Draws
        -----
        - Green  ring        → boundary of the circle
        - Red    filled dot  → centre point
        - Yellow cross-hair  → horizontal & vertical axis lines
        - Green  text label  → radius + centre coordinates

        Returns
        -------
        Annotated copy of the input image.
        """
        output    = image.copy()
        cx, cy, r = circle

        # boundary ring
        cv.circle(output, (cx, cy), r,  (0, 255,   0), 3)

        # centre dot
        cv.circle(output, (cx, cy), 6,  (0,   0, 255), -1)

        # cross-hair
        cv.line(output, (cx - r, cy), (cx + r, cy), (0, 255, 255), 1)
        cv.line(output, (cx, cy - r), (cx, cy + r), (0, 255, 255), 1)

        # text label
        text = label if label else f"Biggest Circle  r={r}px  centre=({cx},{cy})"
        cv.putText(output, text,
                   (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   (0, 255, 0), 2)

        return output

    def process(self, image: np.ndarray):
        """
        Find AND draw the biggest circle in one call.

        Returns
        -------
        annotated_image : np.ndarray  – image with circle drawn (or error text)
        circle_info     : tuple | None – (cx, cy, r) or None
        """
        circle = self.find_biggest_circle(image)

        if circle is not None:
            annotated = self.draw_biggest_circle(image, circle)
        else:
            annotated = image.copy()
            cv.putText(annotated, "No circle detected",
                       (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75,
                       (0, 0, 255), 2)

        return annotated, circle
