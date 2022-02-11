import numpy as np
import cv2


class MotionDetector:
    def __init__(self, background_img, thresh):
        self.background_img = cv2.cvtColor(background_img, cv2.COLOR_RGB2GRAY)
        self.thresh = thresh

    def detect(self, frame, threshold=None):
        if not threshold:
            threshold = self.thresh
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

        frame_diff = cv2.absdiff(self.background_img, frame)
        frame_diff = cv2.dilate(frame_diff, np.ones((5, 5)), 1)
        frame_diff = cv2.threshold(frame_diff, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(frame_diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            # x, y, width, height = cv2.boundingRect(contour)
            bboxes.append(cv2.boundingRect(contour))

        return bboxes
