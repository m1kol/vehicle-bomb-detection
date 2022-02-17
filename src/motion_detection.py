from turtle import back
import numpy as np
import cv2


class BasicDetector:
    """
    Basic detector that uses a reference background image to get the difference with
    the current frame.
    """

    def __init__(self, background, thresh):
        """
        Constructor method.

        :param background: Background reference image.
        :type background: np.ndarray
        :param thresh: Threshold for an inter-frame difference.
        :type thresh: int
        """
        self.background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
        self.thresh = thresh

    def detect(self, frame, threshold=None):
        if not threshold:
            threshold = self.thresh

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

        frame_diff = cv2.absdiff(self.background, frame)
        frame_diff = cv2.dilate(frame_diff, np.ones((5, 5)), 1)
        frame_diff = cv2.threshold(
            frame_diff, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
        )[1]

        contours, _ = cv2.findContours(
            frame_diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            # x, y, width, height = cv2.boundingRect(contour)
            bboxes.append(cv2.boundingRect(contour))

        return bboxes

    def get_frame_difference(self, frame, threshold):
        if not threshold:
            threshold = self.thresh

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

        frame_diff = cv2.absdiff(self.background, frame)
        frame_diff = cv2.dilate(frame_diff, np.ones((5, 5)), 1)
        frame_diff = cv2.threshold(
            frame_diff, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
        )[1]

        return frame_diff


class AverageDetector:
    """
    Detector that stores a background as a mooving avarage of multiple frames.
    """

    def __init__(
        self,
        background: np.ndarray = None,
        thresh: int = None,
        alpha: float = None,
    ):
        self.background = background
        if self.background is not None:
            if self.background.shape[-1] == 3:
                self.background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
            self.background = self.background.astype(np.float32)

        if thresh:
            self.thresh = thresh
        else:
            self.thresh = 125

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = -1

    def detect(self, frame, threshold: int = None):
        if self.background is None:
            self.background = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            return []

        if not threshold:
            if self.thresh:
                threshold = self.thresh
            else:
                self.thresh = 120
                threshold = 120

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

        frame_diff = cv2.absdiff(frame, cv2.convertScaleAbs(self.background))
        frame_diff = cv2.dilate(frame_diff, np.ones((5, 5)), 1)
        frame_diff = cv2.threshold(
            frame_diff, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
        )[1]

        contours, _ = cv2.findContours(
            frame_diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            # x, y, width, height = cv2.boundingRect(contour)
            bboxes.append(cv2.boundingRect(contour))

        # update background
        cv2.accumulateWeighted(frame, self.background, alpha=self.alpha)

        return bboxes

    def get_frame_difference(self, frame, threshold):
        if self.background is None:
            self.background = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

            return []

        if not threshold:
            if self.thresh:
                threshold = self.thresh
            else:
                self.thresh = 120
                threshold = 120

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

        frame_diff = cv2.absdiff(frame, cv2.convertScaleAbs(self.background))
        frame_diff = cv2.dilate(frame_diff, np.ones((5, 5)), 1)
        frame_diff = cv2.threshold(
            frame_diff, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
        )[1]

        return frame_diff


class MogDetector:
    """
    Detector that separates background and foreground using Mixture of Gaussians method (MoG).
    """

    def __init__(self, background_ratio: float = None) -> None:
        self.background_ratio = background_ratio
        self.background_substractor = cv2.createBackgroundSubtractorMOG2()
        self.background_substractor.setShadowValue(0)

    def detect(self, frame, threshold: int = None):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

        foreground = self.background_substractor.apply(frame)

        contours, _ = cv2.findContours(
            foreground, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            # x, y, width, height = cv2.boundingRect(contour)
            bboxes.append(cv2.boundingRect(contour))

        return bboxes

    def get_frame_difference(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

        foreground = self.background_substractor.apply(frame)

        return foreground
