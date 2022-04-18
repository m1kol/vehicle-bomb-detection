from turtle import back
import numpy as np
import cv2


def convert_to_grayscale(img: np.ndarray, ksize: tuple = (3, 3)) -> np.ndarray:
    # assuming that if shape is equal to 2 than it's already a grayscale image
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, ksize=ksize, sigmaX=0)

    return img


class BasicDetector:
    """
    Basic detector that uses a reference background image to get the difference with
    the current frame.
    """

    def __init__(self, background: np.ndarray):
        """
        Constructor method.

        :param background: Background reference image.
        :type background: np.ndarray
        """
        self.background = convert_to_grayscale(background)

    def detect(self, frame: np.ndarray, diff_threshold: int, area_threshold: int = 50):
        frame_diff = self.get_frame_difference(frame, diff_threshold)

        contours, _ = cv2.findContours(
            frame_diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < area_threshold:
                continue

            bboxes.append(cv2.boundingRect(contour))

        return bboxes

    def get_frame_difference(self, frame: np.ndarray, threshold: int):
        frame = convert_to_grayscale(frame)

        frame_diff = cv2.absdiff(self.background, frame)
        frame_diff = cv2.dilate(frame_diff, np.ones((5, 5)), iterations=1)
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
        alpha: float = 0.5,
    ):
        if background is not None:
            self.background = convert_to_grayscale(background).astype(np.float32)
        else:
            self.background = background

        self.alpha = alpha

    def detect(self, frame: np.ndarray, diff_threshold: int, area_threshold: int = 50):
        if self.background is None:
            self.background = convert_to_grayscale(frame).astype(np.float32)

            return []

        frame = convert_to_grayscale(frame)
        frame_diff = self.get_frame_difference(frame, diff_threshold)

        contours, _ = cv2.findContours(
            frame_diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < area_threshold:
                continue

            bboxes.append(cv2.boundingRect(contour))

        # update background
        cv2.accumulateWeighted(frame, self.background, alpha=self.alpha)

        return bboxes

    def get_frame_difference(self, frame: np.ndarray, threshold: int):
        if self.background is None:
            self.background = convert_to_grayscale(frame).astype(np.float32)

            return []

        frame = convert_to_grayscale(frame)

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

    def detect(self, frame, area_threshold: int = 50):
        frame_diff = self.get_frame_difference(frame)

        contours, _ = cv2.findContours(
            frame_diff, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) < area_threshold:
                continue

            bboxes.append(cv2.boundingRect(contour))

        return bboxes

    def get_frame_difference(self, frame):
        frame = convert_to_grayscale(frame)

        foreground = self.background_substractor.apply(frame)

        return foreground
