import cv2
import numpy as np


class FrameConverter:
    def __init__(self, convert_type: str, epsilon: float, window_size: int = None):
        self.convert_type = convert_type
        self.window_size = window_size
        self.epsilon = epsilon

    def convert_frames(self, frames, compute_window: int = 2):
        converted_frames = []
        for i in range(len(frames) - compute_window + 1):
            if self.convert_type == "freq":
                converted_frames.append(self._get_frequency_difference(
                    frames[i:i+compute_window],
                    self.epsilon
                ))
            elif self.convert_type == "ampl":
                converted_frames.append(self._get_amplitude_difference(frames[i:i+compute_window]))

        return converted_frames

    @staticmethod
    def _get_amplitude_difference(frames):
        interframe_diffs = np.array(
            [
                np.abs(frames[i + 1] - frames[i])
                for i in range(len(frames) - 1)
            ],
            dtype=np.uint8
        )

        return interframe_diffs.mean(axis=0)

    @staticmethod
    def _get_frequency_difference(frames, epsilon):
        interframe_diffs = np.array(
            [
                (np.abs(frames[i+1] - frames[i]) > epsilon)
                for i in range(len(frames)-1)
            ],
            dtype=np.uint8
        )

        return 255 * interframe_diffs.mean(axis=0)  # / len(frames)
