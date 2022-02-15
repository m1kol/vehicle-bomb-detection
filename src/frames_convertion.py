import numpy as np


class FrameConverter:
    """
    A class to perform image conversion to inter-frames differences.

    :param convert_type: A type of conversion to perform. One of ``freq`` or ``ampl``.
    :param epsilon: An epsion value for frequency (``freq``) conversion type.
    """

    def __init__(self, convert_type: str, epsilon: int = None):
        self.convert_type = convert_type
        self.epsilon = epsilon

    @staticmethod
    def _get_amplitude_difference(frames):
        interframe_diffs = np.array(
            [np.abs(frames[i + 1] - frames[i]) for i in range(len(frames) - 1)]
        )

        return interframe_diffs.mean(axis=0).astype(np.uint8)

    @staticmethod
    def _get_frequency_difference(frames, epsilon):
        interframe_diffs = np.array(
            [
                (np.abs(frames[i + 1] - frames[i]) > epsilon)
                for i in range(len(frames) - 1)
            ]
        )

        return 255 * interframe_diffs.mean(axis=0).astype(np.uint8)

    def convert(self, frames, compute_window: int = 2, step: int = 1):
        """
        Convert input ``frames`` images to the inter-frame differences based on the
        specified convert type.

        :param frames: Input frames for conversion.
        :param compute_window: Number of frames to use for inter-frames differences calculation. Default: ``2``.
        :param step: A value step between frames in the window. Default: ``1``.

        :return: Converted frames.
        """
        if step > 1:
            frames = frames[::step]

        converted_frames = []
        for i in range(len(frames) - compute_window + 1):
            if self.convert_type == "freq":
                converted_frames.append(
                    self._get_frequency_difference(
                        frames[i : i + compute_window], self.epsilon
                    )
                )
            elif self.convert_type == "ampl":
                converted_frames.append(
                    self._get_amplitude_difference(frames[i : i + compute_window])
                )

        return converted_frames


def get_frequency_diff(frames, epsilon: int):
    interframe_diffs = np.array(
        [np.abs(frames[i + 1] - frames[i]) > epsilon for i in range(len(frames) - 1)]
    )

    return 255 * interframe_diffs.mean(axis=0).astype(
        np.uint8
    )  # 255 * np.divide(interframe_diffs, len(frames)).astype(np.uint8)


def get_aplitude_diff(frames):
    interframe_diffs = np.array(
        [np.abs(frames[i + 1] - frames[i]) for i in range(len(frames) - 1)]
    )

    return np.divide(interframe_diffs, len(frames)).astype(np.uint8)


def frequency_conversion(frames, epsilon, compute_window: int = 2, step: int = 1):
    if step > 1:
        frames = frames[::step]

    converted_frames = []
    for i in range(len(frames) - compute_window + 1):
        converted_frames.append(
            get_frequency_diff(frames[i : i + compute_window], epsilon)
        )

    return converted_frames


def amplitude_conversion(frames, compute_window: int = 2, step: int = 1):
    if step > 1:
        frames = frames[::step]

    converted_frames = []
    for i in range(len(frames) - compute_window + 1):
        converted_frames.append(get_aplitude_diff(frames[i : i + compute_window]))

    return converted_frames
