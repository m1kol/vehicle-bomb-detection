import cv2
import numpy as np
from PIL import Image


def get_object_mask(obj: np.ndarray):
    obj_image = Image.fromarray(obj).convert("L")
    obj_image_arr = np.asarray(obj_image)
    obj_image_mask = (obj_image_arr < 230).astype(np.uint8) * 255
    obj_image_mask = Image.fromarray(obj_image_mask)

    return obj_image_mask


def shift_image(image: np.ndarray, x: int, y: int):
    shift_matrix = np.array(
        [
            [1, 0, x],
            [0, 1, y]
        ],
        dtype=np.float32
    )

    (rows, cols) = image.shape[:2]
    image_shifted = cv2.warpAffine(image, shift_matrix, (cols, rows))

    return image_shifted


def get_object_shift(amplitude, frequency, time):
    return amplitude * np.sin(2 * np.pi * frequency * time)


def generate_data(
        obj1: np.ndarray,
        obj2: np.ndarray,
        obj1_freq: float,
        obj1_ampl: int,
        obj2_freq: float,
        obj2_ampl: int,
        obj2_coords,
        time_step: float,
        generation_time: float
):
    generated_data = []
    elapsed_time = 0
    while elapsed_time < generation_time:
        # get objects shift
        obj1_shift = int(get_object_shift(obj1_ampl, obj1_freq, elapsed_time))
        obj2_shift = int(get_object_shift(obj2_ampl, obj2_freq, elapsed_time))

        # shift the fist object (background)
        obj1_shifted = shift_image(obj1, x=0, y=obj1_shift)

        # set the second object coordinates on the first object
        # accounting for first and second objects shifts
        # (in the center right now)
        # obj2_x_coord = int(obj1.shape[1] // 2)
        # obj2_y_coord = int(obj1.shape[0] // 2 + obj1_shift + obj2_shift)
        obj2_x_coord, obj2_y_coord = obj2_coords
        obj2_y_coord += obj1_shift + obj2_shift

        # get object 2 mask
        obj2_mask = get_object_mask(obj2)

        # place object 2 on object 1 as image
        obj1_image = Image.fromarray(obj1_shifted)
        obj2_image = Image.fromarray(obj2)

        obj1_image.paste(obj2_image, (obj2_x_coord, obj2_y_coord), mask=obj2_mask)

        generated_data.append(np.asarray(obj1_image, dtype=np.uint8))
        elapsed_time += time_step

    return generated_data
