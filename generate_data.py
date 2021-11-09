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


def generate_data(background: np.ndarray, obj: np.ndarray):
    x, y = np.random.randint(1, 10, size=2)
    background_shifted = shift_image(background, x, y)

    image_x, image_y = np.random.randint(1, 4, size=2)
    obj_mask = get_object_mask(obj)
    background_shifted_img = Image.fromarray(background_shifted)
    obj_img = Image.fromarray(obj)
    background_shifted_img.paste(
        obj_img,
        (background_shifted_img.size[0] // 2 + x + image_x, background_shifted_img.size[1] // 2 + y + image_y),
        mask=obj_mask
    )

    return np.asarray(background_shifted_img)