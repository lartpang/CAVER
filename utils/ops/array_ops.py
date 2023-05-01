import os

import cv2
import numpy as np


def _normalize(data_array: np.ndarray) -> np.ndarray:
    max_pred_array = data_array.max()
    min_pred_array = data_array.min()
    if max_pred_array != min_pred_array:
        data_array = (data_array - min_pred_array) / (max_pred_array - min_pred_array)
    return data_array


def clip_to_normalize(data_array: np.ndarray, clip_range: tuple = None) -> np.ndarray:
    clip_range = sorted(clip_range)
    if len(clip_range) == 3:
        clip_min, clip_mid, clip_max = clip_range
        assert 0 <= clip_min < clip_mid < clip_max <= 1, clip_range
        lower_array = data_array[data_array < clip_mid]
        higher_array = data_array[data_array > clip_mid]
        if lower_array.size > 0:
            lower_array = np.clip(lower_array, a_min=clip_min, a_max=1)
            max_lower = lower_array.max()
            lower_array = _normalize(lower_array) * max_lower
            data_array[data_array < clip_mid] = lower_array
        if higher_array.size > 0:
            higher_array = np.clip(higher_array, a_min=0, a_max=clip_max)
            min_lower = higher_array.min()
            higher_array = _normalize(higher_array) * (1 - min_lower) + min_lower
            data_array[data_array > clip_mid] = higher_array
    elif len(clip_range) == 2:
        clip_min, clip_max = clip_range
        assert 0 <= clip_min < clip_max <= 1, clip_range
        if clip_min != 0 and clip_max != 1:
            data_array = np.clip(data_array, a_min=clip_min, a_max=clip_max)
        data_array = _normalize(data_array)
    elif clip_range is None:
        data_array = _normalize(data_array)
    else:
        raise NotImplementedError
    return data_array


def minmax(array: np.ndarray):
    max_value = array.max()
    min_value = array.min()
    if max_value == min_value:
        if max_value == 0:
            return array
        else:
            return array / max_value
    return (array - min_value) / (max_value - min_value)


def save_array_as_image(data_array: np.ndarray, save_name: str, save_dir: str):
    """
    save the ndarray as a image

    Args:
        data_array: np.float32 the max value is less than or equal to 1
        save_name: with special suffix
        save_dir: the dirname of the image path
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    if data_array.dtype != np.uint8:
        if data_array.max() > 1:
            raise Exception("the range of data_array has smoe errors")
        data_array = (data_array * 255).astype(np.uint8)
    cv2.imwrite(save_path, data_array)
