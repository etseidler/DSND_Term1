from PIL import Image
from torch import from_numpy
import numpy as np

def process_image(image):
    resize_num = 256
    w, h = image.size
    (new_w, new_h) = (1 + (resize_num * w // h), resize_num) if w > h else (resize_num, 1 + (resize_num * h // w))
    resized = image.resize((new_w, new_h))
    
    resized_w, resized_h = resized.size
    w_margin = (resized_w - 224) // 2
    h_margin = (resized_h - 224) // 2
    (left, upper, right, lower) = (w_margin, h_margin, resized_w - w_margin, resized_h - h_margin)
    cropped = resized.crop((left, upper, right, lower))
    
    np_image = np.array(cropped)
    np_converted = np_image / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (np_converted - mean) / std
    
    reordered = np.transpose(normalized, (2, 0, 1))

    return reordered


def get_torched_image(image_path):
    img = Image.open(image_path)
    processed = process_image(img)
    # expand_dims used to account for model processing items in batches
    torched = from_numpy(np.expand_dims(processed, axis=0))
    return torched

