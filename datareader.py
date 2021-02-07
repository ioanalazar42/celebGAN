'''Utilities to process and load images'''
import numpy as np
import os

from skimage import io, transform
from timeit import default_timer as timer

def _center_crop_image(image):
    height = image.shape[0]
    width = image.shape[1]
    crop_size = height if height < width else width # if the width is larger we preserve the height and vice versa

    y = int((height - crop_size) / 2)
    x = int((width - crop_size) / 2)

    return image[y : crop_size, x : crop_size]

def _resize_image(image, width, height):
    return transform.resize(image, [height, width, 3], anti_aliasing=True, mode='constant')

def _mean_normalize(image):
    '''norm(img) : [0, 1] -> [-1, 1]'''
    return 2 * image - 1

def _load_image(path):
    image = io.imread(path)

    if image.ndim == 2:
        # grayscale to RGB
        image = np.dstack([image, image, image])

    image = _mean_normalize(_resize_image(_center_crop_image(image), 128, 128))

    # 128x128x3 to 3x128x128
    return image.transpose(2, 0, 1)

def load_images(dir_path):
    pre_load_time = timer()
    file_names = os.listdir(dir_path)
    file_names = file_names[:5000] # load images
    images = np.empty([len(file_names), 3, 128, 128], dtype=np.float32)
    print('Initial instructions took: {:.3f}s'.format(timer() - pre_load_time))
    print('Loading {} images from {}...'.format(len(file_names), dir_path))

    current_time = timer() 
    total_load_time = 0
    for i, file_name in enumerate(file_names):
        image_path = os.path.join(dir_path, file_name)
        images[i] = _load_image(image_path)

        if i > 0 and i % 500 == 0:
            # give report every 500 loaded images
            loaded_500 = timer() - current_time
            current_time = timer()
            total_load_time += loaded_500
            print('[{:.3f}]\t Loaded {}/{} images'.format(loaded_500, i, len(images)))
    print('Total load time: {:.3f}s'.format(total_load_time))
    return images
