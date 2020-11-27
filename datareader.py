'''Utilities to process and load images'''
import numpy as np
import os

from skimage import io, transform

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
    file_names = os.listdir(dir_path)
    file_names = file_names[:10000] # load 10000 images
    images = np.empty([len(file_names), 3, 128, 128], dtype=np.float32)
    print('Loading {} images from {}...'.format(len(file_names), dir_path))

    for i, file_name in enumerate(file_names):
        image_path = os.path.join(dir_path, file_name)
        images[i] = _load_image(image_path)

        if i > 0 and i % 500 == 0:
            # give report every 500 loaded images
            print('Loaded {}/{} images'.format(i, len(images)))

    return images
