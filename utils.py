from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def path_to_tensor(img_path, imagedims):
    img = image.load_img(img_path, target_size=(imagedims[0], imagedims[1]))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)
