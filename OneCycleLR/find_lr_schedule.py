# -*- coding: utf-8 -*-
import os
import keras
from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.utils.generic_utils import CustomObjectScope
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
from common_models.MobileNet_V2 import MobileNetV2
from clr import LRFinder
import numpy as np
import pandas as pd
import argparse
import pickle
import random
import cv2
from PIL import ImageFile
import matplotlib
matplotlib.use("Agg")
ImageFile.LOAD_TRUNCATED_IMAGES = True


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_path",
                default='../datasets',
                help='the direction of datas')
ap.add_argument("-w", "--weight_path",
                default='../model_outputs/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5',
                help='the path of the pre_trained weights')
ap.add_argument("-l", "--labelbin",
                default='../model_outputs/dog_lb.pickle',
                help="path to label binarizer")
ap.add_argument("-s", "--save_model",
                default='weights/mobilenet_v2_schedule.h5',
                help='the restored path of best weights of model')
ap.add_argument("--load_model",
                default='../trained_models/best.weights.dogmobilenetv2.h5',
                help='the  path of f trained model')
args = vars(ap.parse_args())


# initial HyParameters and training switches
EPOCHS = 1
INIT_LR = 1e-4
DECAY = 4e-5
OPT = SGD(lr=0.0005, momentum=0.9, nesterov=True)
BS = 64
NUM_SAMPLES = 6680
IMAGE_DIMS = (96, 96, 3)
freeze_until = None
train_from = 'trained_model'
lr_finder_from = 'close_range_search'


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(96, 96))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


def create_valid_data():
    valid_files, valid_targets = load_dataset(os.path.join(args['data_path'], 'valid'))
    valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
    return valid_tensors, valid_targets


def create_data_generator(data_path):
    train_generator = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                         rescale=1./255,
                                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                         horizontal_flip=True, fill_mode="nearest")
    train_gen = train_generator.flow_from_directory(
        os.path.join(data_path, "train"),
        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
        shuffle=True,
        batch_size=BS)
    test_generator = ImageDataGenerator(rescale=1. / 255)
    test_gen = test_generator.flow_from_directory(
        os.path.join(data_path, "test"),
        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
        shuffle=True,
        batch_size=BS)
    print(train_gen.class_indices)
    return train_gen, test_gen


def load_models(imagedims, nb_classes):  # notes:name is same with keras.models.load_model,so change as load_models
    model = MobileNetV2.build(imagedims, nb_classes)
    return model


def load_model_from_trained_weights(imagedims, nb_classes, weights=None, freeze_until=None):
    model = MobileNetV2.build(imagedims, nb_classes)
    print("[INFO] loading weights...")
    model.load_weights(weights, by_name=False, skip_mismatch=False)
    model = Model(model.inputs, model.get_layer("dropout").output)
    if freeze_until:
        for layer in model.layers[:model.layers.index(model.get_layer(freeze_until))]:
            layer.trainable = False
    out = Dense(units=nb_classes, activation='softmax')(model.output)
    model = Model(model.inputs, out)
    return model


def load_model_from_trained_model():
    print("[INFO] loading network...")
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D':
                           keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(args["load_model"])
    return model


def main(data_path=args['data_path'], train_from=train_from):
    train_gen, test_gen = create_data_generator(data_path)
    valid_x, valid_y = create_valid_data()
    if train_from == 'trained_weights':
        model = load_model_from_trained_weights(imagedims=IMAGE_DIMS, nb_classes=len(train_gen.class_indices),
                                                weights=args['weight_path'],
                                                freeze_until=freeze_until)
    elif train_from == 'trained_model':
        model = load_model_from_trained_model()
    else:
        model = load_models(imagedims=IMAGE_DIMS, nb_classes=len(train_gen.class_indices))
    print('[INFO] compiling model...')
    model.compile(loss="categorical_crossentropy", optimizer=OPT, metrics=["accuracy"])

    checkpoint = ModelCheckpoint(filepath=args['save_model'], monitor='val_loss', verbose=0,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    stop_early = EarlyStopping(monitor='val_loss', min_delta=.0, patience=40, verbose=0, mode='auto')
    if lr_finder_from == 'large_range_search':
        '''Exponential lr finder,
           USE THIS FOR A LARGE RANGE SEARCH
           Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
        '''
        lr_finder = LRFinder(NUM_SAMPLES, BS, minimum_lr=1e-3, maximum_lr=10.,
                             lr_scale='exp',
                             validation_data=(valid_x, valid_y),  # use the validation data for losses
                             validation_sample_rate=5,
                             save_dir='weights/', verbose=True)
    elif lr_finder_from == 'close_range_search':
        '''LINEAR lr finder,
           USE THIS FOR A CLOSE RANGE SEARCH
           Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
        '''
        lr_finder = LRFinder(NUM_SAMPLES, BS, minimum_lr=1e-5, maximum_lr=1e-2,
                             lr_scale='exp',
                             validation_data=(valid_x, valid_y),  # use the validation data for losses
                             validation_sample_rate=5,
                             save_dir='weights/', verbose=True)
    callbacks = [checkpoint, stop_early, lr_finder]
    H = model.fit_generator(train_gen,
                            validation_data=(valid_x, valid_y),
                            epochs=EPOCHS,
                            #steps_per_epoch=209,
                            callbacks=callbacks,
                            verbose=1
                            )
    lr_finder.plot_schedule(clip_beginning=10, clip_endding=5)

    # scores = model.evaluate(test_gen, batch_size=BS)
    # for score, metric_name in zip(scores, model.metrics_names):
    #     print("%s : %0.4f" % (metric_name, score))


if __name__ == "__main__":
    main()
