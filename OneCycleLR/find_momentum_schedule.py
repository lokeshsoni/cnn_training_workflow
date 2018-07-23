# -*- coding: utf-8 -*-
import os
import keras
from keras import backend as K
from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.utils import np_utils
from common_models.MobileNet_V2 import MobileNetV2
from clr import LRFinder
import numpy as np
import argparse
from PIL import ImageFile
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
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
OPT = SGD(lr=0.0001, momentum=0.9, nesterov=True)
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
    MOMENTUMS = [0.9, 0.95, 0.99]
    for momentum in MOMENTUMS:
        K.clear_session()
        # Learning rate range obtained from `find_lr_schedule.py`
        # NOTE : Minimum is 10x smaller than the max found above !
        # NOTE : It is preferable to use the validation data here to get a correct value
        lr_finder = LRFinder(NUM_SAMPLES, BS, minimum_lr=0.0001, maximum_lr=0.001,
                             validation_data=(valid_x, valid_y),
                             validation_sample_rate=5,
                             lr_scale='linear', save_dir='weights/momentum/momentum-%s/' % str(momentum),
                             verbose=True)

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

        # set the weight_decay here !
        # lr doesnt matter as it will be over written by the callback
        optimizer = SGD(lr=0.001, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        callbacks = [lr_finder]
        H = model.fit_generator(train_gen,
                                validation_data=(valid_x, valid_y),
                                epochs=EPOCHS,
                                #steps_per_epoch=209,
                                callbacks=callbacks,
                                verbose=1
                                )
    for momentum in MOMENTUMS:
        directory = 'weights/momentum/momentum-%s/' % str(momentum)

        losses, lrs = LRFinder.restore_schedule_from_dir(directory, 10, 5)
        plt.plot(lrs, losses, label='momentum=%0.2f' % momentum)

    plt.title("Momentum")
    plt.xlabel("Learning rate")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
