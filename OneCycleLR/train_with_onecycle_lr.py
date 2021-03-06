# -*- coding: utf-8 -*-
import os
import keras
from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.utils.vis_utils import plot_model
from keras.utils.generic_utils import CustomObjectScope
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from common_models.MobileNet_V2 import MobileNetV2
from clr import OneCycleLR
import pandas as pd
import argparse
import pickle
from PIL import ImageFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
                help="path to output label binarizer")
ap.add_argument("-s", "--save_model",
                default='../model_outputs/best.weights.dogmobilenetv2.h5',
                help='the restored path of best weights of model')
ap.add_argument("--load_model",
                default='../trained_models/best.weights.dogmobilenetv2.h5',
                help='the  path of trained model')
args = vars(ap.parse_args())


if not os.path.exists('weights/'):
    os.makedirs('weights/')
weights_file = 'weights/mobilenet_v2.h5'
# initial HyParameters and training switches
EPOCHS = 100
INIT_LR = 1e-4
DECAY = 4e-5
OPT = SGD(lr=0.0001, momentum=0.9, nesterov=True)
BS = 664
NUM_SAMPLES = 6680
IMAGE_DIMS = (96, 96, 3)
freeze_until = None
train_from = 'trained_model'


def create_data_generator(data_path):
    train_generator = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                         rescale=1./255,
                                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                         horizontal_flip=True, fill_mode="nearest",
                                         )
    train_gen = train_generator.flow_from_directory(
        os.path.join(data_path, "train"),
        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
        shuffle=True,
        batch_size=BS,
    )

    valid_generator = ImageDataGenerator(rescale=1./255)
    valid_gen = valid_generator.flow_from_directory(
        os.path.join(data_path, "valid"),
        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
        shuffle=True,
        batch_size=BS,
    )
    print(train_gen.class_indices)
    return train_gen, valid_gen


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
    train_gen, valid_gen = create_data_generator(data_path)
    # print('[INFO] serializeing label binarizer...')
    # f = open(args['labelbin'], 'wb')
    # d = train_gen.class_indices
    # pickle.dump(d, f)
    if train_from == 'trained_weights':
        model = load_model_from_trained_weights(imagedims=IMAGE_DIMS, nb_classes=len(train_gen.class_indices),
                                                weights=args['weight_path'],
                                                freeze_until=freeze_until)
    elif train_from == 'trained_model':
        model = load_model_from_trained_model()
    else:
        model = load_models(imagedims=IMAGE_DIMS, nb_classes=len(train_gen.class_indices))
    print('[INFO] compiling model...')
    model.compile(loss="categorical_crossentropy", optimizer=OPT,
                  metrics=["accuracy"])
    plot_model(model, to_file='../model_outputs/architecture.png',
               show_layer_names=True, show_shapes=True)
    checkpoint = ModelCheckpoint(filepath=args['save_model'], monitor='val_loss', verbose=0,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    stop_early = EarlyStopping(monitor='val_loss', min_delta=.0, patience=40, verbose=0, mode='auto')
    lr_manager = OneCycleLR(NUM_SAMPLES, EPOCHS, BS, max_lr=0.001,
                            maximum_momentum=0.9, verbose=True)
    callbacks = [checkpoint, stop_early, lr_manager]
    H = model.fit_generator(train_gen,
                            validation_data=valid_gen,
                            epochs=EPOCHS,
                            #steps_per_epoch=209,
                            callbacks=callbacks,
                            verbose=1
                            )
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["acc"], label="train_acc")
    plt.plot(H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig('../model_outputs/acc_loss.png')
    df = pd.DataFrame.from_dict(H.history)
    df.to_csv('../model_outputs/hist.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
