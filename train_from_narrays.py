# USAGE
import keras
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.utils.generic_utils import CustomObjectScope
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Nadam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from common_models.MobileNetV2_beta import MobileNetV2
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import random
import pickle
import cv2
import os
from PIL import ImageFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_path",
                default='dataset',
                help='the direction of datas')
ap.add_argument("-w", "--weight_path",
                default='model_outputs/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5',
                help='the path of the pre_trained weights')
ap.add_argument("-l", "--labelbin",
                default='model_outputs/lb.pickle',
                help="path to output label binarizer")
ap.add_argument("-s", "--save_model",
                default='model_outputs/best.weights.pokmobilenetv2.h5',
                help='the restored path of best weights of model')
ap.add_argument("--load_model",
                default='trained_models/best.weights.pokmobilenetv2.h5',
                help='the  path of f trained model')
args = vars(ap.parse_args())

# init hyperparameter and training switches
EPOCHS = 200
INIT_LR = 1e-4
BS = 128
IMAGE_DIMS = (96, 96, 3)
freeze_until = None
train_from = 'trained_model'


def load_datas_and_label_binarize():
    data = []
    labels = []
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(args["data_path"])))
    random.seed(42)
    random.shuffle(imagePaths)
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
    print("[INFO] serializing label binarizer...")
    f = open(args["labelbin"], "wb")
    f.write(pickle.dumps(lb))
    f.close()
    return trainX, testX, trainY, testY, lb


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


def main():
    print("[INFO] compiling model...")
    trainX, testX, trainY, testY, lb = load_datas_and_label_binarize()
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    if train_from == 'trained_weights':
        model = load_model_from_trained_weights(imagedims=IMAGE_DIMS, nb_classes=len(lb.classes_),
                                                weights=args['weight_path'],
                                                freeze_until=freeze_until)
    elif train_from == 'trained_model':
        model = load_model_from_trained_model()
    else:
        model = load_models(imagedims=IMAGE_DIMS, nb_classes=len(lb.classes_))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    #opt = Nadam
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    plot_model(model, to_file='model_outputs/architecture.png',
               show_layer_names=True, show_shapes=True)
    # train the network
    print("[INFO] training network...")
    checkpoint = ModelCheckpoint(filepath=args['save_model'], monitor='val_loss', verbose=0,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    stopearly = EarlyStopping(monitor='val_loss', min_delta=.0, patience=30, verbose=0, mode='auto')
    callbacks = [checkpoint, stopearly]
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        callbacks=callbacks,
        epochs=EPOCHS, verbose=1)
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
    plt.savefig('model_outputs/acc_loss.png')
    df = pd.DataFrame.from_dict(H.history)
    df.to_csv('model_outputs/hist.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()