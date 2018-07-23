from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import pickle

BS = 836
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_path",
                default='datasets',
                help='the direction of datas')
ap.add_argument("-m", "--model",
                default='trained_models/best.weights.mobilenetv2.h5',
                help="path to trained model model")
ap.add_argument("-l", "--labelbin",
                default='model_outputs/lb.pickle',
                help="path to label binarizer")
args = vars(ap.parse_args())

IMAGE_DIMS = (96, 96, 3)
test_generator = ImageDataGenerator(rescale=1./255)
test_gen = test_generator.flow_from_directory(
    os.path.join(args['data_path'], "test"),
    target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
    shuffle=True,
    )
print('[INFO] serializeing label binarizer...')
f = open(args['labelbin'], 'wb')
d = test_gen.class_indices
pickle.dump(d, f)

print("[INFO] loading network...")
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D':
                       keras.applications.mobilenet.DepthwiseConv2D}):
                       model = load_model(args["model"])
print(model.evaluate_generator(test_gen))