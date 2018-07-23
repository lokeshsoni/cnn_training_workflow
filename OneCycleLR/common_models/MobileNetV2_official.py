from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import h5py
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Add

from keras.layers import Dense
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine import get_source_inputs

from keras import backend as K

class MobileNetV2():
    @staticmethod
    def build(imagedims, nb_classes):
        def relu6(x):
            return K.relu(x, max_value=6)

        def preprocess_input(x):
            """Preprocesses a numpy array encoding a batch of images.
            This function applies the "Inception" preprocessing which converts
            the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
            function is different from `imagenet_utils.preprocess_input()`.
            # Arguments
                x: a 4D numpy array consists of RGB values within [0, 255].
            # Returns
                Preprocessed array.
            """
            x /= 128.
            x -= 1.
            return x.astype(np.float32)

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _first_inverted_res_block(inputs,
                                      stride,
                                      alpha, filters, block_id):
            in_channels = inputs._keras_shape[-1]
            prefix = 'features.' + str(block_id) + '.conv.'
            pointwise_conv_filters = int(filters * alpha)
            pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

            # Depthwise
            x = DepthwiseConv2D(kernel_size=3,
                                strides=stride, activation=None,
                                use_bias=False, padding='same',
                                name='mobl%d_conv_depthwise' %
                                     block_id)(inputs)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name='bn%d_conv_depthwise' %
                                        block_id)(x)
            x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

            # Project
            x = Conv2D(pointwise_filters,
                       kernel_size=1,
                       padding='same',
                       use_bias=False,
                       activation=None,
                       name='mobl%d_conv_project' %
                            block_id)(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name='bn%d_conv_project' %
                                        block_id)(x)

            if in_channels == pointwise_filters and stride == 1:
                return Add(name='res_connect_' + str(block_id))([inputs, x])
            return x

        def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
            in_channels = inputs._keras_shape[-1]
            prefix = 'features.' + str(block_id) + '.conv.'
            pointwise_conv_filters = int(filters * alpha)
            pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
            # Expand
            x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                       use_bias=False, activation=None,
                       name='mobl%d_conv_expand' % block_id)(inputs)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name='bn%d_conv_bn_expand' %
                                        block_id)(x)
            x = Activation(relu6, name='conv_%d_relu' % block_id)(x)

            # Depthwise
            x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                                use_bias=False, padding='same',
                                name='mobl%d_conv_depthwise' % block_id)(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name='bn%d_conv_depthwise' % block_id)(x)

            x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

            # Project
            x = Conv2D(pointwise_filters,
                       kernel_size=1, padding='same', use_bias=False, activation=None,
                       name='mobl%d_conv_project' % block_id)(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name='bn%d_conv_bn_project' % block_id)(x)

            if in_channels == pointwise_filters and stride == 1:
                return Add(name='res_connect_' + str(block_id))([inputs, x])
            return x

        def mobilenetv2(input_shape,
                        classes,
                        alpha=1.0,
                        input_tensor=None
                        ):

            '''cols and rows in [96, 128, 160, 192, 224],
            If imagenet weights are being loaded, depth multiplier must be 1
            alpha can be one of 0.25, 0.50, 0.75 or 1.0 only.
            '''
            if input_tensor is not None:
                img_input = Input(tensor=input_tensor)
            else:
                img_input = Input(input_shape)
            rows=input_shape[1]
            first_block_filters = _make_divisible(32 * alpha, 8)
            x = Conv2D(first_block_filters,
                       kernel_size=3,
                       strides=(2, 2), padding='same',
                       use_bias=False, name='Conv1')(img_input)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
            x = Activation(relu6, name='Conv1_relu')(x)

            x = _first_inverted_res_block(x,
                                          filters=16,
                                          alpha=alpha,
                                          stride=1,
                                          block_id=0)

            x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                    expansion=6, block_id=1)
            x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                    expansion=6, block_id=2)

            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                    expansion=6, block_id=3)
            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                    expansion=6, block_id=4)
            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                    expansion=6, block_id=5)

            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                                    expansion=6, block_id=6)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                    expansion=6, block_id=7)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                    expansion=6, block_id=8)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                    expansion=6, block_id=9)

            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                    expansion=6, block_id=10)
            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                    expansion=6, block_id=11)
            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                    expansion=6, block_id=12)

            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                                    expansion=6, block_id=13)
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                    expansion=6, block_id=14)
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                    expansion=6, block_id=15)

            x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                                    expansion=6, block_id=16)
            if alpha > 1.0:
                last_block_filters = _make_divisible(1280 * alpha, 8)
            else:
                last_block_filters = 1280

            x = Conv2D(last_block_filters,
                       kernel_size=1,
                       use_bias=False,
                       name='Conv_1')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
            x = Activation(relu6, name='out_relu')(x)

            x = GlobalAveragePooling2D()(x)
            x = Dense(classes, activation='softmax',
                      use_bias=True, name='Logits')(x)
            model = Model(img_input, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))
            return model
        return mobilenetv2(imagedims, nb_classes)