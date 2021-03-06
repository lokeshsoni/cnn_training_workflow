"""The following table describes the size and accuracy of different light-weight networks on size 224 x 224 for ImageNet dataset:
-----------------------------------------------------------------------------
Network                  |   Top 1 acc   |  Multiply-Adds (M) |  Params (M) |
-----------------------------------------------------------------------------
|   MobileNetV1          |    70.6 %     |        575         |     4.2     |
|   ShuffleNet (1.5)     |    69.0 %     |        292         |     2.9     |
|   ShuffleNet (x2)      |    70.9 %     |        524         |     4.4     |
|   NasNet-A             |    74.0 %     |        564         |     5.3     |
|   MobileNetV2          |    71.7 %     |        300         |     3.4     |
-----------------------------------------------------------------------------
# Reference
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation]
(https://arxiv.org/pdf/1801.04381.pdf))
"""
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2D, DepthwiseConv2D
from keras.layers import add
from keras.layers import Dense
from keras.regularizers import l2
from keras import backend as K


class MobileNetV2():
    @staticmethod
    def build(imagedims, classes):
        def relu6(x):
            return K.relu(x, max_value=6)

        def Relu6(x, **kwargs):
            return Activation(relu6, **kwargs)(x)

        def conv_block(inputs, filters, weight_decay, name, kernel=(3, 3), strides=(1, 1)):
            '''
            Normal convolution block performs conv+bn+relu6 operations.
            :param inputs: Input Keras tensor in (B, H, W, C_in)
            :param filters: number of filters in the convolution layer
            :param name: name for the convolutional layer
            :param kernel: kernel size
            :param strides: strides for convolution
            :return: Output tensor in (B, H_new, W_new, filters)
            '''
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
            x = Conv2D(filters, kernel,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=l2(weight_decay),
                       kernel_initializer='he_normal',
                       strides=strides,
                       name=name)(inputs)
            x = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name=name+'_bn')(x)
            return Relu6(x, name=name+'_relu')

        def InvertedResidualBlock(x, expand, out_channels, repeats, stride, weight_decay, block_id):
            '''
            This function defines a sequence of 1 or more identical layers, referring to Table 2 in the original paper.
            :param x: Input Keras tensor in (B, H, W, C_in)
            :param expand: expansion factor in bottlenect residual block
            :param out_channels: number of channels in the output tensor
            :param repeats: number of times to repeat the inverted residual blocks including the one that changes the dimensions.
            :param stride: stride for the 1x1 convolution
            :param weight_decay: hyperparameter for the l2 penalty
            :param block_id: as its name tells
            :return: Output tensor (B, H_new, W_new, out_channels)
            '''
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
            in_channels = K.int_shape(x)[channel_axis]
            x = Conv2D(expand * in_channels, 1, padding='same', strides=stride, use_bias=False,
                       kernel_regularizer=l2(weight_decay),
                       kernel_initializer='he_normal',
                       name='conv_%d_0' % block_id)(x)
            x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_%d_0_bn' % block_id)(x)
            x = Relu6(x, name='conv_%d_0_act_1' % block_id)
            x = DepthwiseConv2D((3, 3),
                                padding='same',
                                depth_multiplier=1,
                                strides=1,
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay),
                                kernel_initializer='he_normal',
                                name='conv_dw_%d_0' % block_id )(x)
            x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_dw_%d_0_bn' % block_id)(x)
            x = Relu6(x, name='conv_%d_0_act_2' % block_id)
            x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
                       kernel_regularizer=l2(weight_decay),
                       kernel_initializer='he_normal',
                       name='conv_bottleneck_%d_0' % block_id)(x)
            x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_0_bn' % block_id)(x)

            for i in range(1, repeats):
                x1 = Conv2D(expand*out_channels, 1, padding='same', strides=1, use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            kernel_initializer='he_normal',
                            name='conv_%d_%d' % (block_id, i))(x)
                x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name='conv_%d_%d_bn' % (block_id, i))(x1)
                x1 = Relu6(x1, name='conv_%d_%d_act_1' % (block_id, i))
                x1 = DepthwiseConv2D((3, 3),
                                     padding='same',
                                     depth_multiplier=1,
                                     strides=1,
                                     use_bias=False,
                                     kernel_regularizer=l2(weight_decay),
                                     kernel_initializer='he_normal',
                                     name='conv_dw_%d_%d' % (block_id, i))(x1)
                x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9, name='conv_dw_%d_%d_bn' % (block_id, i))(x1)
                x1 = Relu6(x1, name='conv_dw_%d_%d_act_2' % (block_id, i))
                x1 = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            kernel_initializer='he_normal',
                            name='conv_bottleneck_%d_%d' % (block_id, i))(x1)
                x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_%d_bn' % (block_id, i))(x1)
                x = add([x, x1], name='block_%d_%d_output' % (block_id, i))
            return x

        def mobilenet_v2(input_shape, classes, weight_decay=0.0005, feat_dropout=0.5, input_tensor=None):
            '''
            The function defines the MobileNet_V2 structure according to the Input column of Table 2 in the original paper.
            :param input_shape: size of the input tensor
            :param classes: number of classes in the data
            :param weight_decay: hyperparameter for the l2 penalty
            :param feat_dropout: dropout level applied to the output of the last hidden layer
            :param input_tensor: Optional input tensor if exists.
            :return: Keras model defined for classification
            '''
            if input_tensor is not None:
                img_input = Input(tensor=input_tensor)
            else:
                img_input = Input(input_shape)

            x = conv_block(img_input, 32, weight_decay=weight_decay, name='conv1', strides=(2, 2))
            x = InvertedResidualBlock(x, expand=1, out_channels=16, repeats=1, stride=1, weight_decay=weight_decay, block_id=1)
            x = InvertedResidualBlock(x, expand=6, out_channels=24, repeats=2, stride=2, weight_decay=weight_decay, block_id=2)
            x = InvertedResidualBlock(x, expand=6, out_channels=32, repeats=3, stride=2, weight_decay=weight_decay, block_id=3)
            x = InvertedResidualBlock(x, expand=6, out_channels=64, repeats=4, stride=2, weight_decay=weight_decay, block_id=4)
            x = InvertedResidualBlock(x, expand=6, out_channels=96, repeats=3, stride=1, weight_decay=weight_decay, block_id=5)
            x = InvertedResidualBlock(x, expand=6, out_channels=160, repeats=3, stride=2, weight_decay=weight_decay, block_id=6)
            x = InvertedResidualBlock(x, expand=6, out_channels=320, repeats=1, stride=1, weight_decay=weight_decay, block_id=7)
            x = conv_block(x, 1280, weight_decay=weight_decay, name='conv2', kernel=(1, 1), strides=1)
            x = GlobalAveragePooling2D()(x)
            if feat_dropout!=0.:
                x = Dropout(feat_dropout, name='dropout')(x)
            x = Dense(classes, kernel_regularizer=l2(weight_decay), name='fc_pred')(x)
            x = Activation('softmax', name='act_softmax')(x)

            return Model(inputs=img_input, outputs=x)
        return mobilenet_v2(imagedims, classes)

