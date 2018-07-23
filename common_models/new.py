from keras.layers import Input, concatenate, Dropout, Dense, Flatten, Activation
from keras.layers import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""


class InceptionV4():
    @staticmethod
    def build(imagedims, classes):
        def conv_block(x, nb_filter, nb_row, nb_col, padding='same', strides=(1, 1), bias=False):
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1

            x = Convolution2D(nb_filter, (nb_row, nb_col), strides=strides, padding=padding, use_bias=bias)(x)
            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation('relu')(x)
            return x

        def inception_stem(input):
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1

            # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
            x = conv_block(input, 32, 3, 3, strides=(2, 2), padding='valid')
            x = conv_block(x, 32, 3, 3, padding='valid')
            x = conv_block(x, 64, 3, 3)

            x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
            x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

            x = concatenate([x1, x2], axis=channel_axis)

            x1 = conv_block(x, 64, 1, 1)
            x1 = conv_block(x1, 96, 3, 3, padding='valid')

            x2 = conv_block(x, 64, 1, 1)
            x2 = conv_block(x2, 64, 1, 7)
            x2 = conv_block(x2, 64, 7, 1)
            x2 = conv_block(x2, 96, 3, 3, padding='valid')

            x = concatenate([x1, x2], axis=channel_axis)

            x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
            x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

            x = concatenate([x1, x2], axis=channel_axis)
            return x

        def inception_A(input):
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1

            a1 = conv_block(input, 96, 1, 1)

            a2 = conv_block(input, 64, 1, 1)
            a2 = conv_block(a2, 96, 3, 3)

            a3 = conv_block(input, 64, 1, 1)
            a3 = conv_block(a3, 96, 3, 3)
            a3 = conv_block(a3, 96, 3, 3)

            a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
            a4 = conv_block(a4, 96, 1, 1)

            m = concatenate([a1, a2, a3, a4], axis=channel_axis)
            return m

        def inception_B(input):
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1

            b1 = conv_block(input, 384, 1, 1)

            b2 = conv_block(input, 192, 1, 1)
            b2 = conv_block(b2, 224, 1, 7)
            b2 = conv_block(b2, 256, 7, 1)

            b3 = conv_block(input, 192, 1, 1)
            b3 = conv_block(b3, 192, 7, 1)
            b3 = conv_block(b3, 224, 1, 7)
            b3 = conv_block(b3, 224, 7, 1)
            b3 = conv_block(b3, 256, 1, 7)

            b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
            b4 = conv_block(b4, 128, 1, 1)

            m = concatenate([b1, b2, b3, b4], axis=channel_axis)
            return m

        def inception_C(input):
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1

            c1 = conv_block(input, 256, 1, 1)

            c2 = conv_block(input, 384, 1, 1)
            c2_1 = conv_block(c2, 256, 1, 3)
            c2_2 = conv_block(c2, 256, 3, 1)
            c2 = concatenate([c2_1, c2_2], axis=channel_axis)

            c3 = conv_block(input, 384, 1, 1)
            c3 = conv_block(c3, 448, 3, 1)
            c3 = conv_block(c3, 512, 1, 3)
            c3_1 = conv_block(c3, 256, 1, 3)
            c3_2 = conv_block(c3, 256, 3, 1)
            c3 = concatenate([c3_1, c3_2], axis=channel_axis)

            c4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
            c4 = conv_block(c4, 256, 1, 1)

            m = concatenate([c1, c2, c3, c4], axis=channel_axis)
            return m

        def reduction_A(input):
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1

            r1 = conv_block(input, 384, 3, 3, strides=(2, 2), padding='valid')

            r2 = conv_block(input, 192, 1, 1)
            r2 = conv_block(r2, 224, 3, 3)
            r2 = conv_block(r2, 256, 3, 3, strides=(2, 2), padding='valid')

            r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

            m = concatenate([r1, r2, r3], axis=channel_axis)
            return m

        def reduction_B(input):
            if K.image_data_format() == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1

            r1 = conv_block(input, 192, 1, 1)
            r1 = conv_block(r1, 192, 3, 3, strides=(2, 2), padding='valid')

            r2 = conv_block(input, 256, 1, 1)
            r2 = conv_block(r2, 256, 1, 7)
            r2 = conv_block(r2, 320, 7, 1)
            r2 = conv_block(r2, 320, 3, 3, strides=(2, 2), padding='valid')

            r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

            m = concatenate([r1, r2, r3], axis=channel_axis)
            return m

        def create_inception_v4(imagedims, classes):
            '''
            Creates a inception v4 network

            :param nb_classes: number of classes.txt
            :return: Keras Model with 1 input and 1 output
            '''
            init = Input(imagedims)
            # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
            x = inception_stem(init)

            # 4 x Inception A
            for i in range(4):
                x = inception_A(x)

            # Reduction A
            x = reduction_A(x)

            # 7 x Inception B
            for i in range(7):
                x = inception_B(x)

            # Reduction B
            x = reduction_B(x)

            # 3 x Inception C
            for i in range(3):
                x = inception_C(x)

            # Average Pooling
            x = AveragePooling2D((8, 8))(x)

            # Dropout
            x = Dropout(0.8)(x)
            x = Flatten(name="logits")(x)

            # Output
            out = Dense(units=classes, activation='softmax')(x)
            return Model(init, out, name='Inception-v4')
        return create_inception_v4(imagedims, classes)


model=InceptionV4.build((299,299,3),128)
plot_model(model,'inception_v4.png',show_layer_names=True,show_shapes=True)