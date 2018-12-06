# imports
import numpy as np

import keras
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Concatenate, Lambda, Dense, Add, Input, BatchNormalization, Flatten, \
    MaxPooling2D

from keras.optimizers import Adam

# own packages:
import data_io
import augmentation

# TRAINING VARIABLES
EPOCHS = 100
BATCH_SIZE = 50
IMAGES_PER_SPLIT = 2
SAMPLE_INTERVALL = 100
GENERATOR_EVOLUTION_DATA = []
GENERATOR_EVOLUTION_INDIZES = [1, 100, 500, 1900]
GENERATED_DATA_LOCATION = 'generated_images/esrgan/'
DATASET_PATH = ''
MODEL_WEIGHTS_PATH = 'models/esrgan/'

# - - - - - - - - - -

class ESRGAN():

    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.img_channels_cond = 3
        self.img_channels_gen = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels_cond)

        self.num_f_g = 32
        self.num_f_d = 64
        self.f_size = 3
        self.num_rrdbs = 16

        self.generator = self.make_generator()
        # self.generator.summary()

        self.discriminator = self.make_discriminator()
        # self.discriminator.summary()

        self.vgg19 = self.make_vgg19()
        self.vgg19.summary()

        self.lr_g = 0.0001
        self.lr_d = 0.0001

        self.opt_g = Adam(self.lr_g)
        self.opt_d = Adam(self.lr_d)



    def make_generator(self):

        # no activation by default
        def conv_lrelu(inp, filters, f_size, use_act=False, alpha=0.0):
            outp = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(inp)
            if use_act:
                outp = LeakyReLU(alpha=alpha)(outp)
            return outp

        def dense_block(inp0, filters, f_size, alpha, residual_scaling_factor):
            # Densely Connected Convolutional Networks
            out1 = conv_lrelu(inp0, filters, f_size, use_act=True, alpha=alpha)
            inp = Concatenate(axis=-1)([inp0, out1])
            out2 = conv_lrelu(inp, filters, f_size, use_act=True, alpha=alpha)
            inp = Concatenate(axis=-1)([inp0, out1, out2])
            out3 = conv_lrelu(inp, filters, f_size, use_act=True, alpha=alpha)
            inp = Concatenate(axis=-1)([inp0, out1, out2, out3])
            out4 = conv_lrelu(inp, filters, f_size, use_act=True, alpha=alpha)
            inp = Concatenate(axis=-1)([inp0, out1, out2, out3, out4])
            out5 = conv_lrelu(inp, filters, f_size)

            out = Lambda(lambda x: x * residual_scaling_factor)(out5)
            return Add()([inp0, out])

        def rrd_block(inp, filters, f_size, alpha, residual_scaling_factor):
            outp = dense_block(inp, filters, f_size, alpha, residual_scaling_factor)
            outp = dense_block(outp, filters, f_size, alpha, residual_scaling_factor)
            outp = dense_block(outp, filters, f_size, alpha, residual_scaling_factor)
            outp = Lambda(lambda x: x * residual_scaling_factor)(outp)
            return Add()([inp, outp])

        inp = Input(shape=self.img_shape)
        # first convolution (no activation function):
        outp = conv_lrelu(inp, filters=self.num_f_g, f_size=self.f_size)
        # save output of first convolution to add it to the output of the RRDB's:
        out1 = outp
        # RRDB's:
        for _ in range(self.num_rrdbs):
            outp = rrd_block(outp, filters=self.num_f_g, f_size=self.f_size, alpha=0.2, residual_scaling_factor=0.2)
        # convolution after RRDB's (no activation function):
        outp = conv_lrelu(outp, filters=self.num_f_g, f_size=self.f_size)
        # Add output of convolution before and after RRDB's:
        outp = Add()([out1, outp])
        # here upsampling would normally take place
        # ...
        # two convolutions at the end; the first with activation, the second without (why no tanh?)
        outp = conv_lrelu(outp, filters=self.num_f_g, f_size=self.f_size, use_act=True, alpha=0.2)
        outp = conv_lrelu(outp, filters=self.img_channels_cond, f_size=self.f_size)
        return Model(inp, outp)

    def make_discriminator(self):
        def conv_block(inp, filters, f_size, strides, alpha, use_bn=True):
            outp = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(inp)
            if use_bn:
                outp = BatchNormalization()(outp)
            outp = LeakyReLU(alpha)(outp)
            return outp

        inp = Input(shape=self.img_shape)
        outp = conv_block(inp, self.num_f_d, self.f_size, strides=1, alpha=0.2, use_bn=False)
        outp = conv_block(outp, self.num_f_d, self.f_size, strides=2, alpha=0.2)
        outp = conv_block(outp, self.num_f_d * 2, self.f_size, strides=1, alpha=0.2)
        outp = conv_block(outp, self.num_f_d * 2, self.f_size, strides=2, alpha=0.2)
        outp = conv_block(outp, self.num_f_d * 4, self.f_size, strides=1, alpha=0.2)
        outp = conv_block(outp, self.num_f_d * 4, self.f_size, strides=2, alpha=0.2)
        outp = conv_block(outp, self.num_f_d * 8, self.f_size, strides=1, alpha=0.2)
        outp = conv_block(outp, self.num_f_d * 8, self.f_size, strides=2, alpha=0.2)
        outp = Flatten()(outp)
        outp = Dense(1024)(outp)
        outp = LeakyReLU(0.2)(outp)
        outp = Dense(1, activation='sigmoid')(outp)
        return Model(inp, outp)

    def make_vgg19(self):
        inp = Input(self.img_shape)
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inp)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # change the definition here to access the values before the activation function is applied (compare ESRGAN)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv4')(x)
        # (LeakyReLu with alpha=0.0 is ReLU)
        x = LeakyReLU(alpha=0.0)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        # Load the weights:
        vgg = Model(inp, x)
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = keras.applications.vgg19.get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models')
        vgg.load_weights(weights_path)
        # return only the required part of the Model:
        return Model(inp, vgg.layers[20].output)

    def train(self):
        pass

    def save_generator(self, name):
        self.generator.save_weights(MODEL_WEIGHTS_PATH + 'generator_weights_' + str(name) + '.hdf5')

    def load_generator(self, name):
        self.generator.load_weights(MODEL_WEIGHTS_PATH + 'generator_weights_' + str(name) + '.hdf5')

    def apply_generator(self, tensor):
        # expect input to be of shape (num_samples, height, width, channels)
        tensor = np.array(tensor / 127.5 - 1, dtype=np.float32)
        img_out = self.generator.predict(tensor)
        return img_out


if __name__ == '__main__':
    esrgan = ESRGAN()
    # esrgan.train()
