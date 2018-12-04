# imports
import numpy as np

import keras
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Concatenate, Lambda, Dense, Add, Input, BatchNormalization, Flatten

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
        self.img_channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        self.num_f_g = 32
        self.num_f_d = 64
        self.f_size = 3
        self.num_rrdbs = 16

        self.generator = self.make_generator()
        # self.generator.summary()

        self.discriminator = self.make_discriminator()
        self.discriminator.summary()

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
        outp = conv_lrelu(outp, filters=self.img_channels, f_size=self.f_size)
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
