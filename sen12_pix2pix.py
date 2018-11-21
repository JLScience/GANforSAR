import sys
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers import Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU

# own packages:
import data_io
import augmentation

# TRAINING VARIABLES:
EPOCHS = 100
BATCH_SIZE = 50
IMAGES_PER_SPLIT = 4
SAMPLE_INTERVAL = 100
GENERATOR_EVOLUTION_DATA = []
GENERATOR_EVOLUTION_INDIZES = [1, 100, 500, 1900]
GENERATED_DATA_LOCATION = 'generated_images/tests/'
DATASET_PATH = ''
MODEL_WEIGHTS_PATH = 'models/tests/64x64/'

# - - - - - - - - - -


class GAN_P2P():

    def __init__(self):
        # image geometry
        self.img_rows = 64
        self.img_cols = 64
        self.channels_cond = 3
        self.channels_gen = 1
        self.img_shape_cond = (self.img_rows, self.img_cols, self.channels_cond)
        self.img_shape_gen = (self.img_rows, self.img_cols, self.channels_gen)

        # base number of filters
        self.num_f_g = 64
        self.num_f_d = 32

        # discriminator output shape
        self.disc_patch = (int(self.img_rows / 16), int(self.img_cols / 16), 1)  # img_rows / (2**num_disc_layers)

        # self.opt_g = Adam(lr=0.0002, beta_1=0.5)  # pix2pix version
        self.opt_g = Adam(lr=0.0002, beta_1=0.5)
        self.opt_d = Adam(lr=0.00005, beta_1=0.5)

        self.generator = self.make_generator_64()
        print('--> Generator Model:')
        self.generator.summary()

        # PRE TRAINING
        if int(sys.argv[1]) < 0 and len(sys.argv) > 2:
            self.load_generator(sys.argv[2])
            print('Loaded pre-trained model ' + sys.argv[2])

        self.discriminator = self.make_discriminator()
        print('--> Discriminator Model:')
        self.discriminator.summary()

        # --- compile discriminator model:
        self.discriminator.compile(loss='mse', optimizer=self.opt_d, metrics=['accuracy'])

        # --- compile generator model:
        self.discriminator.trainable = False
        # define in- / output:
        img_gen_real = Input(shape=self.img_shape_gen)      # img_A
        img_cond = Input(shape=self.img_shape_cond)         # img_B
        img_gen = self.generator(img_cond)                  # fake_A
        validity = self.discriminator([img_gen, img_cond])
        # build and compile model:
        self.combined = Model(inputs=[img_gen_real, img_cond], outputs=[validity, img_gen])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=self.opt_g)
        print('--> Combined Generator Model:')
        self.combined.summary()

    def make_generator(self):
        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input(shape=self.img_shape_cond)

        d1 = conv2d(d0, self.num_f_g, bn=False)
        d2 = conv2d(d1, self.num_f_g * 2)
        d3 = conv2d(d2, self.num_f_g * 4)
        d4 = conv2d(d3, self.num_f_g * 8)
        d5 = conv2d(d4, self.num_f_g * 8)
        d6 = conv2d(d5, self.num_f_g * 8)
        d7 = conv2d(d6, self.num_f_g * 8)

        u1 = deconv2d(d7, d6, self.num_f_g * 8)
        u2 = deconv2d(u1, d5, self.num_f_g * 8)
        u3 = deconv2d(u2, d4, self.num_f_g * 8)
        u4 = deconv2d(u3, d3, self.num_f_g * 4)
        u5 = deconv2d(u4, d2, self.num_f_g * 4)
        u6 = deconv2d(u5, d1, self.num_f_g)

        u7 = UpSampling2D(size=2)(u6)
        output_image = Conv2D(self.channels_gen, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_image)

    def make_generator_64(self):
        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input(shape=self.img_shape_cond)
        d1 = conv2d(d0, self.num_f_g, bn=False)
        d2 = conv2d(d1, self.num_f_g * 2)
        d3 = conv2d(d2, self.num_f_g * 4)
        d4 = conv2d(d3, self.num_f_g * 8)
        d5 = conv2d(d4, self.num_f_g * 8)

        u1 = deconv2d(d5, d4, self.num_f_g * 8)
        u2 = deconv2d(u1, d3, self.num_f_g * 4)
        u3 = deconv2d(u2, d2, self.num_f_g * 2)
        u4 = deconv2d(u3, d1, self.num_f_g)
        u5 = UpSampling2D(size=2)(u4)

        output_image = Conv2D(self.channels_gen, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)

        return Model(d0, output_image)

    def make_discriminator(self):
        def discriminator_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_gen = Input(shape=self.img_shape_gen)
        img_cond = Input(shape=self.img_shape_cond)

        # concatenate by channels:
        combined_imags = Concatenate(axis=-1)([img_gen, img_cond])

        d1 = discriminator_layer(combined_imags, self.num_f_d, bn=False)
        d2 = discriminator_layer(d1, self.num_f_d * 2)
        d3 = discriminator_layer(d2, self.num_f_d * 4)
        d4 = discriminator_layer(d3, self.num_f_d * 8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_gen, img_cond], validity)

    def train_sen12(self):

        name_string = 'sets'
        if len(sys.argv) == 1:
            dataset_nr = [0]
            name_string = name_string + '_0'
        else:
            if sys.argv[1] == '-1':
                dataset_nr = -1
                name_string = name_string + '_all_same'
            elif sys.argv[1] == '-2':
                dataset_nr = -2
                name_string = name_string + '_all_separated'
            else:
                dataset_nr = []
                for i in range(1, len(sys.argv)):
                    dataset_nr.append(sys.argv[i])
                    name_string = name_string + '_' + str(sys.argv[i])
            if int(sys.argv[1]) < 0 and len(sys.argv) > 2:
                name_string = name_string + '_pretrained'

        os.mkdir(GENERATED_DATA_LOCATION + name_string)

        # load datasets:
        if dataset_nr < 0:
            if dataset_nr == -1:
                dataset_opt_train, dataset_sar_train, dataset_opt_test, dataset_sar_test = data_io.load_Sen12_data(
                    portion_mode=1.0, split_mode='same', split_ratio=0.8)
            else:
                dataset_opt_train, dataset_sar_train, dataset_opt_test, dataset_sar_test = data_io.load_Sen12_data(
                    portion_mode=1.0, split_mode='separated', split_ratio=0.8)
        else:
            print('--- Load dataset number(s) {} ...'.format(dataset_nr))
            dataset_opt_train, dataset_sar_train, dataset_opt_test, dataset_sar_test = data_io.load_Sen12_data(
                portion_mode=dataset_nr, split_mode='same', split_ratio=0.7)

        # cut images (from 256x256 to 64x64):
        print('--- divide images ...')
        dataset_sar_test = augmentation.split_images(dataset_sar_test, factor=4, num_images_per_split=IMAGES_PER_SPLIT)
        print('sar_test done')
        dataset_opt_test = augmentation.split_images(dataset_opt_test, factor=4, num_images_per_split=IMAGES_PER_SPLIT)
        print('opt_test done')
        dataset_sar_train = augmentation.split_images(dataset_sar_train, factor=4, num_images_per_split=IMAGES_PER_SPLIT)
        print('sar_train done')
        dataset_opt_train = augmentation.split_images(dataset_opt_train, factor=4, num_images_per_split=IMAGES_PER_SPLIT)
        print('opt_train done')

        # normalize datasets:
        print('--- normalize datasets ...')
        dataset_sar_test = np.array(dataset_sar_test / 127.5 - 1, dtype=np.float32)
        print('sar_test done')
        dataset_opt_test = np.array(dataset_opt_test / 127.5 - 1, dtype=np.float32)
        print('opt_test done')
        dataset_sar_train = np.array(dataset_sar_train / 127.5 - 1, dtype=np.float32)
        print('sar_train done')
        dataset_opt_train = np.array(dataset_opt_train / 127.5 - 1, dtype=np.float32)
        print('opt_train done')

        # # filter sar images:
        # print('--- filter sar datasets ...')
        # dataset_sar_test = augmentation.lee_filter_dataset(dataset_sar_test, window_size=3)
        # print('sar_test done')
        # dataset_sar_train = augmentation.lee_filter_dataset(dataset_sar_train, window_size=3)
        # print('sar_train done')

        num_train = dataset_opt_train.shape[0]
        print('number of training samples: {}'.format(num_train))
        num_test = dataset_opt_test.shape[0]
        print('number of test samples: {}'.format(num_test))

        # ground truths:
        valid = np.ones((BATCH_SIZE,) + self.disc_patch)
        fake = np.zeros((BATCH_SIZE,) + self.disc_patch)

        rep = 0
        for epoch in range(EPOCHS):

            # shuffle datasets:
            p = np.random.permutation(num_train)
            dataset_opt_train = dataset_opt_train[p]
            dataset_sar_train = dataset_sar_train[p]

            for batch_i in range(0, num_train, BATCH_SIZE):
                # get actual batch:
                imgs_gen_real = dataset_sar_train[batch_i:batch_i + BATCH_SIZE]
                imgs_cond = dataset_opt_train[batch_i:batch_i + BATCH_SIZE]
                num_samples = imgs_gen_real.shape[0]

                # train discriminator
                imgs_gen = self.generator.predict(imgs_cond)
                d_loss_real = self.discriminator.train_on_batch(x=[imgs_gen_real, imgs_cond], y=valid[:num_samples])
                d_loss_fake = self.discriminator.train_on_batch(x=[imgs_gen, imgs_cond], y=fake[:num_samples])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train generator:
                g_loss = self.combined.train_on_batch(x=[imgs_gen_real, imgs_cond], y=[valid[:num_samples], imgs_gen_real])

                if rep % SAMPLE_INTERVAL == 0:
                    print("[Epoch {:5d}/{:5d}, Batch {:4d}/{:4d}] \t "
                          "[D loss: {:05.3f}, acc: {:05.2f}%] \t [G loss: {:05.3f}]".format(epoch + 1, EPOCHS,
                                                                                            int(batch_i / BATCH_SIZE),
                                                                                            int(num_train / BATCH_SIZE),
                                                                                            d_loss[0], 100 * d_loss[1],
                                                                                            g_loss[0]))

                    i = np.random.randint(low=0, high=num_test, size=3)
                    img_batch = dataset_sar_test[i], dataset_opt_test[i]
                    self.sample_images(epoch, rep, img_batch, name_string)
                    img_batch = dataset_sar_test[GENERATOR_EVOLUTION_INDIZES], dataset_opt_test[GENERATOR_EVOLUTION_INDIZES]
                    self.generator_evolution(epoch, SAMPLE_INTERVAL, rep, img_batch, name_string)
                rep += 1
        self.save_generator(name_string)

    def train_aerial_map(self):
        # load datasets:
        print('--- Load datasets ...')
        aerial_train, map_train, aerial_test, map_test = data_io.load_dataset_maps(DATASET_PATH)

        # # shrink dataset:
        # aerial_train = aerial_train[:500, ...]
        # map_train = map_train[:500, ...]
        # aerial_test = aerial_test[:50, ...]
        # map_test = map_test[:50, ...]

        # normalize datasets:
        print('--- normalize datasets ...')
        aerial_test = np.array(aerial_test / 127.5 - 1, dtype=np.float32)
        print('aerial_test done')
        map_test = np.array(map_test / 127.5 - 1, dtype=np.float32)
        print('map_test done')
        aerial_train = np.array(aerial_train / 127.5 - 1, dtype=np.float32)
        print('aerial_train done')
        map_train = np.array(map_train / 127.5 - 1, dtype=np.float32)
        print('map_train done')

        # # cut images:
        # print('--- divide images ...')
        # dataset_sar_test = augmentation.split_images(dataset_sar_test, 2)
        # print('sar_test done')
        # dataset_opt_test = augmentation.split_images(dataset_opt_test, 2)
        # print('opt_test done')
        # dataset_sar_train = augmentation.split_images(dataset_sar_train, 2)
        # print('sar_train done')
        # dataset_opt_train = augmentation.split_images(dataset_opt_train, 2)
        # print('opt_train done')

        num_train = aerial_train.shape[0]
        num_test = aerial_test.shape[0]

        # ground truths:
        valid = np.ones((BATCH_SIZE,) + self.disc_patch)
        fake = np.zeros((BATCH_SIZE,) + self.disc_patch)

        rep = 0
        for epoch in range(EPOCHS):

            # shuffle datasets:
            p = np.random.permutation(num_train)
            map_train = map_train[p]
            aerial_train = aerial_train[p]

            for batch_i in range(0, num_train, BATCH_SIZE):
                # input = map, output = aerial
                # get actual batch:
                # imgs_gen_real = aerial_train[batch_i:batch_i + BATCH_SIZE]
                # imgs_cond = map_train[batch_i:batch_i + BATCH_SIZE]
                imgs_gen_real = map_train[batch_i:batch_i + BATCH_SIZE]
                imgs_cond = aerial_train[batch_i:batch_i + BATCH_SIZE]
                num_samples = imgs_gen_real.shape[0]

                # train discriminator
                imgs_gen = self.generator.predict(imgs_cond)
                d_loss_real = self.discriminator.train_on_batch(x=[imgs_gen_real, imgs_cond], y=valid[:num_samples])
                d_loss_fake = self.discriminator.train_on_batch(x=[imgs_gen, imgs_cond], y=fake[:num_samples])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train generator:
                g_loss = self.combined.train_on_batch(x=[imgs_gen_real, imgs_cond],
                                                      y=[valid[:num_samples], imgs_gen_real])

                print("[Epoch {:5d}/{:5d}, Batch {:4d}/{:4d}] \t "
                      "[D loss: {:05.3f}, acc: {:05.2f}%] \t [G loss: {:05.3f}]".format(epoch + 1, EPOCHS,
                                                                                        int(batch_i / BATCH_SIZE),
                                                                                        int(num_train / BATCH_SIZE),
                                                                                        d_loss[0], 100 * d_loss[1],
                                                                                        g_loss[0]))

                if rep % SAMPLE_INTERVAL == 0:
                    i = np.random.randint(low=0, high=num_test, size=3)
                    # img_batch = aerial_test[i], map_test[i]
                    img_batch = map_test[i], aerial_test[i]
                    self.sample_images(epoch, rep, img_batch)
                    # img_batch = aerial_test[GENERATOR_EVOLUTION_INDIZES], map_test[GENERATOR_EVOLUTION_INDIZES]
                    img_batch = map_test[GENERATOR_EVOLUTION_INDIZES], aerial_test[GENERATOR_EVOLUTION_INDIZES]
                    self.generator_evolution(epoch, SAMPLE_INTERVAL, rep, img_batch)
                rep += 1
        self.save_generator()

    def generator_evolution(self, epoch, sample_interval, repetition, img_batch, name_string):

        imgs_gen_real, imgs_cond = img_batch
        imgs_gen = self.generator.predict(imgs_cond)

        imgs_gen = 0.5 * imgs_gen + 0.5
        GENERATOR_EVOLUTION_DATA.append(imgs_gen)
        # self.generator_evolution_data.append(imgs_gen)

        num_images_to_show = 5
        if repetition == 0:
            return
        if repetition % (sample_interval * num_images_to_show) == 0:
            imgs_gen_real = 0.5 * imgs_gen_real + 0.5
            imgs_cond = 0.5 * imgs_cond + 0.5
            fig, axs = plt.subplots(4, 7, figsize=(14, 8))
            for i in range(4):
                # plot condition image
                axs[i, 0].imshow(imgs_cond[i])
                axs[i, 0].set_title('Condition')
                axs[i, 0].axis('off')
                # plot generated images:
                for j in range(1, num_images_to_show + 1):
                    idx = int(j * repetition / (sample_interval * num_images_to_show))
                    axs[i, j].imshow(GENERATOR_EVOLUTION_DATA[idx][i, :, :, 0], cmap='gray')
                    axs[i, j].set_title(idx * sample_interval)
                    axs[i, j].axis('off')
                # plot original image:
                axs[i, 6].imshow(imgs_gen_real[i, :, :, 0], cmap='gray')
                axs[i, 6].set_title('Original')
                axs[i, 6].axis('off')
            fig.savefig(GENERATED_DATA_LOCATION + name_string + '/' + 'evolution_{}_{}.png'.format(epoch+1, repetition))
            plt.close()

    def sample_images(self, epoch, repetition, img_batch, name_string):
        r, c = 3, 3

        imgs_gen_real, imgs_cond = img_batch
        imgs_gen = self.generator.predict([imgs_cond])

        imgs_all = [imgs_cond, imgs_gen_real, imgs_gen]

        # Rescale images 0 - 1
        for i in range(len(imgs_all)):
            imgs_all[i] = 0.5 * imgs_all[i] + 0.5

        titles = ['Condition', 'Original', 'Generated']

        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                # RGB image:
                if titles[j] == 'Condition':
                    axs[i, j].imshow(imgs_all[j][i])
                # gray scale image:
                else:
                    axs[i, j].imshow(imgs_all[j][i, :, :, 0], cmap='gray')
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')

        fig.savefig(GENERATED_DATA_LOCATION + name_string + '/' + '{}_{}.png'.format(epoch+1, repetition))
        plt.close()

    def save_generator(self, name):
        self.generator.save_weights(MODEL_WEIGHTS_PATH + 'generator_weights_' + str(name) + '.hdf5')

    def load_generator(self, name):
        self.generator.load_weights(MODEL_WEIGHTS_PATH + 'generator_weights_' + str(name) + '.hdf5')

    def apply_generator(self, tensor):
        # expect input to be of shape (num_samples, height, width, channels)
        tensor = np.array(tensor / 127.5 - 1, dtype=np.float32)
        img_out = self.generator.predict(tensor)
        return img_out


def translate_eurosat(model_name):
    gan = GAN_P2P()
    # gan.train_sen12()
    gan.load_generator(model_name)
    data = data_io.load_dataset_eurosat()
    f = h5py.File('data/EuroSAT/dataset_translated.hdf5')
    names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
             'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    for i, d in enumerate(data):
        print('Translate set {}: {} ...'.format(i, names[i]))
        d_t = gan.apply_generator(d)
        d_t = (d_t + 1) * 127.5
        d_t = np.array(np.round(d_t), dtype=np.uint8)
        f.create_dataset(name=names[i], data=d_t)


def test_generator(num_images):
    gan = GAN_P2P()
    _, _, _, maps = data_io.load_dataset_maps(DATASET_PATH)
    gan.load_generator()
    p = np.random.permutation(maps.shape[0])
    maps = maps[p]
    aerials = gan.apply_generator(maps[:num_images, ...])

    fig, axs = plt.subplots(2, num_images)
    for i in range(num_images):
        axs[0, i].imshow(maps[i, ...])
        aerials[i, ...] = 0.5 * aerials[i, ...] + 0.5
        axs[1, i].imshow(aerials[i, ...])
    plt.show()



if __name__ == '__main__':
    gan = GAN_P2P()
    gan.train_sen12()
    # translate_eurosat('sets_18_19_20_24_39_49_53')
