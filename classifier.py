#
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import Adam

import data_io
import augmentation
import my_resnet50

MODEL_WEIGHTS_PATH = 'models/classifier/'
EPOCHS = 50
BATCH_SIZE = 50


class Custom_Classifer():

    def __init__(self, network_type):
        self.network_type = network_type
        self.num_classes = 10
        self.class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
        self.class_names_ger = ['EinjKultur', 'Wald', 'KrautKultur', 'Straße', 'Industrie',
                                'Wiese', 'DauerKultur', 'Wohngebiet', 'Fluss', 'Gewässer']
        self.input_shape = (64, 64, 3)
        if network_type == 'vgg19':
            self.classifier = self.build_model_vgg16()
        elif network_type == 'resnet50':
            self.classifier = self.build_model_resnet50()
        else:
            raise ValueError('Unknown network type, you passed' + self.network_type)
        self.classifier.summary()

    def inspect_sar_data(self, class_idx, dataset_names):
        data_opt = data_io.load_dataset_eurosat('data/EuroSAT/dataset.hdf5', mode=self.class_names[class_idx])
        data_sar = []
        for name in dataset_names:
            data_sar.append(data_io.load_dataset_eurosat('data/EuroSAT/dataset_translated_' + name + '.hdf5', mode=self.class_names[class_idx]))
        num_image_pairs = 5
        for idx in range(5):
            fig, axs = plt.subplots(num_image_pairs, 1+len(data_sar), figsize=(3+2*len(data_sar), 2*num_image_pairs-2))
            for i in range(num_image_pairs):
                axs[i, 0].imshow(data_opt[idx*num_image_pairs+i, ...])
                axs[i, 0].axis('off')
                for j in range(len(data_sar)):
                    axs[i, j+1].imshow(data_sar[j][idx*num_image_pairs+i, :, :, 0], cmap='gray')
                    axs[i, j+1].axis('off')
            plt.show()

    def build_model_vgg16(self):
        vgg16_model = keras.applications.vgg19.VGG19(include_top=False, input_shape=(64, 64, 3))
        vgg16_model.trainable = False
        inp = Input(shape=(64, 64, 3))
        outp = vgg16_model(inp)
        outp = Flatten()(outp)
        outp = Dropout(0.3)(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dropout(0.3)(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dense(self.num_classes, activation='softmax')(outp)
        return Model(inp, outp)

    def build_model_resnet50(self):
        # resnet50_model = my_resnet50.ResNet50(include_top=False, input_shape=(64, 64, 3))
        # resnet50_model.trainable = False
        resnet50_model = my_resnet50.ResNet50(include_top=False, weights=None, input_shape=(64, 64, 3))
        inp = Input(shape=(64, 64, 3))
        outp = resnet50_model(inp)
        outp = Flatten()(outp)
        outp = Dropout(0.5)(outp)
        outp = Dense(512, activation='relu')(outp)
        # outp = Dropout(0.3)(outp)
        # outp = Dense(512, activation='relu')(outp)
        outp = Dense(self.num_classes, activation='softmax')(outp)
        return Model(inp, outp)

    def train_optical(self):
        opt = Adam()
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # load dataset
        x_train, y_train, x_val, y_val, x_test, y_test = data_io.divide_dataset_eurosat(0.1, 0.1)

        # augment data:
        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p]
        y_train = y_train[p]
        x_train = augmentation.apply_all(x_train)

        # pre-process data:

        # convert RGB --> BGR:
        x_train = x_train[..., ::-1]
        x_val = x_val[..., ::-1]
        x_test = x_test[..., ::-1]
        # subtract VGG19 mean:
        vgg19_means = [103.939, 116.779, 123.68]
        for i in range(x_train.shape[-1]):
            x_train[:, :, :, i] = np.array(x_train[:, :, :, i] - vgg19_means[i], dtype=np.float32)
            x_val[:, :, :, i] = np.array(x_val[:, :, :, i] - vgg19_means[i], dtype=np.float32)
            x_test[:, :, :, i] = np.array(x_test[:, :, :, i] - vgg19_means[i], dtype=np.float32)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # # convert to gray-scale:
        # x_train = augmentation.to_gray(x_train, mode=3)
        # x_val = augmentation.to_gray(x_val, mode=3)
        # x_test = augmentation.to_gray(x_test, mode=3)

        # train classifier
        self.classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val), verbose=2)
        loss, acc = self.classifier.evaluate(x_test, keras.utils.to_categorical(y_test, self.num_classes))
        print(loss, acc)

        # save classifier:
        self.save_trained_model(self.network_type + '_opt')

        conf = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            x = x_test[y_test == i]
            y_pred = np.argmax(self.classifier.predict(x), axis=1)
            for j in range(self.num_classes):
                conf[j, i] = y_pred[y_pred == j].shape[0] / y_pred.shape[0]
        print(np.round(conf, 2))
        plt.imshow(conf)
        plt.colorbar()
        plt.xlabel('correct class')
        plt.ylabel('predicted class')
        plt.xticks(np.arange(10), self.class_names, rotation=90)
        plt.yticks(np.arange(10), self.class_names)
        plt.show()

    def train_sar(self):
        opt = Adam()
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # load dataset
        # path = 'data/EuroSAT/dataset_translated_sen12_128x128_pre_balanced.hdf5'
        # x_train_esr, y_train_esr, x_val_esr, y_val_esr, x_test_esr, y_test_esr = data_io.divide_dataset_eurosat(0.1, 0.1, path=path)
        path = 'data/EuroSAT/dataset_translated_real_5.hdf5'
        # x_train_p2p, y_train_p2p, x_val_p2p, y_val_p2p, x_test_p2p, y_test_p2p = data_io.divide_dataset_eurosat(0.1, 0.1, path=path)

        x_train, y_train, x_val, y_val, x_test, y_test = data_io.divide_dataset_eurosat(0.1, 0.1, path=path)

        # x_train = np.concatenate((x_train_esr, x_train_p2p))
        # y_train = np.concatenate((y_train_esr, y_train_p2p))
        # x_val = np.concatenate((x_val_esr, x_val_p2p))
        # y_val = np.concatenate((y_val_esr, y_val_p2p))
        # x_test = np.concatenate((x_test_esr, x_test_p2p))
        # y_test = np.concatenate((y_test_esr, y_test_p2p))

        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p]
        y_train = y_train[p]

        # preprocess data
        x_train = np.array(x_train / 127.5 - 1, dtype=np.float32)
        x_val = np.array(x_val / 127.5 - 1, dtype=np.float32)
        x_test = np.array(x_test / 127.5 - 1, dtype=np.float32)

        # one channel input to three channel input:
        x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
        x_val = np.concatenate((x_val, x_val, x_val), axis=-1)
        x_test = np.concatenate((x_test, x_test, x_test), axis=-1)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # train classifier
        now = str(datetime.datetime.now())
        os.mkdir(MODEL_WEIGHTS_PATH + now + '/')
        checkpoint_path = MODEL_WEIGHTS_PATH + now + '/' + self.network_type + '_sar_' + \
                          'weights_E{epoch:02d}_ACC_{val_acc:.2f}.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor='val_acc',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True)
        self.classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(x_val, y_val), verbose=2, callbacks=[checkpoint])

        # self.load_trained_model('resnet50_sar_weights_E14_ACC_0.65')
        #
        # loss, acc = self.classifier.evaluate(x_test, keras.utils.to_categorical(y_test, self.num_classes))
        # print(loss, acc)
        #
        # # generate confusion matrix
        # conf = np.zeros((self.num_classes, self.num_classes))
        # for i in range(self.num_classes):
        #     x = x_test[y_test == i]
        #     y_pred = np.argmax(self.classifier.predict(x), axis=1)
        #     for j in range(self.num_classes):
        #         conf[j, i] = y_pred[y_pred == j].shape[0] / y_pred.shape[0]
        # print(np.round(conf, 2))
        # plt.imshow(conf)
        # plt.colorbar()
        # plt.xlabel('correct class')
        # plt.ylabel('predicted class')
        # plt.xticks(np.arange(10), self.class_names, rotation=90)
        # plt.yticks(np.arange(10), self.class_names)
        # plt.show()

    def train_sar_partial(self):
        opt = Adam()
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
        #          'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

        used_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Industrial', 'Pasture',
                        'PermanentCrop', 'Residential', 'SeaLake']

        paths = []
        paths.append('data/EuroSAT/dataset_translated_sen12_128x128_pre_balanced.hdf5')
        paths.append('data/EuroSAT/dataset_translated_real_5.hdf5')

        for path_idx, path in enumerate(paths):
            data = []
            for idx, used_class in enumerate(used_classes):
                data.append(data_io.load_dataset_eurosat(path=path, mode=used_class))

            if path_idx == 0:
                img_shape = (data[0].shape[1], data[0].shape[2], data[0].shape[3])
                x_train = np.zeros((0,) + img_shape, dtype=np.uint8)
                y_train = np.zeros(0, dtype=np.uint8)
                x_val = np.zeros((0,) + img_shape, dtype=np.uint8)
                y_val = np.zeros(0, dtype=np.uint8)
                x_test = np.zeros((0,) + img_shape, dtype=np.uint8)
                y_test = np.zeros(0, dtype=np.uint8)

            for i, d in enumerate(data):
                num_val = int(d.shape[0] * 0.1)
                num_test = int(d.shape[0] * 0.1)
                num_train = d.shape[0] - num_val - num_test
                x_train = np.append(x_train, d[0:num_train, ...], axis=0)
                y_train = np.append(y_train, i * np.ones(num_train), axis=0)
                x_val = np.append(x_val, d[num_train:num_train + num_val, ...], axis=0)
                y_val = np.append(y_val, i * np.ones(num_val), axis=0)
                x_test = np.append(x_test, d[num_train + num_val:, ...], axis=0)
                y_test = np.append(y_test, i * np.ones(num_test), axis=0)

        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p]
        y_train = y_train[p]

        # preprocess data
        x_train = np.array(x_train / 127.5 - 1, dtype=np.float32)
        x_val = np.array(x_val / 127.5 - 1, dtype=np.float32)
        x_test = np.array(x_test / 127.5 - 1, dtype=np.float32)

        # one channel input to three channel input:
        x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
        x_val = np.concatenate((x_val, x_val, x_val), axis=-1)
        x_test = np.concatenate((x_test, x_test, x_test), axis=-1)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)

        # # train classifier
        # now = str(datetime.datetime.now())
        # os.mkdir(MODEL_WEIGHTS_PATH + now + '/')
        # checkpoint_path = MODEL_WEIGHTS_PATH + now + '/' + self.network_type + '_sar_' + \
        #                   'weights_E{epoch:02d}_ACC_{val_acc:.2f}.hdf5'
        # checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                              monitor='val_acc',
        #                                              verbose=1,
        #                                              save_best_only=True,
        #                                              save_weights_only=True)
        # self.classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
        #                     validation_data=(x_val, y_val), verbose=2, callbacks=[checkpoint])

        self.load_trained_model('resnet50_sar_weights_E13_ACC_0.76')

        self.generate_confusion_matrix(x_test, y_test, num_classes=8, class_labels=used_classes)

    def generate_confusion_matrix(self, x_test, y_test, num_classes=0, class_labels='auto'):
        number_of_classes = self.num_classes if num_classes == 0 else num_classes
        labels_of_classes = self.class_names if class_labels == 'auto' else class_labels
        conf = np.zeros((number_of_classes, number_of_classes))
        for i in range(number_of_classes):
            x = x_test[y_test == i]
            y_pred = np.argmax(self.classifier.predict(x), axis=1)
            for j in range(number_of_classes):
                conf[j, i] = y_pred[y_pred == j].shape[0] / y_pred.shape[0]
        print(np.round(conf, 2))
        plt.imshow(conf)
        plt.colorbar()
        plt.xlabel('correct class')
        plt.ylabel('predicted class')
        plt.xticks(np.arange(number_of_classes), labels_of_classes, rotation=90)
        plt.yticks(np.arange(number_of_classes), labels_of_classes)
        plt.show()

    def save_trained_model(self, name):
        self.classifier.save_weights(MODEL_WEIGHTS_PATH + name + '.hdf5')

    def load_trained_model(self, name):
        self.classifier.load_weights(MODEL_WEIGHTS_PATH + name + '.hdf5')

    def apply(self, batch):
        return self.classifier.predict(batch)


if __name__ == '__main__':
    classifier = Custom_Classifer('resnet50')
    # classifier.load_trained_model('vgg_opt')
    # classifier.train_sar()
    classifier.train_sar_partial()
    # classifier.train_optical()
    # classifier.inspect_sar_data(5, ['real_5', 'sen12_128x128_pre_balanced'])
