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

MODEL_WEIGHTS_PATH = 'models/writing/classifier/'
EPOCHS = 20
BATCH_SIZE = 50


class Custom_Classifer():

    def __init__(self, network_type, pre_trained):
        self.network_type = network_type
        # self.data_type = data_type
        self.pre_trained = pre_trained
        self.num_classes = 10
        self.class_names = ['AnnualCrop', 'Forest', 'Herb.Veg.', 'Highway', 'Industrial',
                            'Pasture', 'Perm.Crop', 'Residential', 'River', 'SeaLake']
        self.class_names_ger = ['EinjKultur', 'Wald', 'KrautKultur', 'Straße', 'Industrie',
                                'Wiese', 'DauerKultur', 'Wohngebiet', 'Fluss', 'Gewässer']

        self.input_shape = (64, 64, 3)

        if network_type == 'vgg19':
            self.classifier = self.build_model_vgg19()
        elif network_type == 'resnet50':
            self.classifier = self.build_model_resnet50()
        else:
            raise ValueError('Unknown network type, you passed' + self.network_type)

        self.classifier.summary()

    def inspect_sar_data(self, class_idx, dataset_names):
        os.mkdir('generated_images/writing/inspect_sar_sata' + str(class_idx))
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
            # plt.imsave()
            fig.savefig('generated_images/writing/inspect_sar_sata' + str(class_idx) + '/' + str(idx))

    def build_model_vgg19(self):
        if self.pre_trained:
            vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)
            vgg19_model.trainable = False
        else:
            vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_shape=self.input_shape)
        inp = Input(shape=self.input_shape)
        outp = vgg19_model(inp)
        outp = Flatten()(outp)
        outp = Dropout(0.3)(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dropout(0.3)(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dense(self.num_classes, activation='softmax')(outp)
        return Model(inp, outp)

    def build_model_resnet50(self):
        if self.pre_trained:
            resnet50_model = my_resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
            resnet50_model.trainable = False
        else:
            resnet50_model = my_resnet50.ResNet50(include_top=False, weights=None, input_shape=self.input_shape)
        inp = Input(shape=self.input_shape)
        outp = resnet50_model(inp)
        outp = Flatten()(outp)
        # outp = Dropout(0.5)(outp)
        # outp = Dense(512, activation='relu')(outp)
        # outp = Dropout(0.3)(outp)
        # outp = Dense(512, activation='relu')(outp)
        outp = Dense(self.num_classes, activation='softmax')(outp)
        return Model(inp, outp)

    def train_optical(self):
        opt = Adam(lr=0.001)
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # load dataset
        x_train, y_train, _, _, x_test, y_test = data_io.divide_dataset_eurosat(0.0, 0.2)

        # augment data:
        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p]
        y_train = y_train[p]
        x_train = augmentation.apply_all(x_train, apply_transpose=False)

        # pre-process data:
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        # # convert RGB --> BGR:
        # x_train = x_train[..., ::-1]
        # x_test = x_test[..., ::-1]
        # # subtract VGG19 mean:
        # vgg19_means = [103.939, 116.779, 123.68]
        # for i in range(x_train.shape[-1]):
        #     x_train[:, :, :, i] = np.array(x_train[:, :, :, i] - vgg19_means[i], dtype=np.float32)
        #     x_test[:, :, :, i] = np.array(x_test[:, :, :, i] - vgg19_means[i], dtype=np.float32)
        # training labels to one-hot-encoding:
        y_train = keras.utils.to_categorical(y_train, self.num_classes)

        # train classifier
        self.classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(x_test, keras.utils.to_categorical(y_test, self.num_classes)), verbose=2)
        loss, acc = self.classifier.evaluate(x_test, keras.utils.to_categorical(y_test, self.num_classes))
        print(loss, acc)

        # save classifier:
        # self.save_trained_model(self.network_type + '_opt')

        conf = self.calculate_confusion_matrix(x_test, y_test)
        print(np.round(conf, 2))
        self.plot_confusion_matrix(conf)

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

    def evaluate(self, dataset, grayscale=False):

        name_addition = '_optical_' if dataset == 'optical' else '_sar_'

        optimizer = Adam(lr=0.001)

        # load dataset
        if dataset == 'optical':
            x_train, y_train, x_val, y_val, x_test, y_test = data_io.divide_dataset_eurosat(
                val_fraction=0.1, test_fraction=0.1)
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = data_io.divide_dataset_eurosat(
                val_fraction=0.1, test_fraction=0.1, path='data/EuroSAT/' + str(dataset))

        # augment data:
        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p]
        y_train = y_train[p]
        x_train = augmentation.apply_all(x_train, apply_transpose=False)

        # pre-process data:
        x_train = x_train / 127.5 - 1
        x_val = x_val / 127.5 - 1
        x_test = x_test / 127.5 - 1
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)

        if grayscale:
            x_train = np.expand_dims(np.mean(x_train, axis=-1), axis=-1)
            x_val = np.expand_dims(np.mean(x_val, axis=-1), axis=-1)
            x_test = np.expand_dims(np.mean(x_test, axis=-1), axis=-1)

        print('shaperino: ' + str(x_train.shape))

        if dataset != 'optical' or grayscale:
            # one channel input to three channel input:
            x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
            x_val = np.concatenate((x_val, x_val, x_val), axis=-1)
            x_test = np.concatenate((x_test, x_test, x_test), axis=-1)

        num_runs = 5
        losses = np.zeros(num_runs, dtype=np.float32)
        accuracys = np.zeros(num_runs, dtype=np.float32)
        confusion_matrizes = np.zeros((num_runs, self.num_classes, self.num_classes), dtype=np.float32)
        now = str(datetime.datetime.now())
        os.mkdir(MODEL_WEIGHTS_PATH + now + '/')

        for run_idx in range(num_runs):

            print(' - - - - RUN ' + str(run_idx + 1) + ' - - - - - ')

            # reload model:
            if self.network_type == 'vgg19':
                self.classifier = self.build_model_vgg19()
            elif self.network_type == 'resnet50':
                self.classifier = self.build_model_resnet50()
            self.classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            # shuffle data:
            p = np.random.permutation(x_train.shape[0])
            x_train = x_train[p]
            y_train = y_train[p]

            # train classifier
            checkpoint_path = MODEL_WEIGHTS_PATH + now + '/' + self.network_type + name_addition + \
                              'best_weights_run_' + str(run_idx+1) + '.hdf5'
            checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor='val_acc',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True)
            self.classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                validation_data=(x_val, y_val), verbose=2, callbacks=[checkpoint])

            self.classifier.load_weights(checkpoint_path)

            losses[run_idx], accuracys[run_idx] = self.classifier.evaluate(
                x_test, keras.utils.to_categorical(y_test, self.num_classes))

            print(' ')
            print('test loss: ' + str(losses[run_idx]))
            print('test_accuracy: ' + str(accuracys[run_idx]))

            confusion_matrizes[run_idx] = self.calculate_confusion_matrix(x_test, y_test)
            print(np.round(confusion_matrizes[run_idx], 2))

        mean_acc = np.mean(accuracys)
        print('Mean accuracy: ' + str(mean_acc))
        std_dev_acc = np.std(accuracys)
        print('Std dev accuracy: ' + str(std_dev_acc))
        mean_confusion_matrix = np.mean(confusion_matrizes, axis=0)
        print(np.round(mean_confusion_matrix, 2))
        self.plot_confusion_matrix(mean_confusion_matrix)

    def test_handmade_dataset(self, weights_file, dataset='opt'):
        data_path = 'data/handmade_sen12_subset/handmade_dataset_' + dataset + '.hdf5'
        x_test, y_test, _, _, _, _ = data_io.divide_dataset_eurosat(0.0, 0.0, path=data_path)

        print(x_test.shape)
        print(y_test.shape)

        p = np.random.permutation(x_test.shape[0])
        x_test = x_test[p]
        y_test = y_test[p]

        x_test = x_test / 127.5 - 1

        if dataset != 'opt':
            x_test = np.concatenate((x_test, x_test, x_test), axis=-1)

        self.classifier.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        num_runs = 5
        losses = np.zeros(num_runs, dtype=np.float32)
        accuracys = np.zeros(num_runs, dtype=np.float32)
        confusion_matrizes = np.zeros((num_runs, self.num_classes, self.num_classes), dtype=np.float32)
        for i in range(num_runs):

            weights = weights_file + str(i+1) + '.hdf5'
            self.classifier.load_weights(weights)

            losses[i], accuracys[i] = self.classifier.evaluate(x_test, keras.utils.to_categorical(y_test, 10))

            confusion_matrizes[i] = self.calculate_confusion_matrix(x_test, y_test)

        mean_acc = np.mean(accuracys)
        print('Mean accuracy: ' + str(mean_acc))
        std_dev_acc = np.std(accuracys)
        print('Std dev accuracy: ' + str(std_dev_acc))
        mean_confusion_matrix = np.mean(confusion_matrizes, axis=0)
        print(np.round(mean_confusion_matrix, 2))
        self.plot_confusion_matrix(mean_confusion_matrix)

    def calculate_confusion_matrix(self, x_test, y_test, num_classes=0):
        number_of_classes = self.num_classes if num_classes == 0 else num_classes
        conf = np.zeros((number_of_classes, number_of_classes))
        for i in range(number_of_classes):
            x = x_test[y_test == i]
            y_pred = np.argmax(self.classifier.predict(x), axis=1)
            for j in range(number_of_classes):
                conf[j, i] = y_pred[y_pred == j].shape[0] / y_pred.shape[0]
        return conf

    def plot_confusion_matrix(self, conf, class_labels='auto'):
        import itertools
        number_of_classes = conf.shape[0]
        labels_of_classes = self.class_names if class_labels == 'auto' else class_labels
        conf = np.round(conf, 2)
        plt.imshow(conf, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xlabel('correct class')
        plt.ylabel('predicted class')
        plt.xticks(np.arange(number_of_classes), labels_of_classes, rotation=45)
        plt.yticks(np.arange(number_of_classes), labels_of_classes)
        # show values in matrix:
        thresh = 0.5
        for i,j in itertools.product(range(number_of_classes), range(number_of_classes)):
            plt.text(j, i, str(conf[i, j]), horizontalalignment='center',
                     color='white' if conf[i, j] > thresh else 'black')
        # plt.tight_layout()
        plt.show()

    def save_trained_model(self, name):
        self.classifier.save_weights(MODEL_WEIGHTS_PATH + name + '.hdf5')

    def load_trained_model(self, name):
        self.classifier.load_weights(MODEL_WEIGHTS_PATH + name + '.hdf5')

    def apply(self, batch):
        return self.classifier.predict(batch)


if __name__ == '__main__':
    classifier = Custom_Classifer('resnet50', pre_trained=True)
    # weights_file = 'models/writing/classifier/2019-02-07 11:16:56.935895/resnet50_sar_best_weights_run_'
    # classifier.test_handmade_dataset(weights_file=weights_file, dataset='SAR')
    classifier.evaluate(dataset='optical', grayscale=True)
    # classifier.load_trained_model('vgg_opt')
    # classifier.train_sar()
    # classifier.train_sar_partial()
    # classifier.train_optical()
    # for i in range(10):
    #     classifier.inspect_sar_data(i, ['2019-02-01 22:24:23.763360_sets_0_4_10_18_19_20_24_25_32_38_39_41_45_47_49_51_53_54_56_57_intermediate'])
