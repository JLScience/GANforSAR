#
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import Adam

import data_io

MODEL_WEIGHTS_PATH = 'models/'
EPOCHS = 20
BATCH_SIZE = 50

class Custom_VGG16():

    def __init__(self):
        self.num_classes = 10
        self.input_shape = (64, 64, 3)
        self.classifier = self.build_model()
        self.classifier.summary()

    def build_model(self):
        vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(64, 64, 3))
        vgg16_model.trainable = False
        inp = Input(shape=(64, 64, 3))
        outp = vgg16_model(inp)
        outp = Flatten()(outp)
        outp = Dropout(0.3)(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dropout(0.3)(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dense(10, activation='softmax')(outp)
        return Model(inp, outp)

    def train_optical(self):
        opt = Adam()
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # load dataset
        x_train, y_train, x_val, y_val, x_test, y_test = data_io.divide_dataset_eurosat(0.1, 0.1)

        # preprocess data
        x_train = np.array(x_train / 127.5 - 1, dtype=np.float32)
        x_val = np.array(x_val / 127.5 - 1, dtype=np.float32)
        x_test = np.array(x_test / 127.5 - 1, dtype=np.float32)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # train classifier
        self.classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val), verbose=2)
        loss, acc = self.classifier.evaluate(x_test, y_test)
        print(loss, acc)

        # TODO: save weights (in between)

    def train_sar(self):
        opt = Adam()
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # load dataset
        path = 'data/EuroSAT/dataset_translated.hdf5'
        x_train, y_train, x_val, y_val, x_test, y_test = data_io.divide_dataset_eurosat(0.1, 0.1, path=path)

        # preprocess data
        x_train = np.array(x_train / 127.5 - 1, dtype=np.float32)
        x_val = np.array(x_val / 127.5 - 1, dtype=np.float32)
        x_test = np.array(x_test / 127.5 - 1, dtype=np.float32)

        x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
        x_val = np.concatenate((x_val, x_val, x_val), axis=-1)
        x_test = np.concatenate((x_test, x_test, x_test), axis=-1)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_val = keras.utils.to_categorical(y_val, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # TODO: train classifier
        self.classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val),
                            verbose=2)
        loss, acc = self.classifier.evaluate(x_test, y_test)
        print(loss, acc)

        # TODO: save weights (in between)

    def save_trained_model(self):
        self.classifier.save_weights(MODEL_WEIGHTS_PATH + 'classifier.hdf5')

    def load_trained_model(self):
        self.classifier.load_weights(MODEL_WEIGHTS_PATH + 'classifier_trained.hdf5')

    def apply(self, batch):
        return self.classifier.predict(batch)


if __name__ == '__main__':
    my_vgg = Custom_VGG16()
    my_vgg.train_sar()
