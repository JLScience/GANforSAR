# imports:
import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout

# imports (self made):
import data_io
import augmentation


def load_opt_classifier(name):
    vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(64, 64, 3))
    vgg16_model.trainable = False
    inp = Input(shape=(64, 64, 3))
    outp = vgg16_model(inp)
    outp = Flatten()(outp)
    outp = Dropout(0.4)(outp)
    outp = Dense(512, activation='relu')(outp)
    outp = Dropout(0.4)(outp)
    outp = Dense(512, activation='relu')(outp)
    outp = Dense(10, activation='softmax')(outp)
    model = Model(inp, outp)
    model.load_weights('models/classifier/' + name + '.hdf5')
    return model


def apply_opt_classifier(classifier, inp):
    # inp has to be of shape (num_images, 64, 64, 3)
    return classifier.predict(inp)


def classify_dataset(model, dataset_path):
    pass


def main():
    model = load_opt_classifier('vgg_opt')
    dataset_path = '/home/jlscience/PycharmProjects/SAR_GAN/data/Sen1-2/summer/'
    classify_dataset(model, dataset_path)


if __name__ == '__main__':
    main()
