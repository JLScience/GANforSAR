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


def classify_dataset(classifier, dataset_path):
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                   'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    f_opt = h5py.File(dataset_path + 'optical_dataset.hdf5')
    f_sar = h5py.File(dataset_path + 'sar_dataset.hdf5')
    idx_list = np.array(f_sar['index_list'], dtype=np.uint8)
    # for idx in idx_list:
    for idx in [41]:
        # load data:
        sar = f_sar['s1_' + str(idx)]
        opt = f_opt['s2_' + str(idx)]
        print(opt.shape)

        # cut images:
        sar = augmentation.split_images(sar, factor=4)
        opt = augmentation.split_images(opt, factor=4)
        print(opt.shape)

        # create a normalized copy of the optical data to put into the network
        opt_norm = np.array(opt / 127.5 - 1, dtype=np.float32)

        # apply classifier:
        y = apply_opt_classifier(classifier, opt_norm)
        print(y.shape)

        idz = [100, 350, 3500, 7200, 10500, 11500, 12000]
        fig, axs = plt.subplots(1, len(idz))
        for i, id in enumerate(idz):
            axs[i].imshow(opt[id])
            label = np.argmax(y[id, :])
            axs[i].set_title('l: {}, c.: {}%'.format(class_names[label], int(100*y[id, label])))
            axs[i].axis('off')
        plt.show()



def main():
    classifier = load_opt_classifier('vgg_opt')
    dataset_path = '/home/jlscience/PycharmProjects/SAR_GAN/data/Sen1-2/summer/'
    classify_dataset(classifier, dataset_path)


if __name__ == '__main__':
    main()
