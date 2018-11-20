#
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam

MODEL_WEIGHTS_PATH = ''


class Custom_VGG16():

    def __init__(self):
        self.input_shape = (64, 64, 3)
        self.classifier = self.build_model()
        self.classifier.summary()

    def build_model(self):
        vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(64, 64, 3))
        vgg16_model.trainable = False
        inp = Input(shape=(64, 64, 3))
        outp = vgg16_model(inp)
        outp = Flatten()(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dense(512, activation='relu')(outp)
        outp = Dense(10, activation='softmax')(outp)
        return Model(inp, outp)

    def train(self):
        opt = Adam()
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # TODO: load datasets

        # TODO: preprocess datasets

        # TODO: train classifier

        # TODO: save weights (in between)


    def save_trained_model(self):
        self.classifier.save_weights(MODEL_WEIGHTS_PATH)

    def load_trained_model(self):
        self.classifier.load_weights(MODEL_WEIGHTS_PATH)

    def apply(self, batch):
        return self.classifier.predict(batch)


if __name__ == '__main__':
    my_vgg = Custom_VGG16()
    my_vgg.train()
    # x = np.random.rand(12, 64, 64, 3)
    # y = my_vgg.apply(x)
    # print(y.shape)
