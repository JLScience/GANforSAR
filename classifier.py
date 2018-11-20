#
import numpy as np
import keras

vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(64, 64, 3))
# resnet_model = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(64, 64, 3))
# mobilenet_model = keras.applications.mobilenet.MobileNet(include_top=False)

vgg16_model.summary()
# resnet_model.summary()
# mobilenet_model.summary()

# x = np.random.rand(1, 64, 64, 3)
# y = mobilenet_model.predict(x)
