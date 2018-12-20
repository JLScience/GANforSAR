# import h5py
# print(h5py.__version__)

# l = [3, 5, 9]
# print(str(l))

# import numpy as np
# import matplotlib.pyplot as plt
# data = np.loadtxt('train_output.txt')
# print(data.shape)
# x = np.arange(0, data.shape[0], 1)
# plt.plot(x, data[:, 1])
# plt.show()

# import numpy as np
# num_images_per_split = 2
# factor = 2
# num = 10
# width = 8
# w = int(width / factor)
# channels = 3
#
# dataset = np.arange(0, num*width*width*channels, 1).reshape((num, width, width, channels))
#
# if num_images_per_split != 0:
#     new_dset = np.zeros((num * num_images_per_split, w, w, channels), dtype=dataset.dtype)
# else:
#     new_dset = np.zeros((num * factor**2, w, w, channels), dtype=dataset.dtype)
#
# ind_list = [0]
# if num_images_per_split == 2:
#     if factor == 4:
#         ind_list = [0, 2]
#     else:
#         ind_list = [0, 1]
# elif num_images_per_split == 4:
#     ind_list = [0, 1, 2, 3]
#
# counter = 0
# for i in range(factor):
#     for j in range(factor):
#         if num_images_per_split == 0:
#             new_dset[counter*num:(counter+1)*num, :, :, :] = dataset[:, i*w:(i+1)*w, j*w:(j+1)*w, :]
#             counter += 1
#         else:
#             if i == j and i in ind_list:
#                 new_dset[counter*num:(counter+1)*num, :, :, :] = dataset[:, i*w:(i+1)*w, j*w:(j+1)*w, :]
#                 counter += 1
# print(dataset[0, :, :, 0])
# print(new_dset[0, :, :, 0])
# print(new_dset[10, :, :, 0])
# # print(new_dset[20, :, :, 0])
# # print(new_dset[30, :, :, 0])
# print(new_dset.shape)

# import keras
# resnet50 = keras.applications.resnet50.ResNet50()
# resnet50.summary()
#
# vgg16 = keras.applications.vgg16.VGG16()
# vgg16.summary()

# import sys
#
# lr_d = 0.1
# lr_g = 0.2
# print(sys.argv)
# idz = []
# for idx, arg in enumerate(sys.argv):
#     if 'd' in arg:
#         lr_d = float(arg.replace('d_', ''))
#         idz.append(idx)
#         # sys.argv.pop(idx)
#     if 'g' in arg:
#         lr_g = float(arg.replace('g_', ''))
#         idz.append(idx)
#         # sys.argv.pop(idx)
# for i in idz[::-1]:
#     sys.argv.pop(i)
# print(sys.argv)
# print(lr_d)
# print(lr_g)

# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras import optimizers
#
# epochs = 10
# batchsize = 50
#
# model = Sequential()
# model.add(Dense(100, activation='relu', input_shape=(1,), kernel_initializer='ones'))
# model.add(Dense(2, activation='softmax', kernel_initializer='ones'))
# model.summary()
#
# opt = keras.optimizers.Adam(lr=0.1, decay=0.1)
# model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
#
# print(model.optimizer.get_config()['lr'])
#
# x = np.arange(0, 1000, 1)
# y = keras.utils.to_categorical(x % 2, 2)
#
# for e in range(epochs):
#     for b in range(0, 1000, batchsize):
#         x_b = x[b:b+batchsize, ...]
#         y_b = y[b:b+batchsize, ...]
#         loss = model.train_on_batch(x_b, y_b)
#         if e == 4:
#             print('YO')
#             opt_tmp = keras.optimizers.Adam(lr=0.0001)
#             model.optimizer = opt_tmp
#             model.model.optimizer = optimizers.get(opt_tmp)
#             model.model.train_function = None
#         print('Epoch: {}, Batch: {}, loss: {}, lr: {}'.format(e, int(b/batchsize), loss[0], opt.get_config()['lr']))
#         # print('Epoch: {}, Batch: {}, loss: {}, lr: {}'.format(e, int(b/batchsize), loss[0], model.optimizer.lr.eval()))

# print(int(1.2))


# # TEST IF CLASSIFIER WORKS AT ALL:
# import numpy as np
# import matplotlib.pyplot as plt
# import data_io
# import augmentation
# from classifier import Custom_Classifer
# data_list = data_io.load_dataset_eurosat()
# my_vgg = Custom_Classifer('resnet50')
# my_vgg.load_trained_model('resnet50_opt_7_correct_norm')
# for i, d in enumerate(data_list):
#     # d_norm = np.array(d / 127.5 - 1, dtype=np.float32)
#     # d_norm = augmentation.to_gray(d_norm, mode=3)
#     d_norm = np.array(d[..., ::-1])
#     vgg19_means = [103.939, 116.779, 123.68]
#     for k in range(3):
#         d_norm[:, :, :, k] = np.array(d_norm[:, :, :, k] - vgg19_means[k], dtype=np.float32)
#     pred = my_vgg.apply(d_norm)
#     fig, axs = plt.subplots(1, 7)
#     indizes = np.arange(d.shape[0]-7, d.shape[0], 1)
#     for j in range(7):
#         axs[j].imshow(d[indizes[j]])
#         label = np.argmax(pred[indizes[j], :])
#         print('Label: {}, {}'.format(label, i))
#         axs[j].set_title('l: {}, c.: {}%'.format(my_vgg.class_names_ger[label], int(100 * pred[indizes[j], label])))
#         axs[j].axis('off')
#     plt.show()

# # TEST IF CLASSIFIER WORKS WITH SEN12:
# import numpy as np
# import matplotlib.pyplot as plt
# import data_io
# import augmentation
# from classifier import Custom_Classifer
# data, _, _, _ = data_io.load_Sen12_data('/home/jlscience/PycharmProjects/SAR_GAN/data/Sen1-2/summer/', [4], split_ratio=1.0)
# data = augmentation.split_images(data, 4)
# # data_norm = np.array(data / 127.5 - 1, dtype=np.float32)
# # data_norm = augmentation.to_gray(data_norm, mode=3)
# d_norm = np.array(data[..., ::-1])
# vgg19_means = [103.939, 116.779, 123.68]
# for k in range(3):
#     d_norm[:, :, :, k] = np.array(d_norm[:, :, :, k] - vgg19_means[k], dtype=np.float32)
# print(data.shape)
# my_vgg = Custom_Classifer('resnet50')
# my_vgg.load_trained_model('resnet50_opt_7_correct_norm')
# indizes = [20, 60, 300, 450, 750, 980, 2000]
# pred = my_vgg.apply(d_norm)
# fig, axs = plt.subplots(1, 7)
# for j in range(7):
#     axs[j].imshow(data[indizes[j]])
#     label = np.argmax(pred[indizes[j], :])
#     axs[j].set_title('l: {}, c.: {}%'.format(my_vgg.class_names_ger[label], int(100 * pred[indizes[j], label])))
#     axs[j].axis('off')
# plt.show()

# from classifier import Custom_Classifer
# my_net = Custom_Classifer('resnet50')

# import numpy as np
# x = np.zeros((5, 6, 6, 3))
# x[:, :, :, :] = np.ones((5, 6, 6))

# from keras.models import Model
# from keras.layers import Input, Lambda, BatchNormalization
# from keras import backend as K
# import keras
# import numpy as np
#
# def myFunc(x):
#     width = x.shape[1]
#     height = x.shape[2]
#     mean_tensor = K.mean(x, axis=[1,2,3])    # considering shapes of (size,width, heigth,channels)
#     print(mean_tensor.shape)
#     # std_tensor = K.std(x,axis=[0,1,2])
#     result = x
#     for i in range(2):
#         result[i, ...] = x[i, ...] - mean_tensor[1]
#     return result
#     # x = K.reshape(x, (-1, 1, 1, 1))    # shapes of mean and std are (3,) here.
#     # print(x.shape)
#     # result = (x - mean_tensor)  # / (std_tensor + K.epsilon())
#
#     # return K.reshape(result, (-1, width, height, 3))
#
# z = np.ones((2, 10, 10, 3))
# z[0, ...] = -0.5*z[0, ...]
# z[1, ...] = 0.5*z[1, ...]
# # print(np.min(z), np.max(z))
# print(z[0, :, :, 0])
# print(z[1, :, :, 0])
# inp = Input(shape=(10, 10, 3))
# x = Lambda(lambda x: (x+1)*127.5)(inp)
# # x = Lambda(lambda x: x / K.mean(x, axis=[1, 2, 3]))(x)
# # x = Lambda(myFunc)(x)
# # x = BatchNormalization(axis=-1, momentum=0.0, epsilon=0.0, scale=False)(x)
# x = Lambda(keras.applications.vgg19.preprocess_input)(x)
# model = Model(inp, x)
# model.compile(optimizer='Adam', loss='mse')
# out = model.predict(z)
# # print(np.min(out), np.max(out))
# print(out[0, :, :, 0])
# print(out[1, :, :, 0])
# print(out[0, :, :, 1])
# print(out[1, :, :, 1])
# model.summary()
# from keras.applications.resnet50 import preprocess_input

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--lr_g', type=float, default=1e-4, help='Learning rate', required=True)
# parser.add_argument('--lr_d', type=float, default=2e-4, help='Learning rate', required=False)
# parser.add_argument('--abc', type=int, default=0, help='ABC', required=False)
# args = parser.parse_args()
# my_lr_g = args.lr_g
# my_lr_d = args.lr_d
# print(my_lr_g)
# print(my_lr_d)
# print(args)

# import numpy as np
# z = np.zeros((10, 8, 8, 1))
# mz = np.mean(z, axis=0)
# print(mz.shape)
# zeta = np.zeros((10, 1))
# mzeta = np.mean(zeta, axis=0)
# print(mzeta.shape)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--x', nargs='+')
# args = parser.parse_args()
# x = args.x
# if len(x) > 1:
#     print('is list')
#     y = [int(a) for a in args.x]
#     print(type(y[0]))
# else:
#     try:
#         x = int(x[0])
#         print('was int')
#     except ValueError:
#         x = float(x[0])
#         print('was float')
#     print(x)

# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
#
# # generate training and test data:
# modulo = 10
# num_samples = 50000
# nums_up_to = 30                                                                   # numbers in range [0, nums_up_to-1]
# x_train = np.random.randint(low=0, high=nums_up_to, size=num_samples)
# y_train = np.mod(x_train, modulo)
# y_train = keras.utils.to_categorical(y_train, num_classes=modulo)                 # transform output to one-hot-encoding
#
# x_test = np.random.randint(low=nums_up_to, high=nums_up_to*2, size=num_samples)
# y_test = np.mod(x_test, modulo)
# y_test = keras.utils.to_categorical(y_test, num_classes=modulo)                   # transform output to one-hot-encoding
#
# # design and compile model:
# model = Sequential()
# model.add(Dense(100, activation='sigmoid', input_shape=(1, )))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(modulo, activation='sigmoid'))
# model.summary()
# model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])               # use mean squared error loss
#
# # train and evaluate model:
# model.fit(x_train, y_train, epochs=50, verbose=2)
# print('')
# print('Test the model ...')
# print('Test with same numbers as in training: ')
# hist = model.evaluate(x_train, y_train, verbose=2)
# print('Achieved {}% accuracy!'.format(hist[1]*100))
# print('Test with unknown numbers: ')
# hist = model.evaluate(x_test, y_test, verbose=2)
# print('Achieved {}% accuracy!'.format(hist[1]*100))

