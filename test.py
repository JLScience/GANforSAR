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


# # TEST IF CLASSIFIER WORKS AT ALL: CHECK
# import numpy as np
# import matplotlib.pyplot as plt
# import data_io
# from classifier import Custom_Classifer
# data_list = data_io.load_dataset_eurosat()
# my_vgg = Custom_Classifer('resnet50')
# my_vgg.load_trained_model('resnet50_opt_5')
# for i, d in enumerate(data_list):
#     d_norm = np.array(d / 127.5 - 1, dtype=np.float32)
#     pred = my_vgg.apply(d_norm)
#     fig, axs = plt.subplots(1, 7)
#     indizes = np.arange(d.shape[0]-7, d.shape[0], 1)
#     print(indizes.shape)
#     for j in range(7):
#         axs[j].imshow(d[indizes[j]])
#         label = np.argmax(pred[indizes[j], :])
#         print('Label: {}, {}'.format(label, i))
#         axs[j].set_title('l: {}, c.: {}%'.format(my_vgg.class_names_ger[label], int(100 * pred[indizes[j], label])))
#         axs[j].axis('off')
#     plt.show()

# # TEST IF CLASSIFIER WORKS WITH SEN12: WRONG
# import numpy as np
# import matplotlib.pyplot as plt
# import data_io
# import augmentation
# from classifier import Custom_Classifer
# data, _, _, _ = data_io.load_Sen12_data('/home/jlscience/PycharmProjects/SAR_GAN/data/Sen1-2/summer/', [4], split_ratio=1.0)
# data = augmentation.split_images(data, 4)
# data_norm = np.array(data / 127.5 - 1, dtype=np.float32)
# print(data.shape)
# my_vgg = Custom_Classifer('resnet50')
# my_vgg.load_trained_model('resnet50_opt_5')
# indizes = [20, 60, 300, 450, 750, 980, 2000]
# pred = my_vgg.apply(data_norm)
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
