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
