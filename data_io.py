import numpy as np
import os
import h5py
import imageio


# - - - - - PROCEDURE [SEN1-2] - - - - -
#
#
#
# - - - - - ------------------ - - - - -

DATASET_SPRING_SIZE = 75724
DATASET_SUMMER_SIZE = 53508
DATASET_FALL_SIZE = 86370
DATASET_WINTER_SIZE = 66782

# - - - - - Functions for Sen1-2 dataset - - - - -

def create_dataset_sen12(is_sar, dataset_name):
    # Set Variables:
    image_shape = 256
    channels = 1 if is_sar else 3
    name = 'sar' if is_sar else 'optical'
    path = 'data/Sen1-2/' + dataset_name + '/'
    save_dir = 'data/Sen1-2/summer/' + name + '_dataset.hdf5'
    print('Save dataset from {} --to-> {} ...'.format(path, save_dir))
    # Access optical subdirectories:
    dirs = os.listdir(path)
    if is_sar:
        filtered_dirs = [dir for dir in dirs if ('s1' in dir)]
    else:
        filtered_dirs = [dir for dir in dirs if ('s2' in dir)]
    # Open .hdf5 file:
    f = h5py.File(save_dir)
    # Save index list:
    idz = np.zeros(len(filtered_dirs), dtype=np.uint32)
    for i, dir in enumerate(filtered_dirs):
        idz[i] = int(dir[dir.find('_')+1:])
    idz.sort()
    f.create_dataset('index_list', data=idz)
    # Create one dataset per directory:
    for i, dir in enumerate(filtered_dirs):
        # load image names and sort them:
        image_names = os.listdir(path + dir)
        image_nr = lambda image_name : int(image_name[image_name.find('_p')+2:image_name.find('.png')])
        image_names.sort(key=image_nr)
        print('Load files from folder {} of {} ...'.format(i+1, len(filtered_dirs)))
        dataset = np.zeros((len(image_names), image_shape, image_shape, channels), dtype=np.uint8)
        # Save each image of the scene:
        for j, image_name in enumerate(image_names):
            uri = path + dir + '/' + image_name
            image = imageio.imread(uri)
            image = np.array(image, dtype=np.uint8)
            if is_sar:
                image = np.reshape(image, (image_shape, image_shape, 1))
            dataset[j, :, :, :] = image
        f.create_dataset(dir, data=dataset)
    f.close()
    print('done!')


# create_dataset_sen12(is_sar=True, dataset_name='ROIs1868_summer')

def load_Sen12_data(path='data/Sen1-2/summer/', portion_mode=1.0, split_mode='same', split_ratio=0.8):
    """
    :param path:          location at which the optical and sar dataset can be found
    :param portion_mode:  used for specifying which parts of the data should be provided
                          if type == float: load datasets (scenes) until portion_mode >= num_samples / dataset_size
                          if type == list: load specified datasets
    :param split_mode:    defines how train and test set should be split
                          'same': split each scene into train an test
                          'separated': use different scenes for training and testing
    :param split_ratio:   specify the relative amount of training data
    :return:              sar and optical dataset, each split into train and test set
    """

    dataset_size = DATASET_SUMMER_SIZE
    f_opt = h5py.File(path + 'optical_dataset.hdf5')
    f_sar = h5py.File(path + 'sar_dataset.hdf5')

    datasets = []
    sizes = []

    # Load datasets:
    if type(portion_mode) == list:
        print('--> Load Sen12 data: input is list')
        for num in portion_mode:
            opt_dataset_name = 's2_{}'.format(num)
            sar_dataset_name = 's1_{}'.format(num)
            opt = np.array(f_opt[opt_dataset_name])
            sar = np.array(f_sar[sar_dataset_name])
            datasets.append([opt, sar])
            sizes.append(opt.shape[0])
    elif type(portion_mode) == float:
        print('--> Load Sen12 data: input is float')
        index_list = np.array(f_opt['index_list'])
        idx = 0
        while np.sum(sizes) < portion_mode * dataset_size:
            opt_dataset_name = 's2_{}'.format(index_list[idx])
            sar_dataset_name = 's1_{}'.format(index_list[idx])
            print('--> Load Sen12 data: ' + opt_dataset_name)
            opt = np.array(f_opt[opt_dataset_name])
            sar = np.array(f_sar[sar_dataset_name])
            datasets.append([opt, sar])
            sizes.append(opt.shape[0])
            idx += 1
    else:
        print('--> Load Sen12 data: incorrect input for parameter <portion_mode>')
        return -1

    # Split datasets:
    num_samples = np.sum(sizes)
    num_samples_train = int(num_samples * split_ratio)
    opt_train = np.zeros((num_samples_train, ) + datasets[0][0].shape[1:], dtype=np.uint8)
    sar_train = np.zeros((num_samples_train,) + datasets[0][1].shape[1:], dtype=np.uint8)
    opt_test = np.zeros((num_samples - num_samples_train,) + datasets[0][0].shape[1:], dtype=np.uint8)
    sar_test = np.zeros((num_samples - num_samples_train,) + datasets[0][1].shape[1:], dtype=np.uint8)
    sample_counter = 0
    idx = 0
    if split_mode == 'same':
        test_sample_counter = 0
        for idx in range(len(sizes)):
            # shuffle datasets:
            p = np.random.permutation(sizes[idx])
            datasets[idx][0] = datasets[idx][0][p]
            datasets[idx][1] = datasets[idx][1][p]
            # split the last scene such that train and test sets are completely filled:
            if idx == len(sizes) - 1:
                missing_train_samples = num_samples_train - sample_counter
                opt_train[sample_counter:, ...] = datasets[idx][0][:missing_train_samples, ...]
                sar_train[sample_counter:, ...] = datasets[idx][1][:missing_train_samples, ...]
                opt_test[test_sample_counter:, ...] = datasets[idx][0][missing_train_samples:, ...]
                sar_test[test_sample_counter:, ...] = datasets[idx][1][missing_train_samples:, ...]
            # split each scene into train and test (rounding issues are treated in if-condition)
            else:
                edge = int(sizes[idx]*split_ratio)
                opt_train[sample_counter:sample_counter+edge, ...] = datasets[idx][0][:edge, ...]
                sar_train[sample_counter:sample_counter+edge, ...] = datasets[idx][1][:edge, ...]
                opt_test[test_sample_counter:test_sample_counter+sizes[idx]-edge] = datasets[idx][0][edge:, ...]
                sar_test[test_sample_counter:test_sample_counter+sizes[idx]-edge] = datasets[idx][1][edge:, ...]
                sample_counter += edge
                test_sample_counter += sizes[idx] - edge
    elif split_mode == 'separated':
        # put complete scenes into training data until scene has to be split:
        while sample_counter + sizes[idx] < num_samples_train:
            opt_train[sample_counter:sample_counter+sizes[idx], ...] = datasets[idx][0]
            sar_train[sample_counter:sample_counter+sizes[idx], ...] = datasets[idx][1]
            sample_counter += sizes[idx]
            idx += 1
            # print('sample counter: {}'.format(sample_counter))
            # print('idx: {}'.format(idx))
        # split scene to match split_ratio:
        remainder = num_samples_train-sample_counter
        opt_train[sample_counter:, ...] = datasets[idx][0][:remainder, ...]
        sar_train[sample_counter:, ...] = datasets[idx][1][:remainder, ...]
        # print('-- remainder: {}'.format(remainder))
        sample_counter = sizes[idx] - remainder
        opt_test[:sample_counter] = datasets[idx][0][remainder:, ...]
        sar_test[:sample_counter] = datasets[idx][1][remainder:, ...]
        idx += 1
        # print('sample counter {}'.format(sample_counter))
        print('--> Load Sen12 data: split dataset at index: {}'.format(index_list[idx]))
        # put remaining data into test set:
        for i in range(idx, len(sizes)):
            opt_test[sample_counter:sample_counter+sizes[idx], ...] = datasets[idx][0]
            sar_test[sample_counter:sample_counter+sizes[idx], ...] = datasets[idx][1]
            sample_counter += sizes[idx]
            idx += 1
            # print('sample counter {}'.format(sample_counter))
            # print('idx: {}'.format(idx))
    else:
        print('--> Load Sen12 data: incorrect input for parameter <split_mode>')
        return -1

    print('--> Load Sen12 data: {} samples have been loaded ({} training, {} test, split ratio: {}).'.format(
        num_samples, num_samples_train, num_samples - num_samples_train, split_ratio))

    return opt_train, sar_train, opt_test, sar_test


# x, y, xt, yt = load_Sen12_data(portion_mode=0.1, split_mode='same', split_ratio=0.8)
#
# fig, axs = plt.subplots(1, 4, figsize=(8, 2))
# axs[0].imshow(x[1000, ...])
# axs[1].imshow(y[1000, :, :, 0], cmap='gray')
# axs[2].imshow(xt[1000, ...])
# axs[3].imshow(yt[1000, :, :, 0], cmap='gray')
# plt.show()


# convert .png to .tiff files
def png2tif(im_png, im_tiff):
    im = imageio.imread(im_png)
    imageio.imsave(im_tiff, im)


def img2noise(img_path, img_name, x, y, size):
    img = imageio.imread(img_path + img_name)
    img = img[y:y+size, x:x+size]
    imageio.imsave(img_path + 'noise.tiff', img)


# path = '/media/jlscience/Volume/Uni/Masterarbeit/SAR GAN/Sen1-2 Dataset/ROIs1868_summer/s1_0/'
# img2noise(path, 'ROIs1868_summer_s1_0_p1.png', 0, 128, 64)


def imgs2scene(imgs_path, img_name_tiff, num_imgs):
    img_dim = 256
    imgs_list = os.listdir(imgs_path)
    imgs_list.sort()
    if num_imgs > len(imgs_list):
        num_imgs = imgs_list
    scene = np.zeros((img_dim*num_imgs, img_dim), dtype=np.uint8)
    for i, img in enumerate(imgs_list[:num_imgs]):
        img = imageio.imread(imgs_path + img)
        print(img_dim*i)
        print(img_dim*(i+1))
        scene[img_dim*i:img_dim*(i+1), :] = img
    imageio.imsave(imgs_path + img_name_tiff, scene)


# path = '/media/jlscience/Volume/Uni/Masterarbeit/SAR GAN/Sen1-2 Dataset/ROIs1868_summer/s1_0/'
# imgs2scene(path, 'scene_10.tiff', 10)


# im = imageio.imread('noise.png')
# print(im.shape)
# import matplotlib.pyplot as plt
# plt.hist(im.reshape(im.shape[0]*im.shape[1]*3), bins=256)
# plt.show()
# png2tif('ROIs1868_summer_s1_4_p44.png', 'ROIs1868_summer_s1_4_p44.tiff')
# impng = imageio.imread('ROIs1868_summer_s1_4_p151.png')
# imageio.imsave('noise_sea.tiff', impng[:64, :64])
# impng = imageio.imread('ROIs1868_summer_s1_0_p1.png')
# imageio.imsave('noise_forest.tiff', impng[:64, :64])

# - - - - - ---------------------------- - - - - -


# - - - - - Functions for EuroSAT dataset - - - - -

def create_dataset_eurosat(path):
    img_size = 64
    img_channels = 3
    labels = os.listdir(path)
    print(labels)
    f = h5py.File(path + 'dataset.hdf5')
    for label in labels:
        num_labels = len(os.listdir(path + label))
        tensor = np.zeros((num_labels, img_size, img_size, img_channels), dtype=np.uint8)
        for i in range(num_labels):
            name = path + label + '/' + label + '_' + str(i+1) + '.jpg'
            tensor[i, :, :, :] = imageio.imread(name)
        # print('mean values for label {}: R: {}  G: {}  B: {}'.format(label, int(np.mean(tensor[:, :, :, 0])), int(np.mean(tensor[:, :, :, 1])), int(np.mean(tensor[:, :, :, 2]))))
        f.create_dataset(label, data=tensor)
        print(label + ' saved.')


def load_dataset_eurosat(path='data/EuroSAT/', mode='all'):
    names = ['Highway', 'SeaLake', 'Industrial', 'River', 'PermanentCrop',
             'Forest', 'AnnualCrop', 'HerbaceousVegetation', 'Pasture', 'Residential']
    f = h5py.File(path + 'dataset.hdf5')
    if mode in names:
        data = np.array(f[mode])
    if mode == 'all':
        data = np.zeros((0, 64, 64, 3), dtype=np.uint8)
        for name in names:
            data = np.append(data, np.array(f[name]), 0)
    return data


def divide_dataset_eurosat(val_fraction, test_fraction, path='data/EuroSAT/', mode='all'):
    pass

# create_dataset_eurosat('data/EuroSAT/')
# data = load_dataset_eurosat(mode='all')

# - - - - - ----------------------------- - - - - -


# - - - - - Functions for other datasets - - - - -

# USAGE: create_dataset_maps('ex_maps.hdf5', 'data/maps/')
def create_dataset_maps(dataset_name, file_location):
    shape = 256
    channels = 3
    training_location = file_location + 'train/'
    test_location = file_location + 'val/'
    print(training_location)
    print(test_location)

    f = h5py.File(file_location + dataset_name)

    # training data:
    files = os.listdir(training_location)
    files.sort()
    print('create training dataset ({} examples) ...'.format(len(files)))
    aerial = np.zeros((len(files), shape, shape, channels), dtype=np.uint8)
    map = np.zeros((len(files), shape, shape, channels), dtype=np.uint8)
    for i, file in enumerate(files):
        im = imageio.imread(training_location + file)
        aerial[i, :, :, :] = im[:shape, :shape, :]
        map[i, :, :, :] = im[:shape, 600:600+shape, :]
    print('save dataset ...')
    f.create_dataset('aerial_train', data=aerial)
    f.create_dataset('map_train', data=map)

    # test data
    files = os.listdir(test_location)
    files.sort()
    print('create test dataset ({} examples) ...'.format(len(files)))
    aerial = np.zeros((len(files), shape, shape, channels), dtype=np.uint8)
    map = np.zeros((len(files), shape, shape, channels), dtype=np.uint8)
    for i, file in enumerate(files):
        im = imageio.imread(test_location + file)
        aerial[i, :, :, :] = im[:shape, :shape, :]
        map[i, :, :, :] = im[:shape, 600:600+shape, :]
    print('save dataset ...')
    f.create_dataset('aerial_test', data=aerial)
    f.create_dataset('map_test', data=map)

    f.close()


# USAGE: atr, mtr, ate, mte = load_dataset_maps('data/maps/ex_maps.hdf5')
def load_dataset_maps(path):
    f = h5py.File(path)
    aerial_train = np.array(f['aerial_train'])
    map_train = np.array(f['map_train'])
    aerial_test = np.array(f['aerial_test'])
    map_test = np.array(f['map_test'])
    return aerial_train, map_train, aerial_test, map_test


# create_dataset_maps('ex_maps_small.hdf5', 'data/maps/')
# atr, mtr, ate, mte = load_dataset_maps('data/maps/ex_maps.hdf5')
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(atr[0, ...])
# axs[1].imshow(mtr[0, ...])
# plt.show()


# - - - - - ---------------------------- - - - - -
