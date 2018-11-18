#
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def transpose(tensor):
    rotated = np.zeros(tensor.shape, dtype=tensor.dtype)
    rows = tensor.shape[1]
    cols = tensor.shape[2]
    for i in range(rows):
        for j in range(cols):
            rotated[:, i, j, :] = tensor[:, j, i, :]
    return rotated


def reflect_vertical(tensor):
    rotated = np.zeros(tensor.shape, dtype=tensor.dtype)
    rows = tensor.shape[1]
    cols = tensor.shape[2]
    for i in range(rows):
        for j in range(cols):
            rotated[:, i, j, :] = tensor[:, i, cols - 1 - j, :]
    return rotated


def reflect_horizontal(tensor):
    rotated = np.zeros(tensor.shape, dtype=tensor.dtype)
    rows = tensor.shape[1]
    cols = tensor.shape[2]
    for i in range(rows):
        for j in range(cols):
            rotated[:, i, j, :] = tensor[:, cols - 1 - i, j, :]
    return rotated


def rotate_90_clockwise(tensor):
    return reflect_vertical(transpose(tensor))


def rotate_180(tensor):
    return reflect_horizontal(reflect_vertical(tensor))


def rotate_270_clockwise(tensor):
    return reflect_horizontal(transpose(tensor))


def apply_all(tensor1, tensor2):
    funcs = [transpose, reflect_horizontal, reflect_vertical, rotate_90_clockwise, rotate_180, rotate_270_clockwise]
    descriptions = ['Apply transposition \t\t\t\t (1/6) ...',
                    'Apply horizontal reflection \t\t (2/6) ...',
                    'Apply vertical reflection \t\t\t (3/6) ...',
                    'Apply rotation (90 deg clockwise) \t (4/6) ...',
                    'Apply rotation (180 deg clockwise) \t (5/6) ...',
                    'Apply rotation (270 deg clockwise) \t (6/6) ...', ]
    out1 = np.zeros(tensor1.shape, dtype=tensor1.dtype)
    out2 = np.zeros(tensor2.shape, dtype=tensor2.dtype)
    step = np.array(np.linspace(0, tensor1.shape[0], len(funcs)+1), dtype=np.uint32)
    for i, f in enumerate(funcs):
        print(descriptions[i])
        out1[step[i]:step[i+1], ...] = f(tensor1[step[i]:step[i+1], ...])
        out2[step[i]:step[i+1], ...] = f(tensor2[step[i]:step[i+1], ...])
    print('Done!')
    return out1, out2


def lee_filter(img, filter_size):
    img_mean = uniform_filter(img, (filter_size, filter_size))
    img_sqr_mean = uniform_filter(img**2, (filter_size, filter_size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_out = img_mean + img_weights * (img - img_mean)

    return img_out


def lee_filter_dataset(dataset_sar_norm, window_size=3):
    dataset_sar_filtered = np.zeros(dataset_sar_norm.shape, dataset_sar_norm.dtype)
    for i in range(dataset_sar_norm.shape[0]):
        dataset_sar_filtered[i, :, :, 0] = lee_filter(dataset_sar_norm[i, :, :, 0], window_size)
    return dataset_sar_filtered


# implemented for factor in [2, 4] and num_images_per_split <= factor and in [0, 1, 2, 4]
def split_images(dataset, factor, num_images_per_split=0):
    num = dataset.shape[0]
    width = dataset.shape[1]
    channels = dataset.shape[3]
    w = int(width / factor)

    if num_images_per_split != 0:
        new_dset = np.zeros((num * num_images_per_split, w, w, channels), dtype=dataset.dtype)
    else:
        new_dset = np.zeros((num * factor ** 2, w, w, channels), dtype=dataset.dtype)

    ind_list = [0]
    if num_images_per_split == 2:
        if factor == 4:
            ind_list = [0, 2]
        else:
            ind_list = [0, 1]
    elif num_images_per_split == 4:
        ind_list = [0, 1, 2, 3]

    counter = 0
    for i in range(factor):
        for j in range(factor):
            if num_images_per_split == 0:
                new_dset[counter * num:(counter + 1) * num, :, :, :] = dataset[:, i * w:(i + 1) * w, j * w:(j + 1) * w,
                                                                       :]
                counter += 1
            else:
                if i == j and i in ind_list:
                    new_dset[counter * num:(counter + 1) * num, :, :, :] = dataset[:, i * w:(i + 1) * w,
                                                                           j * w:(j + 1) * w, :]
                    counter += 1

    return new_dset


#  - - - - - TESTING  - - - - -

def test_1():
    import data_io
    import matplotlib.pyplot as plt
    dataset_opt_train, dataset_sar_train, dataset_opt_test, dataset_sar_test = data_io.load_dataset('data/')
    print(dataset_sar_test.shape)
    trasposed = transpose(dataset_sar_test)
    reflected_v = reflect_vertical(dataset_sar_test)
    reflected_h = reflect_horizontal(dataset_sar_test)
    rotated_90 = rotate_90_clockwise(dataset_sar_test)
    rotated_180 = rotate_180(dataset_sar_test)
    rotated_270 = rotate_270_clockwise(dataset_sar_test)

    names = ['normal', 'transposed', 'refl_v', 'refl_h', 'r90', 'r180', 'r270']

    images = []
    num = 100
    images.append(dataset_sar_test[num, :, :, 0])
    images.append(trasposed[num, :, :, 0])
    images.append(reflected_v[num, :, :, 0])
    images.append(reflected_h[num, :, :, 0])
    images.append(rotated_90[num, :, :, 0])
    images.append(rotated_180[num, :, :, 0])
    images.append(rotated_270[num, :, :, 0])


    print(dataset_opt_test.shape)
    dataset_opt_test = dataset_opt_test * 0.5 + 0.5
    trasposed = transpose(dataset_opt_test)
    reflected_v = reflect_vertical(dataset_opt_test)
    reflected_h = reflect_horizontal(dataset_opt_test)
    rotated_90 = rotate_90_clockwise(dataset_opt_test)
    rotated_180 = rotate_180(dataset_opt_test)
    rotated_270 = rotate_270_clockwise(dataset_opt_test)

    images2 = []
    images2.append(dataset_opt_test[num, :, :, :])
    images2.append(trasposed[num, :, :, :])
    images2.append(reflected_v[num, :, :, :])
    images2.append(reflected_h[num, :, :, :])
    images2.append(rotated_90[num, :, :, :])
    images2.append(rotated_180[num, :, :, :])
    images2.append(rotated_270[num, :, :, :])

    fig, axs = plt.subplots(2, 7, figsize=(21, 3))
    for i in range(2):
        for j in range(7):
            if i == 0:
                axs[i, j].imshow(images[j], cmap='gray')
            else:
                axs[i, j].imshow(images2[j])
            axs[i, j].set_title(names[j])
            axs[i, j].axis('off')
    plt.show()


def test_2():
    import data_io
    import matplotlib.pyplot as plt
    dataset_opt_train, dataset_sar_train, dataset_opt_test, dataset_sar_test = data_io.load_dataset('data/')
    print(dataset_sar_test.shape)

    dataset_opt_test = dataset_opt_test * 0.5 + 0.5
    dataset_sar_test = dataset_sar_test * 0.5 + 0.5
    n_s = 10
    augmented_opt_test, augmented_sar_test = apply_all(dataset_opt_test[0:n_s, ...], dataset_sar_test[0:n_s, ...])

    fig, axs = plt.subplots(4, n_s, figsize=(n_s*3, 8))
    for i in range(2):
        for j in range(n_s):
            if i == 0:
                axs[0, j].imshow(dataset_sar_test[j, :, :, 0], cmap='gray')
                axs[1, j].imshow(augmented_sar_test[j, :, :, 0], cmap='gray')
            else:
                axs[2, j].imshow(dataset_opt_test[j, :, :, :])
                axs[3, j].imshow(augmented_opt_test[j, :, :, :])
            # axs[i, j].set_title(names[j])
            axs[i, j].axis('off')
            axs[i+2, j].axis('off')
    plt.show()


def test_lee_filter():
    import data_io
    import matplotlib.pyplot as plt

    opt, sar, _, _ = data_io.load_Sen12_data(portion_mode=[10], split_mode='same', split_ratio=0.9)
    opt = np.array(opt / 127.5 - 1, dtype=np.float32)
    sar = np.array(sar / 127.5 - 1, dtype=np.float32)

    sar_f = np.zeros(sar.shape, dtype=sar.dtype)
    for i in range(sar.shape[0]):
        if i % 100 == 0:
            print(i)
        sar_f[i, :, :, 0] = lee_filter(sar[i, :, :, 0], 3)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(sar[10, :, :, 0], cmap='gray')
    axs[1].imshow(sar_f[10, :, :, 0], cmap='gray')
    plt.show()

    # fig, axs = plt.subplots(5, 5)
    # for j in range(5):
    #     sar_f_10 = lee_filter(sar[j+10, :, :, 0], 3)
    #     sar_f_20 = lee_filter(sar[j+10, :, :, 0], 4)
    #     sar_f_30 = lee_filter(sar[j+10, :, :, 0], 5)
    #     opt = 0.5 * opt + 0.5
    #     sar = 0.5 * sar + 0.5
    #     sar_f_10 = 0.5 * sar_f_10 + 0.5
    #     sar_f_20 = 0.5 * sar_f_20 + 0.5
    #     sar_f_30 = 0.5 * sar_f_30 + 0.5
    #     axs[0, j].imshow(sar[j+10, :, :, 0], cmap='gray')
    #     axs[1, j].imshow(sar_f_10, cmap='gray')
    #     axs[2, j].imshow(sar_f_20, cmap='gray')
    #     axs[3, j].imshow(sar_f_30, cmap='gray')
    #     axs[4, j].imshow(opt[j+10, :, :, :])
    #     for i in range(5):
    #         axs[i, j].axis('off')
    # plt.show()


def test_split_images_old():
    import data_io
    import matplotlib.pyplot as plt

    opt, sar, _, _ = data_io.load_Sen12_data(portion_mode=[10], split_mode='same', split_ratio=0.9)
    num_samples = opt.shape[0]

    opt_ = split_images(opt, 4)
    sar_ = split_images(sar, 4)

    opt = np.array(opt / 127.5 - 1, dtype=np.float32)
    sar = np.array(sar / 127.5 - 1, dtype=np.float32)
    opt_ = np.array(opt_ / 127.5 - 1, dtype=np.float32)
    sar_ = np.array(sar_ / 127.5 - 1, dtype=np.float32)
    opt = 0.5 * opt + 0.5
    sar = 0.5 * sar + 0.5
    opt_ = 0.5 * opt_ + 0.5
    sar_ = 0.5 * sar_ + 0.5

    fig, axs = plt.subplots(2, 5)
    axs[0, 0].imshow(opt[0, :, :, :])
    axs[1, 0].imshow(sar[0, :, :, 0], cmap='gray')
    for j in range(1, 5):
        axs[0, j].imshow(opt_[(j-1)*num_samples, :, :, :])
        axs[1, j].imshow(sar_[(j-1)*num_samples, :, :, 0], cmap='gray')
    for i in range(2):
        for j in range(5):
            axs[i, j].axis('off')
    plt.show()


def test_split_images():
    import matplotlib.pyplot as plt
    import imageio
    im1 = imageio.imread('data/EuroSAT/AnnualCrop/AnnualCrop_12.jpg')
    im2 = imageio.imread('data/EuroSAT/AnnualCrop/AnnualCrop_13.jpg')
    tensor = np.concatenate([im1.reshape((1, 64, 64, 3)), im2.reshape((1, 64, 64, 3))], axis=0)
    print(tensor.shape)
    tensor_small = split_images(tensor, 4, 4)
    print(tensor_small.shape)
    fig, axs = plt.subplots(2, 5)
    axs[0, 0].imshow(tensor[0, ...])
    axs[0, 1].imshow(tensor_small[0, ...])
    axs[0, 2].imshow(tensor_small[2, ...])
    axs[0, 3].imshow(tensor_small[4, ...])
    axs[0, 4].imshow(tensor_small[6, ...])
    axs[1, 0].imshow(tensor[1, ...])
    axs[1, 1].imshow(tensor_small[1, ...])
    axs[1, 2].imshow(tensor_small[3, ...])
    axs[1, 3].imshow(tensor_small[5, ...])
    axs[1, 4].imshow(tensor_small[7, ...])
    plt.show()

# - - - - - - - - - - - - - - -

test_split_images()