from random import randint

import numpy as np
import tensorflow as tf

'''
    This function peforms various data augmentation techniques to the dataset

    @parameters:
        dataset: the feature training dataset in numpy array with shape [num_examples, num_rows, num_cols, num_channels] (since it is an image in numpy array)
        dataset_labels: the corresponding training labels of the feature training dataset in the same order, and numpy array with shape [num_examples, <anything>]
        augmentation_factor: how many times to perform augmentation.
        use_random_rotation: whether to use random rotation. default: true
        use_random_shift: whether to use random shift. default: true
        use_random_shear: whether to use random shear. default: true
        use_random_zoom: whether to use random zoom. default: true

    @returns:
        augmented_image: augmented dataset
        augmented_image_labels: labels corresponding to augmented dataset in order.

    for the augmentation techniques documentation, go here:
    	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_rotation
    	https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_shear
        https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_shift
        https://www.tensorflow.org/api_docs/python/tf/contrib/keras/preprocessing/image/random_zoom
'''


def augment_data(dataset, dataset_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True,
                 use_random_shift=True, use_random_zoom=True):
    augmented_image = []
    augmented_image_labels = []

    for num in range(0, dataset.shape[0]):

        for i in range(0, augementation_factor):
            # original image:
            augmented_image.append(dataset[num])
            augmented_image_labels.append(dataset_labels[num])

            if use_random_rotation:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1,
                                                                         channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shear:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1,
                                                                      channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shift:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1,
                                                                      channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_zoom:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], 0.9, row_axis=0, col_axis=1,
                                                                     channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

    return np.array(augmented_image), np.array(augmented_image_labels)


def crop(data, dim=24, rand=False):
    """Crop image data to dim x dim, selector for random cencer or centered image

    # Arguments
    Data = 4D np.array[number, height, width, channels]
    dim = output shape image, default 24x24
    random to choose if randomized center of cropped image or not, default: False

    # Returns
    np.array of cropped image data
    """

    if data.shape[1] < dim:
        raise ValueError('Cropped image cant be bigger than original, set karg dim < ', data.shape[1])
    if data.ndim != 4:
        raise ValueError('Expected image array to have rank 4 [elements, heigth, width, channel]. '
                         'Set channe = 1 if in greyscale'
                         'Got array with shape:', data.shape)

    cropped_image = []
    to_crop = (data.shape[1] - dim) // 2

    if rand is True:
        for i in range(data.shape[0]):
            rnd = randint(0, to_crop)
            cropped_image.append(
                data[i, to_crop - rnd:data.shape[1] - to_crop - rnd, to_crop - rnd:data.shape[1] - to_crop - rnd, :])

        return np.array(cropped_image)

    else:
        return np.array(data[:, to_crop:data.shape[1] - to_crop, to_crop:data.shape[1] - to_crop, :])
