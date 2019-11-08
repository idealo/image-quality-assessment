
import os
import numpy as np
import tensorflow as tf
from utils import utils


class TrainDataGenerator(tf.keras.utils.Sequence):
    '''inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator'''
    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(256, 256), img_crop_dims=(224, 224), shuffle=True):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.shuffle = shuffle
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_crop_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = utils.load_image(img_file, self.img_load_dims)
            if img is not None:
                img = utils.random_crop(img, self.img_crop_dims)
                img = utils.random_horizontal_flip(img)
                X[i, ] = img

            # normalize labels
            y[i, ] = utils.normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


class TestDataGenerator(tf.keras.utils.Sequence):
    '''inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator'''
    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(224, 224)):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_load_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = utils.load_image(img_file, self.img_load_dims)
            if img is not None:
                X[i, ] = img

            # normalize labels
            if sample.get('label') is not None:
                y[i, ] = utils.normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y
