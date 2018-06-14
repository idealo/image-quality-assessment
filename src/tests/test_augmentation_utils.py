
import unittest
import numpy as np
from utils import utils
from unittest.mock import patch


class TestUtils(unittest.TestCase):

    @patch('numpy.random.randint')
    def test_random_crop(self, mock_np_random_randint):
        mock_np_random_randint.return_value = 1

        test_img = np.expand_dims(np.array([[0, 255], [0, 255]]), axis=2)
        crop_dims = (1, 1)
        cropped_img = utils.random_crop(test_img, crop_dims)
        self.assertEqual([255], cropped_img)

    @patch('numpy.random.random')
    def test_random_flip(self, mock_np_random_randint):
        mock_np_random_randint.return_value = 0
        temp = np.array([[0, 255], [0, 255]])
        test_img = np.dstack((temp, temp, temp))

        temp = np.array([[255, 0], [255, 0]])
        expected = np.dstack((temp, temp, temp))

        flipped_img = utils.random_horizontal_flip(test_img)
        np.testing.assert_array_equal(expected, flipped_img)

    def test_normalize_label(self):
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        normed_label = utils.normalize_labels(labels)
        np.testing.assert_array_equal(np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]), normed_label)
