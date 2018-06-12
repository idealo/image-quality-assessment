import unittest
import os
from data.AVA.scripts import get_labels as ava_get_labels
from data.tid2013.scripts import get_labels as tid2013_get_labels
import numpy as np


class TestGetLabels(unittest.TestCase):

    def test_get_ava_labels(self):
        txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test-ava-samples.txt')

        with open(txt, 'r') as f:
            raw_labels = f.read().splitlines()
        samples = ava_get_labels.parse_raw_data(raw_labels)
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0]['image_id'], '953619')
        self.assertEqual(samples[0]['label'], [0, 1, 5, 17, 38, 36, 15, 6, 5, 1])
        self.assertEqual(len(samples[0]['label']), 10)

    def test_get_tid2013_labels(self):
        txt_mos = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test-tid-mos-samples.txt')
        txt_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test-tid-std-samples.txt')

        df = tid2013_get_labels.get_dataframe(txt_mos, txt_std)
        priors = tid2013_get_labels.get_priors(df)
        samples = tid2013_get_labels.parse_raw_data(df, priors)

        # test if df is 5
        self.assertEqual(len(df), 5)
        self.assertCountEqual(df.columns, ['mos', 'id', 'std', 'dist_grade'])

        # test if priors have proper shape
        self.assertEqual(np.array(priors).shape, (5, 10))

        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0]['image_id'], 'I01_01_1')
        self.assertEqual(len(samples[0]['label']), 10)
        np.testing.assert_almost_equal(np.array(samples[0]['label']).sum(), 1)

    def test_get_max_entropy(self):
        prior = np.array([1]+[0]*9)

        max_entropy_dist = tid2013_get_labels.get_max_entropy_distribution(mean=0, std=0, prior=prior)

        np.testing.assert_array_almost_equal(max_entropy_dist, prior)
