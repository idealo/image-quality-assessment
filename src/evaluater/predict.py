import os
import glob
import json
import argparse
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
from keras import backend as K
from PIL import ImageFile, Image


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    splits = os.path.basename(img_path).split('.')
    splits.pop()
    img_id = ".".join(splits)

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        splits = os.path.basename(img_path).split('.')
        splits.pop()
        img_id = ".".join(splits)
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(base_model_name, weights_file, image_source, predictions_file, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)
    K.clear_session()

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    if predictions_file is not None:
        save_json(samples, predictions_file)

    return samples


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)

    args = parser.parse_args()

    main(**args.__dict__)
