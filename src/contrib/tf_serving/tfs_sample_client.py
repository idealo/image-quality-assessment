from __future__ import absolute_import, division, print_function

import tensorflow as tf
import keras
import numpy as np
import skimage
from scipy import misc
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

TFS_HOST = 'localhost'
TFS_PORT = 8500
MODEL_NAME = 'mobilenet_technical'


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist*np.arange(1, 11)).sum()


def get_image_quality_predictions(image_path):
    # Load and preprocess image
    image = skimage.io.imread(image_path)
    image = misc.imresize(image, (224, 224), interp='nearest')
    image = np.float32(image)
    image = keras.applications.mobilenet.preprocess_input(image)

    # Run through model
    channel = implementations.insecure_channel(TFS_HOST, TFS_PORT)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = 'image_quality'

    request.inputs['input_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(np.expand_dims(image, 0)))

    response = stub.Predict(request, 10.0)
    result = round(calc_mean_score(response.outputs['quality_prediction'].float_val), 2)

    return result


print(get_image_quality_predictions('/home/bmachin/Downloads/technical1.jpg'))
