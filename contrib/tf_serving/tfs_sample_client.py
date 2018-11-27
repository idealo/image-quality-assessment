import json
import argparse
import keras
import numpy as np
import tensorflow as tf
from src.utils import utils
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

TFS_HOST = 'localhost'
TFS_PORT = 8500


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def get_image_quality_predictions(image_path, model_name):
    # Load and preprocess image
    image = utils.load_image(image_path, target_size=(224, 224))
    image = keras.applications.mobilenet.preprocess_input(image)

    # Run through model
    channel = implementations.insecure_channel(TFS_HOST, TFS_PORT)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'image_quality'

    request.inputs['input_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(np.expand_dims(image, 0))
    )

    response = stub.Predict(request, 10.0)
    result = round(calc_mean_score(response.outputs['quality_prediction'].float_val), 2)

    print(json.dumps({'mean_score_prediction': np.round(result, 3)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image-path', help='Path to image file.', required=True)
    parser.add_argument(
        '-mn', '--model-name', help='mobilenet_aesthetic or mobilenet_technical', required=True
    )
    args = parser.parse_args()
    get_image_quality_predictions(**args.__dict__)
