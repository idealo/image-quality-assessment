import tensorflow.keras.backend as K
import argparse
from tensorflow.keras.applications.mobilenet import DepthwiseConv2D, relu6
from tensorflow.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import \
    predict_signature_def

from src.handlers.model_builder import Nima


def main(base_model_name, weights_file, export_path):
    # Load model and weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # Tell keras that this will be used for making predictions
    K.set_learning_phase(0)

    # CustomObject required by MobileNet
    with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
        builder = saved_model_builder.SavedModelBuilder(export_path)
        signature = predict_signature_def(
            inputs={'input_image': nima.nima_model.input},
            outputs={'quality_prediction': nima.nima_model.output}
        )

        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[tag_constants.SERVING],
            signature_def_map={'image_quality': signature}
        )
        builder.save()

    print(f'TF model exported to: {export_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-ep', '--export-path', help='path to save the tfs model', required=True)

    args = parser.parse_args()

    main(**args.__dict__)
