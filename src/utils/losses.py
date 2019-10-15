
from tensorflow.keras import backend as K


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)
