from keras import backend as K

def msle(y_true, y_pred):
    return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))
