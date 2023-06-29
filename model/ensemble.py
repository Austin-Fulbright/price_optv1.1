from keras.models import clone_model
from keras.optimizers import Adam
from utils.metrics import msle, rmse
from keras.callbacks import EarlyStopping
import numpy as np

num_models = 5
ensemble_predictions = []

def create_ensemble(best_model, X_train_padded, y_train, X_test_padded, y_test):
    for _ in range(num_models):
        model_clone = clone_model(best_model)
        model_clone.set_weights(best_model.get_weights())
        model_clone.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mape', msle, rmse])
        model_clone.fit(X_train_padded, y_train, batch_size=32, epochs=100, validation_data=(X_test_padded, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
        ensemble_predictions.append(model_clone.predict(X_test_padded).flatten())

    ensemble_predictions = np.array(ensemble_predictions)
    ensemble_mean = np.mean(ensemble_predictions, axis=0)
    ensemble_mape = np.mean(np.abs(y_test - ensemble_mean) / y_test) * 100
    print('Ensemble Mean MAPE:', ensemble_mape)
    return ensemble_mean, ensemble_mape
