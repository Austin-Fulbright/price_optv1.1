

import numpy as np

def search(tuner, X_train_padded, y_train, X_test_padded, y_test):
    tuner.search(X_train_padded, y_train,
             epochs=5,
             validation_data=(X_test_padded, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]
    loss = best_model.evaluate(X_test_padded, y_test)
    print('Best Model Loss:', loss)

    y_test = y_test.values.flatten()
    # Calculate MAPE
    y_pred = best_model.predict(X_test_padded).flatten() # Flatten the output
    mape = np.mean(np.abs(y_test - y_pred) / y_test) * 100
    print('Best Model MAPE:', mape)
    return best_model, loss, mape
