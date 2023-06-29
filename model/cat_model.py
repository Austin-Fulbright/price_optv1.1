
import pandas as pd

def cat_model(data):
    data['Categories'].fillna('Unknown', inplace=True)
    data['Categories'] = data['Categories'].apply(lambda x: x.split('|'))
    categories_encoded = pd.get_dummies(data['Categories'].apply(pd.Series).stack()).sum(level=0)

    # Define the build_model_categories function here
    def build_model_categories(hp):
    # Your code here
        pass

# Define the X_train_cat and X_test_cat variables here
    X_train_cat = # Your code here
    X_test_cat = # Your code here

    tuner_cat = RandomSearch(build_model_categories, objective='val_loss', max_trials=5, executions_per_trial=3, directory='my_dir', project_name='helloworld')
    tuner_cat.search(X_train_cat, y_train, epochs=5, validation_data=(X_test_cat, y_test))
    best_model_cat = tuner_cat.get_best_models(num_models=1)[0]

def build_model_combined(hp, max_sequence_length, tokenizer):
    input_text = Input(shape=(max_sequence_length,), name='text_input')
    input_cat = Input(shape=(len(categories_encoded.columns),), name='cat_input')

    embedding = Embedding(len(tokenizer.word_index) + 1, hp.Int('embedding_dim', min_value=50, max_value=200, step=50))(input_text)
    lstm_1 = LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=64), return_sequences=True, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5))(embedding)
    lstm_2 = LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=64), kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5))(lstm_1)

    concatenated = concatenate([lstm_2, input_cat])

    dense = Dense(64, activation='relu')(concatenated)
    dropout = Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1))(dense)
    output = Dense(1, activation='linear')(dropout)

    model = Model(inputs=[input_text, input_cat], outputs=output)
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=0.001, max_value=0.1, step=0.001))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mape', msle, rmse])
    return model

tuner_combined = RandomSearch(build_model_combined, objective='val_loss', max_trials=5, executions_per_trial=3, directory='my_dir', project_name='helloworld')
tuner_combined.search([X_train_padded, X_train_cat], y_train, epochs=5, validation_data=([X_test_padded, X_test_cat], y_test))

best_model_combined = tuner_combined.get_best_models(num_models=1)[0]
loss_combined = best_model_combined.evaluate([X_test_padded, X_test_cat], y_test)
print('Combined Model Loss:', loss_combined)

# Make predictions with the combined model
y_pred_combined = best_model_combined.predict([X_test_padded, X_test_cat]).flatten()

# Calculate MAPE for the combined model
mape_combined = np.mean(np.abs(y_test - y_pred_combined) / y_test) * 100
print('Combined Model MAPE:', mape_combined)
