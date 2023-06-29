from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.initializers import GlorotUniform
from keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from utils.metrics import msle, rmse

def build_model(hp):
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, hp.Int('embedding_dim', min_value=50, max_value=200, step=50), input_length=max_sequence_length))
    model.add(LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=64), return_sequences=True, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
    model.add(LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=64), kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)))
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=0.001, max_value=0.1, step=0.01))
    model.compile(loss=msle, optimizer=optimizer, metrics=[rmse])
    return model
