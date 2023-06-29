from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.initializers import GlorotUniform
from keras.optimizers import Adam
from utils.metrics import msle, rmse

def build_model(hp):
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
