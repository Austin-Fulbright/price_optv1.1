from preprocessor.data_cleaning import load_and_clean_data
from utils.outliers import remove_outliers
from preprocessor.scaling import scale_data
from preprocessor.text_preprocessing import preprocess
from preprocessor.split_test_train import split_test, tokenizers, padd_x
from model.tuner import tuner
from model.base_model import build_model
from model.evaluation import search
from model.ensemble import create_ensemble

file_path = 'data/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv'
data = load_and_clean_data(file_path)
data =  remove_outliers(data, 'Selling Price($)', multiplier=1.5)
data, scaler = scale_data(data, 'Selling Price($)')
data['preprocessed_description'] = data['About Product'].apply(preprocess)

X_train, X_test, y_train, y_test = split_test(data)
max_sequence_length = 100
tokenizer, X_train_seq, X_test_seq = tokenizers(data, X_train, X_test)
X_train_padded, X_test_padded = padd_x(max_sequence_length, X_train_seq, X_test_seq)


def build_model_func(hp):
    return build_model(tokenizer, max_sequence_length, hp)

tuner_model = tuner(build_model_func)
best_model, loss, mape = search(tuner_model, X_train_padded, y_train, X_test_padded, y_test)


create_ensemble(best_model, X_train_padded, y_train, X_test_padded, y_test)
