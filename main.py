

data = load_and_clean_data(file_path)

data =  remove_outliers(data, 'Selling Price($)', multiplier=1.5)

data, scaler = cale_data(data, 'Selling Price($)')
data['preprocessed_description'] = data['About Product'].apply(preprocess)

