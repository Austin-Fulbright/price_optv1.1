from sklearn.preprocessing import MinMaxScaler

def scale_data(data, feature):
    scaler = MinMaxScaler()
    data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
    return data, scaler
