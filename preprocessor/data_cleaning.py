import pandas as pd

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['About Product', 'Selling Price'])
    data.rename(columns={'Uniq Id': 'Id', 'Shipping Weight': 'Shipping Weight(Pounds)', 'Selling Price': 'Selling Price($)'}, inplace=True)
    data['Selling Price($)'] = data['Selling Price($)'].str.replace('$', '').str.replace(' ', '').str.split('.').str[0] + '.'
    data = data[~data['Selling Price($)'].str.contains('[a-zA-Z]', na=False)]
    data['Selling Price($)'] = data['Selling Price($)'].str.replace(',', '').astype(float)
    data['Selling Price($)'] = data['Selling Price($)'].apply(lambda x: "{:.2f}".format(x)).astype(float)
    return data
