import pandas as pd

def simplyfy_categories(data):
    # Fill missing categories with 'Unknown'
    data['Categories'].fillna('Unknown', inplace=True)

    # Simplify categories - we will only use the top-level category
    data['main_category'] = data['Categories'].apply(lambda x: x.split('|')[0].strip())

    # Apply one-hot encoding
    categories_encoded = pd.get_dummies(data['main_category'], prefix='category')
    data = pd.concat([data, categories_encoded], axis=1)
    return data
