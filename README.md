# E-commerce Price Prediction

This project contains a sequence of scripts for predicting the price of a product based on its description using machine learning techniques. 

## Project Structure

The project has the following structure:

- `data`: Contains the data file `marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv`.
- `preprocessor`: Contains scripts for preprocessing the data.
  - `data_cleaning.py`: Script for initial data cleaning.
  - `text_preprocessing.py`: Script for preprocessing text data.
  - `scaling.py`: Script for scaling numerical data.
- `utils`: Contains scripts with utility functions.
  - `metrics.py`: Contains metrics like MSLE, RMSE.
  - `outliers.py`: Contains functions to handle outliers.
- `model`: Contains scripts related to model training, tuning, evaluation, and saving.
  - `base_model.py`: Contains the base LSTM model.
  - `model_saver.py`: Contains functions to save and load models.
  - `tuner.py`: Contains hyperparameter tuning with Keras Tuner.
  - `evaluation.py`: Contains functions to evaluate the model.
  - `ensemble.py`: Contains the ensemble learning method.
- `main.py`: The main script that calls all other scripts and runs the entire process.
- `test.py`: Script for testing.

## How to Run

You can run the main script using the following command:
python main.py
# price_optv1.1
