import pandas as pd

def remove_outliers(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    df_out = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_out
