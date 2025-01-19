import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df

def find_and_delete_nulls(df):
    print("Brakujące dane przed usunięciem:")
    print(df.isnull().sum())

    df = df.dropna()

    print("\nBrakujące dane po usunięciu:")
    print(df.isnull().sum())

    return df

def find_and_delete_nagative_quantity(df):
    negative_values = df['Quantity'] < 0
    df = df[~negative_values]
    print(f"liczba anomali: {df[negative_values].shape[0]}")

    return df

def show_basic_stats(df):
    print(df.describe())
    print(f"Mediana ilosci: {df['Quantity'].median()}")

def create_totalordervalue_and_averageordervalue(df):
    df['TotalOrderValue'] = df['Quantity'] * df['UnitPrice']
    df['AverageOrderValue'] = df.groupby('StockCode')['TotalOrderValue'].transform('mean')
    print(df)
    return df