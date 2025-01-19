import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns

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

def describe_cluster_group(df, cluster):
    summary = df.groupby(cluster).agg({
        'TotalOrderValue': ['mean', 'median', 'std', 'min', 'max'],
        'AverageOrderValue': ['mean', 'median', 'std', 'min', 'max'],
        'Quantity': ['mean', 'median', 'std', 'min', 'max']
    })

    print(f"Podsumowanie statystyk dla {cluster}:")
    print(summary)
    summary.to_csv(f'{cluster}.csv', index=False)

def prepare_and_execute_clustering(df):
    df = df[['TotalOrderValue', 'AverageOrderValue', 'Quantity']]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_KMeans'] = kmeans.fit_predict(data_scaled)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['Cluster_DBSCAN'] = dbscan.fit_predict(data_scaled)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df['TotalOrderValue'], y=df['AverageOrderValue'], hue=df['Cluster_KMeans'], palette='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('TotalOrderValue')
    plt.ylabel('AverageOrderValue')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df['TotalOrderValue'], y=df['AverageOrderValue'], hue=df['Cluster_DBSCAN'], palette='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('TotalOrderValue')
    plt.ylabel('AverageOrderValue')

    plt.tight_layout()
    plt.show()

    df.to_csv('klastry.csv', index=False)

    describe_cluster_group(df, "Cluster_KMeans")
    describe_cluster_group(df, "Cluster_DBSCAN")

