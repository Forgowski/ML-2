import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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


def find_nagative_quantity(df):
    negative_values = df['Quantity'] < 0
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

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster_KMeans'] = kmeans.fit_predict(data_scaled)

    dbscan = DBSCAN(eps=0.2, min_samples=15)
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

    return df


def tsne(df):
    tsne_df = df[['TotalOrderValue', 'AverageOrderValue', 'Quantity']]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    tsne_result = tsne.fit_transform(tsne_df)

    tsne_data_with_clusters = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
    tsne_data_with_clusters['Cluster_KMeans'] = df['Cluster_KMeans'].values
    tsne_data_with_clusters['Cluster_DBSCAN'] = df['Cluster_DBSCAN'].values

    plt.figure(figsize=(10, 6))
    for cluster_id in tsne_data_with_clusters['Cluster_KMeans'].unique():
        cluster_points = tsne_data_with_clusters[tsne_data_with_clusters['Cluster_KMeans'] == cluster_id]
        plt.scatter(cluster_points['Dimension 1'], cluster_points['Dimension 2'], label=f'Cluster {cluster_id}',
                    alpha=0.7)
    plt.title('t-SNE Visualization with K-Means Clusters', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for cluster_id in tsne_data_with_clusters['Cluster_DBSCAN'].unique():
        cluster_points = tsne_data_with_clusters[tsne_data_with_clusters['Cluster_DBSCAN'] == cluster_id]
        plt.scatter(cluster_points['Dimension 1'], cluster_points['Dimension 2'], label=f'Cluster {cluster_id}',
                    alpha=0.7)
    plt.title('t-SNE Visualization with DBSCAN Clusters', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    cluster_means = df.groupby('Cluster_DBSCAN')[['TotalOrderValue', 'AverageOrderValue', 'Quantity']].mean()
    print(cluster_means)

    cluster_means = df.groupby('Cluster_KMeans')[['TotalOrderValue', 'AverageOrderValue', 'Quantity']].mean()
    print(cluster_means)

    return df


def group_with_the_biggest_amount_of_returns(df):
    returns = df[df['Quantity'] < 0]

    dbscan_returns = returns.groupby('Cluster_DBSCAN')['Quantity'].sum()
    print("Sumaryczna wartość zwrotów w grupach DBSCAN:")
    print(dbscan_returns)

    kmeans_returns = returns.groupby('Cluster_KMeans')['Quantity'].sum()
    print("\nSumaryczna wartość zwrotów w grupach KMeans:")
    print(kmeans_returns)

    max_dbscan_cluster = dbscan_returns.idxmin()
    max_kmeans_cluster = kmeans_returns.idxmin()

    print(
        f"\nGrupa DBSCAN z największą sumaryczną wartością zwrotów: {max_dbscan_cluster} ({dbscan_returns[max_dbscan_cluster]} jednostek zwróconych)")
    print(
        f"Grupa KMeans z największą sumaryczną wartością zwrotów: {max_kmeans_cluster} ({kmeans_returns[max_kmeans_cluster]} jednostek zwróconych)")


def assign_rfm_score(series, reversed=False):
    labels = [0, 1, 2, 3, 4, 5]
    if reversed:
        labels = labels[::-1]
        
    return pd.cut(series,
                  bins=[-np.inf, series.quantile(0.166), series.quantile(0.333), series.quantile(0.5),
                        series.quantile(0.666), series.quantile(0.833), np.inf],
                  labels=labels)


def new_df_for_rfm(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    reference_date = pd.to_datetime('2011-12-31')

    recency = df.groupby('CustomerID')['InvoiceDate'].max()
    recency = (reference_date - recency).dt.days

    frequency = df.groupby('CustomerID')['InvoiceDate'].count()

    monetary = df.groupby('CustomerID')['TotalOrderValue'].sum()

    rfm = pd.DataFrame({
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary
    })

    rfm.to_csv("rfm.csv")

    rfm['R_score'] = assign_rfm_score(rfm['recency'])

    rfm['F_score'] = assign_rfm_score(rfm['frequency'])

    rfm['M_score'] = assign_rfm_score(rfm['monetary'])

    rfm['RFM_score'] = rfm['R_score'].astype(int) + rfm['F_score'].astype(int) + rfm['M_score'].astype(int)
