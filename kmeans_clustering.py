import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def perform_kmeans_clustering(data, n_clusters=5, labels=None, random_state=42):
    """
    对数据进行K-means聚类分析
    
    参数:
    data: numpy数组，形状为(n_samples, n_features)
    n_clusters: 聚类数量
    labels: 真实标签（如果有）
    random_state: 随机种子
    
    返回:
    cluster_labels: 聚类标签
    cluster_centers: 聚类中心
    """
    # 确保数据是二维的
    if len(data.shape) > 2:
        n_samples = data.shape[0] * data.shape[1]
        n_features_total = data.shape[2]
        X_for_clustering = data.reshape(n_samples, n_features_total)
    else:
        X_for_clustering = data
    
    print(f"数据形状: {X_for_clustering.shape}")
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_for_clustering)
    cluster_centers = kmeans.cluster_centers_
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_for_clustering, cluster_labels)
    print(f"聚类数量 K={n_clusters}, 轮廓系数={silhouette_avg:.4f}")
    
    # 计算每个聚类的样本数量
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("各聚类样本数量:")
    for cluster_id, count in cluster_counts.items():
        print(f"聚类 {cluster_id}: {count} 个样本")
    
    # 使用PCA将数据降维到2D进行可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_for_clustering)
    centers_pca = pca.transform(cluster_centers)
    
    # 可视化聚类结果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='聚类中心')
    plt.colorbar(scatter, label='聚类')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.title(f'K-means 聚类结果 (K={n_clusters})')
    plt.legend()
    plt.grid(True)
    plt.savefig('kmeans_clustering_results.png', dpi=300)
    plt.show()
    
    # 如果有标签信息，分析聚类与标签的关系
    if labels is not None:
        cluster_label_distribution = pd.crosstab(
            pd.Series(cluster_labels, name='聚类'),
            pd.Series(labels, name='真实标签')
        )
        print("\n聚类-标签分布:")
        print(cluster_label_distribution)
        
        # 可视化每个聚类中的标签分布
        cluster_label_distribution.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.xlabel('聚类')
        plt.ylabel('样本数量')
        plt.title('各聚类中真实标签的分布')
        plt.legend(title='真实标签')
        plt.savefig('cluster_label_distribution.png', dpi=300)
        plt.show()
    
    return cluster_labels, cluster_centers

def find_optimal_k(data, k_range=range(2, 11), random_state=42):
    """
    寻找最佳的K值（聚类数量）
    
    参数:
    data: numpy数组，形状为(n_samples, n_features)
    k_range: 要尝试的K值范围
    random_state: 随机种子
    
    返回:
    best_k: 最佳K值
    """
    # 确保数据是二维的
    if len(data.shape) > 2:
        n_samples = data.shape[0]
        n_features_total = data.shape[1] * data.shape[2]
        X_for_clustering = data.reshape(n_samples, n_features_total)
    else:
        X_for_clustering = data
    
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_for_clustering)
        silhouette_avg = silhouette_score(X_for_clustering, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"K={k}, 轮廓系数={silhouette_avg:.4f}")
    
    # 绘制不同K值的轮廓系数
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'o-')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('轮廓系数')
    plt.title('不同K值的轮廓系数')
    plt.grid(True)
    plt.savefig('kmeans_silhouette_scores.png', dpi=300)
    plt.show()
    
    # 选择最佳K值
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"最佳K值: {best_k}")
    
    return best_k