import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE

class ClusterVisualizer:
    def __init__(self, vectors, labels, texts=None):
        self.vectors = vectors
        self.labels = labels
        self.texts = texts
    
    def reduce_dimensionality(self, method='umap', n_components=2):
        """Уменьшение размерности для визуализации"""
        if method == 'umap':
            reducer = UMAP(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'umap' or 'tsne'")
        
        return reducer.fit_transform(self.vectors)
    
    def plot_clusters(self, method='umap', figsize=(12, 8), save_path=None):
        """Визуализация кластеров в 2D пространстве"""
        # Уменьшение размерности
        embeddings_2d = self.reduce_dimensionality(method=method)
        
        # Создание графика
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=self.labels,
            cmap='Spectral',
            alpha=0.7,
            s=10
        )
        
        plt.colorbar(scatter)
        plt.title(f'Clusters Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cluster_sizes(self, save_path=None):
        """Визуализация распределения размеров кластеров"""
        unique, counts = np.unique(self.labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar([str(u) for u in unique], counts)
        plt.title('Cluster Sizes Distribution')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def get_top_terms_per_cluster(vectorizer, clustering_model, n_terms=10):
    """Получение топ-термов для каждого кластера (для TF-IDF)"""
    if not hasattr(vectorizer, 'get_feature_names_out'):
        return {}
    
    feature_names = vectorizer.get_feature_names_out()
    cluster_centers = clustering_model.cluster_centers_
    
    top_terms = {}
    for i, center in enumerate(cluster_centers):
        top_indices = center.argsort()[-n_terms:][::-1]
        top_terms[i] = [feature_names[idx] for idx in top_indices]
    
    return top_terms