import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

class ClusterEvaluator:
    def __init__(self, vectors, labels, true_labels=None):
        self.vectors = vectors
        self.labels = labels
        self.true_labels = true_labels
    
    def calculate_internal_metrics(self):
        """Вычисление внутренних метрик"""
        if len(np.unique(self.labels)) < 2:
            return {
                "silhouette": -1,
                "calinski_harabasz": -1,
                "davies_bouldin": float('inf')
            }
        
        metrics = {}
        
        # Silhouette Score
        try:
            metrics["silhouette"] = silhouette_score(self.vectors, self.labels)
        except:
            metrics["silhouette"] = -1
        
        # Calinski-Harabasz Index
        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(self.vectors, self.labels)
        except:
            metrics["calinski_harabasz"] = -1
        
        # Davies-Bouldin Index
        try:
            metrics["davies_bouldin"] = davies_bouldin_score(self.vectors, self.labels)
        except:
            metrics["davies_bouldin"] = float('inf')
        
        return metrics
    
    def calculate_external_metrics(self):
        """Вычисление внешних метрик (если есть разметка)"""
        if self.true_labels is None:
            return {}
        
        metrics = {}
        metrics["adjusted_rand"] = adjusted_rand_score(self.true_labels, self.labels)
        metrics["normalized_mutual_info"] = normalized_mutual_info_score(self.true_labels, self.labels)
        metrics["v_measure"] = v_measure_score(self.true_labels, self.labels)
        
        return metrics
    
    def plot_metrics_comparison(self, results_dict, save_path=None):
        """Визуализация сравнения метрик"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Silhouette Score
        methods = list(results_dict.keys())
        silhouette_scores = [results_dict[m]["internal"]["silhouette"] for m in methods]
        
        axes[0, 0].bar(methods, silhouette_scores)
        axes[0, 0].set_title("Silhouette Score")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz
        ch_scores = [results_dict[m]["internal"]["calinski_harabasz"] for m in methods]
        axes[0, 1].bar(methods, ch_scores)
        axes[0, 1].set_title("Calinski-Harabasz Index")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin
        db_scores = [results_dict[m]["internal"]["davies_bouldin"] for m in methods]
        axes[1, 0].bar(methods, db_scores)
        axes[1, 0].set_title("Davies-Bouldin Index (lower is better)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Внешние метрики (если есть)
        if self.true_labels is not None:
            nmi_scores = [results_dict[m]["external"]["normalized_mutual_info"] for m in methods]
            axes[1, 1].bar(methods, nmi_scores)
            axes[1, 1].set_title("Normalized Mutual Information")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()