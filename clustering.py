import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from config import CLUSTERING_CONFIG

class TextClustering:
    def __init__(self, method="kmeans"):
        self.method = method
        self.model = None
        self.labels_ = None
    
    def fit_kmeans(self, vectors, n_clusters=8):
        """K-means кластеризация"""
        self.model = KMeans(
            n_clusters=n_clusters,
            **CLUSTERING_CONFIG["kmeans"]
        )
        self.labels_ = self.model.fit_predict(vectors)
        return self.labels_
    
    def fit_minibatch_kmeans(self, vectors, n_clusters=8, batch_size=1000):
        """MiniBatch K-means"""
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            **CLUSTERING_CONFIG["kmeans"]
        )
        self.labels_ = self.model.fit_predict(vectors)
        return self.labels_
    
    def fit_spherical_kmeans(self, vectors, n_clusters=8):
        """Spherical K-means (K-means с косинусной метрикой)"""
        # Нормализуем векторы для косинусной метрики
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.model = KMeans(
            n_clusters=n_clusters,
            **CLUSTERING_CONFIG["kmeans"]
        )
        self.labels_ = self.model.fit_predict(vectors_norm)
        return self.labels_
    
    def fit_dbscan(self, vectors, eps=0.5, min_samples=5):
        """DBSCAN кластеризация"""
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine'
        )
        self.labels_ = self.model.fit_predict(vectors)
        return self.labels_
    
    def fit_hdbscan(self, vectors, min_cluster_size=10):
        """HDBSCAN кластеризация"""
        self.model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='cosine'
        )
        self.labels_ = self.model.fit_predict(vectors)
        return self.labels_
    
    def fit_agglomerative(self, vectors, n_clusters=8, linkage='ward'):
        """Иерархическая кластеризация"""
        metric = 'euclidean' if linkage == 'ward' else 'cosine'
        
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric
        )
        self.labels_ = self.model.fit_predict(vectors)
        return self.labels_
    
    def fit_gmm(self, vectors, n_components=8):
        """Гауссовские смеси"""
        self.model = GaussianMixture(
            n_components=n_components,
            random_state=42
        )
        self.model.fit(vectors)
        self.labels_ = self.model.predict(vectors)
        return self.labels_
    
    def fit_lda(self, tokenized_texts, n_topics=8, passes=10):
        """LDA тематическое моделирование"""
        dictionary = Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_texts]
        
        self.model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=passes,
            random_state=42
        )
        
        # Назначение тем документам
        topic_assignments = []
        for doc in corpus:
            topic_probs = self.model.get_document_topics(doc)
            if topic_probs:
                topic_assignments.append(max(topic_probs, key=lambda x: x[1])[0])
            else:
                topic_assignments.append(-1)
        
        self.labels_ = np.array(topic_assignments)
        return self.labels_
    
    def fit_spectral(self, vectors, n_clusters=8):
        """Спектральная кластеризация"""
        # Создание матрицы сходства
        similarity_matrix = cosine_similarity(vectors)
        
        self.model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        self.labels_ = self.model.fit_predict(similarity_matrix)
        return self.labels_
    
    def fit(self, vectors, **kwargs):
        """Общий метод для кластеризации"""
        method_name = f"fit_{self.method}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(vectors, **kwargs)
        else:
            raise ValueError(f"Method {self.method} not supported")