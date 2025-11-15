import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import traceback
from datetime import datetime

# Импорты для машинного обучения
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score, fowlkes_mallows_score
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# Импорты для NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Импорты для дополнительных методов
from rank_bm25 import BM25Okapi
import hdbscan
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer
import gensim
from gensim.models import Word2Vec, FastText

# Загрузка данных NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class AdvancedTextProcessor:
    """Продвинутый процессор текста с различными методами предобработки"""
    
    def __init__(self):
        try:
            self.stemmer = SnowballStemmer("russian")
            self.stop_words = set(stopwords.words('russian'))
        except:
            # Резервный список стоп-слов
            self.stemmer = None
            self.stop_words = {
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 
                'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 
                'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 
                'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 
                'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 
                'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 
                'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 
                'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 
                'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 
                'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 
                'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 
                'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 
                'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 
                'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 
                'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 
                'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 
                'более', 'всегда', 'конечно', 'всю', 'между'
            }
    
    def clean_text(self, text):
        """Очистка текста"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^а-яёa-z\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def process_word(self, word, use_stemming=True):
        """Обработка одного слова"""
        if len(word) > 2 and word not in self.stop_words:
            if use_stemming and self.stemmer:
                try:
                    return self.stemmer.stem(word)
                except:
                    return word
            return word
        return None
    
    def tokenize(self, text, use_stemming=True):
        """Токенизация текста"""
        text = self.clean_text(text)
        if not text:
            return []
        
        tokens = text.split()
        processed_tokens = []
        
        for token in tokens:
            processed_word = self.process_word(token, use_stemming)
            if processed_word:
                processed_tokens.append(processed_word)
        
        return processed_tokens
    
    def preprocess_texts(self, texts, use_stemming=True):
        """Предобработка списка текстов"""
        processed_texts = []
        for text in texts:
            tokens = self.tokenize(text, use_stemming)
            processed_texts.append(tokens)
        return processed_texts

class AdvancedVectorizer:
    """Продвинутый класс для векторизации текстов с различными методами"""
    
    def __init__(self):
        self.vectorizer = None
        self.bm25 = None
        self.sentence_model = None
        self.word2vec_model = None
        self.fasttext_model = None
    
    def fit_tfidf(self, tokenized_texts, max_features=5000, ngram_range=(1, 2)):
        """TF-IDF векторизация"""
        texts = [' '.join(tokens) for tokens in tokenized_texts]
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=ngram_range
        )
        return self.vectorizer.fit_transform(texts).toarray()
    
    def fit_count(self, tokenized_texts, max_features=5000, ngram_range=(1, 2)):
        """Count векторизация"""
        texts = [' '.join(tokens) for tokens in tokenized_texts]
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=ngram_range
        )
        return self.vectorizer.fit_transform(texts).toarray()
    
    def fit_bm25(self, tokenized_texts):
        """BM25 векторизация"""
        self.bm25 = BM25Okapi(tokenized_texts)
        vectors = []
        for doc in tokenized_texts:
            vectors.append(self.bm25.get_scores(doc))
        return np.array(vectors)
    
    def fit_sentence_transformers(self, texts, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """Векторизация с помощью Sentence Transformers"""
        try:
            self.sentence_model = SentenceTransformer(model_name)
            return self.sentence_model.encode(texts)
        except Exception as e:
            st.warning(f"Ошибка Sentence Transformers: {e}")
            # Возвращаем TF-IDF как запасной вариант
            return self.fit_tfidf([[' '.join(text.split()[:10])] for text in texts])
    
    def fit_word2vec(self, tokenized_texts, vector_size=100, window=5, min_count=2):
        """Word2Vec векторизация"""
        try:
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=4
            )
            
            # Создаем векторы документов как среднее векторов слов
            doc_vectors = []
            for tokens in tokenized_texts:
                vectors = []
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        vectors.append(self.word2vec_model.wv[token])
                if vectors:
                    doc_vectors.append(np.mean(vectors, axis=0))
                else:
                    doc_vectors.append(np.zeros(vector_size))
            
            return np.array(doc_vectors)
        except Exception as e:
            st.warning(f"Ошибка Word2Vec: {e}")
            return self.fit_tfidf(tokenized_texts)
    
    def fit_fasttext(self, tokenized_texts, vector_size=100, window=5, min_count=2):
        """FastText векторизация"""
        try:
            self.fasttext_model = FastText(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=4
            )
            
            # Создаем векторы документов как среднее векторов слов
            doc_vectors = []
            for tokens in tokenized_texts:
                vectors = []
                for token in tokens:
                    if token in self.fasttext_model.wv:
                        vectors.append(self.fasttext_model.wv[token])
                if vectors:
                    doc_vectors.append(np.mean(vectors, axis=0))
                else:
                    doc_vectors.append(np.zeros(vector_size))
            
            return np.array(doc_vectors)
        except Exception as e:
            st.warning(f"Ошибка FastText: {e}")
            return self.fit_tfidf(tokenized_texts)
    
    def fit_doc2vec(self, tokenized_texts, vector_size=100, window=5, min_count=2):
        """Doc2Vec-like векторизация (упрощенная)"""
        # Используем Word2Vec и усредняем
        return self.fit_word2vec(tokenized_texts, vector_size, window, min_count)

class AdvancedClusteringMethods:
    """Класс с расширенными методами кластеризации"""
    
    @staticmethod
    def kmeans(vectors, n_clusters=8, random_state=42):
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(vectors)
        return labels, model
    
    @staticmethod
    def dbscan(vectors, eps=0.5, min_samples=5):
        vectors_norm = normalize(vectors)
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = model.fit_predict(vectors_norm)
        return labels, model
    
    @staticmethod
    def hierarchical(vectors, n_clusters=8, linkage='ward'):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(vectors)
        return labels, model
    
    @staticmethod
    def gaussian_mixture(vectors, n_components=8, random_state=42):
        model = GaussianMixture(n_components=n_components, random_state=random_state)
        labels = model.fit_predict(vectors)
        return labels, model
    
    @staticmethod
    def spectral(vectors, n_clusters=8, random_state=42):
        model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, 
                                 affinity='nearest_neighbors', n_neighbors=10)
        labels = model.fit_predict(vectors)
        return labels, model
    
    @staticmethod
    def hdbscan_method(vectors, min_cluster_size=5):
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = model.fit_predict(vectors)
        return labels, model
    
    @staticmethod
    def birch(vectors, n_clusters=8, threshold=0.5):
        model = Birch(n_clusters=n_clusters, threshold=threshold)
        labels = model.fit_predict(vectors)
        return labels, model

class MetricsCalculator:
    """Класс для расчета всех метрик кластеризации"""
    
    @staticmethod
    def calculate_all_metrics(vectors, labels, true_labels=None):
        """Расчет всех метрик кластеризации"""
        metrics = {}
        
        n_clusters = len(set(labels))
        n_samples = len(labels)
        
        # Основные метрики
        if n_clusters > 1 and n_clusters < n_samples:
            try:
                metrics['Silhouette Score'] = silhouette_score(vectors, labels)
            except:
                metrics['Silhouette Score'] = np.nan
            
            try:
                metrics['Calinski-Harabasz'] = calinski_harabasz_score(vectors, labels)
            except:
                metrics['Calinski-Harabasz'] = np.nan
            
            try:
                metrics['Davies-Bouldin'] = davies_bouldin_score(vectors, labels)
            except:
                metrics['Davies-Bouldin'] = np.nan
        
        # Метрики, требующие истинные метки
        if true_labels is not None and len(true_labels) == len(labels):
            try:
                metrics['Adjusted Rand Index'] = adjusted_rand_score(true_labels, labels)
            except:
                metrics['Adjusted Rand Index'] = np.nan
            
            try:
                metrics['Adjusted Mutual Info'] = adjusted_mutual_info_score(true_labels, labels)
            except:
                metrics['Adjusted Mutual Info'] = np.nan
            
            try:
                metrics['Homogeneity'] = homogeneity_score(true_labels, labels)
            except:
                metrics['Homogeneity'] = np.nan
            
            try:
                metrics['Completeness'] = completeness_score(true_labels, labels)
            except:
                metrics['Completeness'] = np.nan
            
            try:
                metrics['V-measure'] = v_measure_score(true_labels, labels)
            except:
                metrics['V-measure'] = np.nan
            
            try:
                metrics['Fowlkes-Mallows'] = fowlkes_mallows_score(true_labels, labels)
            except:
                metrics['Fowlkes-Mallows'] = np.nan
        
        # Статистические метрики
        metrics['Number of Clusters'] = n_clusters
        metrics['Number of Samples'] = n_samples
        metrics['Noise Points'] = sum(labels == -1) if -1 in labels else 0
        
        # Размеры кластеров
        cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
        if cluster_sizes:
            metrics['Largest Cluster'] = max(cluster_sizes)
            metrics['Smallest Cluster'] = min(cluster_sizes)
            metrics['Avg Cluster Size'] = np.mean(cluster_sizes)
            metrics['Cluster Size Std'] = np.std(cluster_sizes)
        
        return metrics

class AdvancedVisualization:
    """Продвинутый класс для визуализации результатов"""
    
    @staticmethod
    def plot_clusters_plotly(vectors, labels, method='PCA', title=None):
        """Визуализация кластеров с использованием Plotly"""
        
        if method == 'PCA':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(vectors)
            title = title or 'Визуализация кластеров (PCA)'
            explained_var = reducer.explained_variance_ratio_.sum()
            title += f' (Объясненная дисперсия: {explained_var:.2%})'
            
        elif method == 'TSNE':
            reducer = TSNE(n_components=2, random_state=42, 
                          perplexity=min(30, len(vectors)-1))
            embeddings_2d = reducer.fit_transform(vectors)
            title = title or 'Визуализация кластеров (t-SNE)'
            
        elif method == 'UMAP':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(vectors)
            title = title or 'Визуализация кластеров (UMAP)'
            
        else:  # Truncated SVD
            reducer = TruncatedSVD(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(vectors)
            title = title or 'Визуализация кластеров (Truncated SVD)'
        
        df_plot = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': labels,
            'cluster_label': [f'Кластер {l}' if l != -1 else 'Шум' for l in labels]
        })
        
        fig = px.scatter(
            df_plot, x='x', y='y', color='cluster_label',
            title=title,
            hover_data={'cluster': True},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(metrics_df):
        """Визуализация сравнения метрик"""
        metrics_to_plot = ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        if not available_metrics:
            return None
        
        fig = go.Figure()
        
        for metric in available_metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Сравнение метрик качества кластеризации",
            xaxis_title="Метод кластеризации",
            yaxis_title="Значение метрики",
            barmode='group',
            height=500
        )
        
        return fig

def find_optimal_eps(vectors, min_samples=5):
    """Поиск оптимального EPS для DBSCAN"""
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(vectors)
    distances, indices = neighbors_fit.kneighbors(vectors)
    distances = np.sort(distances[:, -1], axis=0)
    return distances

def main():
    st.set_page_config(
        page_title="Продвинутый анализ кластеризации текстов",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Продвинутый анализ кластеризации текстовых данных")
    st.markdown("---")
    
    # Инициализация классов
    processor = AdvancedTextProcessor()
    vectorizer = AdvancedVectorizer()
    clustering = AdvancedClusteringMethods()
    metrics_calc = MetricsCalculator()
    viz = AdvancedVisualization()
    
    # Сайдбар с настройками
    st.sidebar.header("⚙️ Настройки эксперимента")
    
    # Загрузка данных
    uploaded_file = st.sidebar.file_uploader(
        "📁 Загрузите данные",
        type=['csv', 'txt'],
        help="CSV с колонкой текста или TXT с документами построчно"
    )
    
    # Демо данные
    use_demo = st.sidebar.checkbox("Использовать демо данные", value=True)
    
    if uploaded_file is None and not use_demo:
        st.info("👆 Загрузите файл или используйте демо данные для начала анализа")
        return
    
    # Загрузка и подготовка данных
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Загружен CSV файл с {len(df)} строками")
                
                # Выбор колонки
                text_cols = [col for col in df.columns if df[col].dtype == 'object']
                if text_cols:
                    text_column = st.sidebar.selectbox("Выберите текстовую колонку", text_cols)
                    texts = df[text_column].dropna().astype(str).tolist()
                    
                    # Выбор колонки с истинными метками (опционально)
                    all_cols = df.columns.tolist()
                    if len(all_cols) > 1:
                        label_col = st.sidebar.selectbox(
                            "Выберите колонку с истинными метками (опционально)", 
                            ['Нет'] + [col for col in all_cols if col != text_column]
                        )
                        if label_col != 'Нет':
                            true_labels = df[label_column].dropna().astype(str).tolist()
                        else:
                            true_labels = None
                    else:
                        true_labels = None
                else:
                    st.error("❌ В файле нет текстовых колонок")
                    return
            else:
                text_data = uploaded_file.read().decode('utf-8')
                texts = [line.strip() for line in text_data.split('\n') if line.strip()]
                st.success(f"✅ Загружено {len(texts)} текстовых строк")
                true_labels = None
        except Exception as e:
            st.error(f"❌ Ошибка загрузки файла: {e}")
            return
    else:
        # Демо данные с истинными метками
        demo_data = [
            ("Машинное обучение и искусственный интеллект сегодня очень популярны", "AI"),
            ("Глубокое обучение использует нейронные сети для решения сложных задач", "AI"),
            ("Обработка естественного языка позволяет компьютерам понимать человеческую речь", "NLP"),
            ("Анализ данных включает статистические методы и визуализацию", "Data Science"),
            ("Визуализация данных помогает понять сложные паттерны и тренды", "Data Science"),
            ("Базы данных хранят структурированную информацию для быстрого доступа", "Databases"),
            ("SQL язык запросов используется для работы с реляционными базами данных", "Databases"),
            ("Веб разработка создает интерфейсы и серверные приложения", "Web"),
            ("Фронтенд разработка отвечает за пользовательский интерфейс", "Web"),
            ("Бэкенд разработка работает с серверной логикой и базами данных", "Web"),
            ("Мобильные приложения разрабатываются для iOS и Android платформ", "Mobile"),
            ("Кроссплатформенная разработка позволяет создавать приложения для разных ОС", "Mobile"),
            ("Облачные вычисления предоставляют ресурсы через интернет", "Cloud"),
            ("Amazon AWS и Microsoft Azure популярные облачные платформы", "Cloud"),
            ("Кибербезопасность защищает информацию от несанкционированного доступа", "Security"),
            ("Шифрование данных обеспечивает конфиденциальность информации", "Security")
        ]
        texts = [item[0] for item in demo_data]
        true_labels = [item[1] for item in demo_data]
        st.info("🔮 Используются демонстрационные данные о технологиях с истинными метками")
    
    # Настройки
    st.sidebar.subheader("🔧 Предобработка текста")
    use_stemming = st.sidebar.checkbox("Использовать стемминг", value=True)
    
    st.sidebar.subheader("📊 Векторизация")
    vectorization_method = st.sidebar.selectbox(
        "Метод векторизации",
        ['TF-IDF', 'BM25', 'Count Vectorizer', 'Sentence Transformers', 'Word2Vec', 'FastText', 'Doc2Vec']
    )
    
    st.sidebar.subheader("🎯 Кластеризация")
    clustering_methods = st.sidebar.multiselect(
        "Алгоритмы кластеризации",
        ['KMeans', 'DBSCAN', 'Hierarchical', 'GaussianMixture', 'Spectral', 'HDBSCAN', 'BIRCH'],
        default=['KMeans', 'DBSCAN', 'Hierarchical']
    )
    
    n_clusters = st.sidebar.slider("Количество кластеров", 2, 15, 5)
    
    st.sidebar.subheader("📈 Визуализация")
    viz_method = st.sidebar.selectbox(
        "Метод визуализации",
        ['PCA', 'TSNE', 'UMAP', 'Truncated SVD']
    )
    
    # Основной интерфейс
    if st.sidebar.button("🚀 Запустить анализ", type="primary"):
        
        with st.spinner("🔄 Выполняется предобработка текстов..."):
            try:
                # Предобработка
                tokenized_texts = processor.preprocess_texts(texts, use_stemming=use_stemming)
                
                st.write(f"📊 Обработано {len(tokenized_texts)} документов")
                
                # Векторизация
                if vectorization_method == 'TF-IDF':
                    vectors = vectorizer.fit_tfidf(tokenized_texts)
                elif vectorization_method == 'BM25':
                    vectors = vectorizer.fit_bm25(tokenized_texts)
                elif vectorization_method == 'Count Vectorizer':
                    vectors = vectorizer.fit_count(tokenized_texts)
                elif vectorization_method == 'Sentence Transformers':
                    vectors = vectorizer.fit_sentence_transformers(texts)
                elif vectorization_method == 'Word2Vec':
                    vectors = vectorizer.fit_word2vec(tokenized_texts)
                elif vectorization_method == 'FastText':
                    vectors = vectorizer.fit_fasttext(tokenized_texts)
                else:  # Doc2Vec
                    vectors = vectorizer.fit_doc2vec(tokenized_texts)
                
                st.success(f"✅ Векторизация завершена. Размерность: {vectors.shape}")
                
                # Сравнение методов кластеризации
                all_results = {}
                all_metrics = {}
                
                for method in clustering_methods:
                    with st.spinner(f"🔄 Выполняется {method} кластеризация..."):
                        try:
                            if method == 'KMeans':
                                labels, model = clustering.kmeans(vectors, n_clusters=n_clusters)
                            elif method == 'DBSCAN':
                                labels, model = clustering.dbscan(vectors)
                            elif method == 'Hierarchical':
                                labels, model = clustering.hierarchical(vectors, n_clusters=n_clusters)
                            elif method == 'GaussianMixture':
                                labels, model = clustering.gaussian_mixture(vectors, n_components=n_clusters)
                            elif method == 'Spectral':
                                labels, model = clustering.spectral(vectors, n_clusters=n_clusters)
                            elif method == 'HDBSCAN':
                                labels, model = clustering.hdbscan_method(vectors)
                            else:  # BIRCH
                                labels, model = clustering.birch(vectors, n_clusters=n_clusters)
                            
                            # Расчет метрик
                            metrics = metrics_calc.calculate_all_metrics(vectors, labels, true_labels)
                            all_results[method] = labels
                            all_metrics[method] = metrics
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка в {method}: {e}")
                
                # Сводная таблица метрик
                st.subheader("📊 Сводная таблица метрик")
                
                metrics_df = pd.DataFrame(all_metrics).T
                
                # Стилизация таблицы
                def highlight_extremes(s):
                    if s.dtype in [np.float64, np.int64]:
                        is_max = s == s.max()
                        is_min = s == s.min()
                        return ['background-color: lightgreen' if max_val else 
                                'background-color: lightcoral' if min_val else '' 
                                for max_val, min_val in zip(is_max, is_min)]
                    return [''] * len(s)
                
                styled_metrics = metrics_df.style.format("{:.3f}").apply(highlight_extremes)
                st.dataframe(styled_metrics, use_container_width=True)
                
                # Визуализация сравнения метрик
                comparison_fig = viz.plot_metrics_comparison(metrics_df)
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Детальный анализ для каждого метода
                for method in clustering_methods:
                    if method in all_results:
                        st.subheader(f"🔍 Детальный анализ: {method}")
                        
                        labels = all_results[method]
                        metrics = all_metrics[method]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📈 Основные метрики:**")
                            for metric_name in ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin']:
                                if metric_name in metrics:
                                    st.metric(metric_name, f"{metrics[metric_name]:.3f}")
                        
                        with col2:
                            st.write("**📊 Статистика кластеров:**")
                            cluster_counts = pd.Series(labels).value_counts().sort_index()
                            st.write(f"Количество кластеров: {metrics['Number of Clusters']}")
                            st.write(f"Точек шума: {metrics['Noise Points']}")
                        
                        # Визуализация
                        fig = viz.plot_clusters_plotly(vectors, labels, method=viz_method, 
                                                     title=f'{method} кластеризация')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Анализ содержимого кластеров
                        st.write("**🔍 Анализ содержимого кластеров:**")
                        
                        results_df = pd.DataFrame({
                            'Текст': texts[:len(labels)],
                            'Токены': [' '.join(tokens) for tokens in tokenized_texts[:len(labels)]],
                            'Кластер': labels
                        })
                        
                        if true_labels is not None:
                            results_df['Истинная_метка'] = true_labels[:len(labels)]
                        
                        for cluster_id in sorted(set(labels)):
                            with st.expander(f"📂 Кластер {cluster_id} ({sum(labels == cluster_id)} документов)"):
                                cluster_data = results_df[results_df['Кластер'] == cluster_id]
                                
                                # Показываем первые 3 документа
                                st.write("**Примеры документов:**")
                                for i, (idx, row) in enumerate(cluster_data.head(3).iterrows()):
                                    st.write(f"{i+1}. {row['Текст']}")
                                
                                # Анализ частых слов
                                all_tokens = []
                                for tokens_str in cluster_data['Токены']:
                                    all_tokens.extend(tokens_str.split())
                                
                                if all_tokens:
                                    common_words = Counter(all_tokens).most_common(8)
                                    st.write("**Топ-8 частых слов:**")
                                    words_df = pd.DataFrame(common_words, columns=['Слово', 'Частота'])
                                    st.dataframe(words_df, use_container_width=True)
                                
                                # Сравнение с истинными метками
                                if true_labels is not None and 'Истинная_метка' in cluster_data.columns:
                                    true_labels_dist = cluster_data['Истинная_метка'].value_counts()
                                    st.write("**Распределение истинных меток:**")
                                    st.dataframe(true_labels_dist, use_container_width=True)
                        
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Ошибка при выполнении анализа: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()