import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec, FastText, KeyedVectors
from config import MODELS_DIR

class TextVectorizer:
    def __init__(self, method="tfidf"):
        self.method = method
        self.vectorizer = None
        self.embedding_model = None
    
    def load_embedding_model(self, model_path, model_type):
        """Загрузка предобученных эмбеддингов"""
        if model_type == "word2vec":
            self.embedding_model = Word2Vec.load(str(model_path))
        elif model_type == "fasttext":
            self.embedding_model = FastText.load(str(model_path))
        elif model_type == "glove":
            self.embedding_model = KeyedVectors.load(str(model_path))
    
    def fit_tfidf(self, tokenized_texts):
        """Обучение TF-IDF векторизатора"""
        texts = [" ".join(tokens) for tokens in tokenized_texts]
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.8
        )
        return self.vectorizer.fit_transform(texts)
    
    def fit_bm25(self, tokenized_texts):
        """Создание BM25 модели"""
        self.vectorizer = BM25Okapi(tokenized_texts)
        return np.array([self.vectorizer.get_scores(doc) for doc in tokenized_texts])
    
    def document_to_embedding(self, tokens):
        """Преобразование документа в эмбеддинг (усреднение токенов)"""
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")
        
        vectors = []
        for token in tokens:
            if token in self.embedding_model.wv:
                vectors.append(self.embedding_model.wv[token])
        
        if len(vectors) == 0:
            return np.zeros(self.embedding_model.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def fit_embeddings(self, tokenized_texts, model_type, model_path):
        """Векторизация с использованием эмбеддингов"""
        self.load_embedding_model(model_path, model_type)
        
        embeddings = []
        for tokens in tokenized_texts:
            doc_vector = self.document_to_embedding(tokens)
            embeddings.append(doc_vector)
        
        return np.array(embeddings)
    
    def normalize_vectors(self, vectors):
        """L2 нормализация векторов"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Избегаем деления на ноль
        return vectors / norms