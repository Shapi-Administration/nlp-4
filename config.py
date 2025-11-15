import os
from pathlib import Path

# Пути к данным и моделям
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Создание директорий
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Настройки предобработки
PREPROCESSING_CONFIG = {
    "lemmatization": True,
    "remove_stopwords": True,
    "min_token_length": 2,
    "max_token_length": 25
}

# Настройки кластеризации
CLUSTERING_CONFIG = {
    "kmeans": {"n_init": 10, "random_state": 42},
    "dbscan": {"min_samples": 5},
    "hdbscan": {"min_cluster_size": 10},
    "agglomerative": {"linkage": "ward"}
}