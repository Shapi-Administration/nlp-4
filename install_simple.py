import subprocess
import sys
import os

def install_package(package):
    try:
        # Используем флаг --prefer-binary для установки предварительно скомпилированных пакетов
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--prefer-binary", package])
        print(f"✓ Успешно установлен: {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Ошибка установки: {package}")
        return False

# Список пакетов с совместимыми версиями для Windows
packages = [
    "numpy",
    "pandas", 
    "scikit-learn==1.3.2",
    "scipy==1.10.1",
    "spacy==3.7.2",
    "pymorphy2==0.9.1",
    "pymorphy2-dicts-ru",
    "nltk==3.8.1",
    "regex==2023.10.3",
    "gensim==4.3.2",
    "rank-bm25==0.2.2",
    "hdbscan==0.8.29",
    "matplotlib==3.7.2",
    "seaborn==0.12.2",
    "plotly==5.17.0",
    "umap-learn==0.5.5",
    "streamlit==1.28.1",
    "threadpoolctl==3.2.0",
    "joblib==1.3.2"
]

print("Установка предварительно скомпилированных пакетов...")
success_count = 0

for package in packages:
    if install_package(package):
        success_count += 1

print(f"\nУстановлено {success_count} из {len(packages)} пакетов")