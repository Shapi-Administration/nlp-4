import importlib
import sys

# Список пакетов для проверки
packages_to_check = [
    "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn",
    "nltk", "regex", "pymorphy2", "gensim", "rank_bm25", "hdbscan",
    "umap", "plotly", "streamlit", "joblib", "threadpoolctl", "tqdm"
]

print("Проверка установленных пакетов:")
print("-" * 50)

missing_packages = []
for package in packages_to_check:
    try:
        # Для scikit-learn используем другое имя
        if package == "sklearn":
            import sklearn
            version = sklearn.__version__
            print(f"✓ scikit-learn {version}")
        else:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'версия неизвестна')
            print(f"✓ {package} {version}")
    except ImportError:
        print(f"✗ {package} - НЕ УСТАНОВЛЕН")
        missing_packages.append(package)

print("-" * 50)
print(f"Установлено: {len(packages_to_check) - len(missing_packages)}/{len(packages_to_check)}")
if missing_packages:
    print(f"Отсутствуют: {', '.join(missing_packages)}")