import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))))
print(f"Project Root Folder: {PROJECT_ROOT}")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_SEARCH_LIMIT = 5
DOCUMENT_PREVIEW_LIMIT = 100
DEFAULT_WEIGHT = 0.5
DEFAULT_K = 60
