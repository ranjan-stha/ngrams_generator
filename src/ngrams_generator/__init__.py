from .ngrams_generator import NGramsGenerator

def download_pkgs():
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')