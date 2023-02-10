import string
import nltk
import logging

from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer, FrenchStemmer, SpanishStemmer, ArabicStemmer, PortugueseStemmer

from collections import Counter, OrderedDict
from langdetect import detect

from typing import List, Dict

logging.getLogger().setLevel(logging.INFO)

class NGramsGenerator:
    def __init__(self,
        max_ngrams_items: int=10,
        generate_unigrams: bool=True,
        generate_bigrams: bool=True,
        generate_trigrams: bool=True,
        enable_stopwords: bool=True,
        enable_stemming: bool=True,
        enable_case_sensitive: bool=True
    ):
        self.stopwords_ar = stopwords.words("arabic")
        self.stopwords_en = stopwords.words("english")
        self.stopwords_fr = stopwords.words("french")
        self.stopwords_es = stopwords.words("spanish")
        self.stopwords_pt = stopwords.words("portuguese")

        stemmer_ar = ArabicStemmer()
        stemmer_en = EnglishStemmer()
        stemmer_fr = FrenchStemmer()
        stemmer_es = SpanishStemmer()
        stemmer_pt = PortugueseStemmer()

        self.max_ngrams_items = max_ngrams_items
        self.generate_unigrams = generate_unigrams
        self.generate_bigrams = generate_bigrams
        self.generate_trigrams = generate_trigrams
        self.enable_stopwords = enable_stopwords
        self.enable_stemming = enable_stemming
        self.enable_case_sensitive = enable_case_sensitive

        self.allowed_languages = ["ar", "en", "fr", "es", "pt"]

        self.language_mapper = {
            "ar": "arabic",
            "en": "english",
            "fr": "french",
            "es": "spanish",
            "pt": "portuguese"
        }

        language_fn_mapper = {
            "ar": {
                "stopwords": self.stopwords_ar,
                "stemmer": stemmer_ar
            },
            "en": {
                "stopwords": self.stopwords_en,
                "stemmer": stemmer_en
            },
            "fr": {
                "stopwords": self.stopwords_fr,
                "stemmer": stemmer_fr
            },
            "es": {
                "stopwords": self.stopwords_es,
                "stemmer": stemmer_es
            },
            "pt": {
                "stopwords": self.stopwords_pt,
                "stemmer": stemmer_pt
            }
        }

        self.fn_stopwords = lambda tokens, lang: [w for w in tokens if w not in language_fn_mapper[lang]["stopwords"]]
        self.fn_stemmer = lambda tokens, lang: [language_fn_mapper[lang]["stemmer"].stem(w) for w in tokens]
    
    def detect_language(self, entry: str)->str:
        try:
            lang = detect(entry)
            if lang not in self.allowed_languages:
                logging.warning(f"{lang} not found in allowed languages. Using english(en) instead.")
                return "en"
            return lang
        except Exception as e:
            logging.warning(f"{e} Using english(en) instead.")
        return "en"
    
    def clean_entry(
        self,
        entry: str,
        language: str,
        return_tokens: bool=True
    ):
        entry = entry.strip()
        entry = "".join([w for w in entry if w not in string.punctuation]) # Removes the punctuation from the sentence
        entry_tokens = word_tokenize(entry, language=self.language_mapper.get(language, "english"))
        if self.enable_stopwords:
            entry_tokens = self.fn_stopwords(entry_tokens, language)
        if self.enable_stemming:
            entry_tokens = self.fn_stemmer(entry_tokens, language)
        if not self.enable_case_sensitive:
            entry_tokens = [w.lower() for w in entry_tokens]
        
        if return_tokens:
            return entry_tokens
        return " ".join(entry_tokens)


    def get_ngrams(
        self,
        entries: List[str],
        n: int=1
    ):
        ngrams_op = [ngrams(entry_tokens, n) for entry_tokens in entries]
        ngrams_lst = [list(x) for x in ngrams_op]
        # Flatten the list of list
        ngrams_flat_lst = [item for sublist in ngrams_lst for item in sublist]
        return Counter(ngrams_flat_lst).most_common(self.max_ngrams_items)

    def __call__(
        self,
        entries: List[str]
    )->Dict[str, Dict]:
        ngrams = dict()
        processed_entries = list()
        for entry in entries:
            if entry.strip() == "":
                continue
            detected_language = self.detect_language(entry)
            processed_entries.append(
                self.clean_entry(
                    entry,
                    language=detected_language
                )
            )
        
        if self.generate_unigrams:
            unigrams = self.get_ngrams(processed_entries, n=1)
            ngrams["unigrams"] = OrderedDict({
                " ".join(k): v for k, v in dict(unigrams).items()
            })
        
        if self.generate_bigrams:
            bigrams = self.get_ngrams(processed_entries, n=2)
            ngrams["bigrams"] = OrderedDict({
                " ".join(k): v for k, v in dict(bigrams).items()
            })
        
        if self.generate_trigrams:
            trigrams = self.get_ngrams(processed_entries, n=3)
            ngrams["trigrams"] = OrderedDict({
                " ".join(k): v for k, v in dict(trigrams).items()
            })

        return ngrams
