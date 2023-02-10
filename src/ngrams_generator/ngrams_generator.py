import re
import string
import nltk

from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer, FrenchStemmer, SpanishStemmer

from collections import Counter, OrderedDict
from langdetect import detect

from typing import List, Dict

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
        self.stopwords_en = stopwords.words("english")
        self.stopwords_fr = stopwords.words("french")
        self.stopwords_es = stopwords.words("spanish")

        stemmer_en = EnglishStemmer()
        stemmer_fr = FrenchStemmer()
        stemmer_es = SpanishStemmer()

        self.max_ngrams_items = max_ngrams_items
        self.generate_unigrams = generate_unigrams
        self.generate_bigrams = generate_bigrams
        self.generate_trigrams = generate_trigrams
        self.enable_stopwords = enable_stopwords
        self.enable_stemming = enable_stemming
        self.enable_case_sensitive = enable_case_sensitive

        self.language_mapper = {
            "en": "english",
            "fr": "french",
            "es": "spanish"
        }

        language_stopwords_mapper = {
            "en": self.stopwords_en,
            "fr": self.stopwords_fr,
            "es": self.stopwords_es
        }

        language_stemmer_mapper = {
            "en": stemmer_en,
            "fr": stemmer_fr,
            "es": stemmer_es
        }

        self.fn_stopwords = lambda tokens, lang: [w for w in tokens if w not in language_stopwords_mapper[lang]]
        self.fn_stemmer = lambda tokens, lang: [language_stemmer_mapper[lang].stem(w) for w in tokens]
    
    def detect_language(self, entry: str)->str:
        try:
            return detect(entry)
        except Exception as e:
            # use logger
            print(f"{e} Using default language as english")
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
