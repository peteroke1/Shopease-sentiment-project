import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import sys
import nltk
import logging
from src.data_injection import data_injection

from config.constant import Cleaned_Data

logging.basicConfig(level=logging.INFO)

Sample_Sentiment_Analysis_Dataset2 = data_injection()

class DataCleaning:
    def __init__(self):
        self._ensure_nltk()

    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                return spacy.load(model)
            except OSError:
                continue
        nlp_fallback = spacy.blank("xx")
        return nlp_fallback


    def _ensure_nltk(self) -> None:
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")
        # Newer NLTK versions split tokenizer tables into "punkt_tab"
        try:
                nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            try:
                nltk.download("punkt_tab")
            except Exception:
                pass


    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing noise:

        Processing Steps:
        1. Converts all charaters ti lowercase.
        2. Remove URLs and special characters.
        3. Remove extra whitespaces.

        This helps the model focus on the actual words instead of punctuation.
        """
        text = str(text).lower()
        # The regex below keeps: letters (a-z), accented letter (À-ÿ)(https://www.accentletters.com/), numbers, and spaces.
        # Everything else (emojis, punctuation, and symbols) is removed.
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize(self, text: str) -> str:
        """
        Groups different forms of word so they can be analyzed as a single item.
        Example: "running", "run", all become "run".

        We use spaCy (NLP Libraries) to lemmatize the text.
        """
        NLP = self._load_nlp()
        doc = NLP(text)
        return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)


    def remove_stopwords(self, text: str) -> str:
        """
        Remove common, low-importance words (e.g the, is, in).
        Focuses on the model on the meaningful keywords like "great", "bad", or "broken".
        """
        tokens = word_tokenize(text)
        print(tokens)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)

def clean_data(Sample_Sentiment_Analysis_Dataset2: pd.DataFrame):
    try:
        cleaner = DataCleaning()
        Sample_Sentiment_Analysis_Dataset2["clean_text"] = Sample_Sentiment_Analysis_Dataset2["review"].apply(cleaner.clean_text)
        Sample_Sentiment_Analysis_Dataset2["lemma_text"] = Sample_Sentiment_Analysis_Dataset2["clean_text"].apply(cleaner.lemmatize)
        Sample_Sentiment_Analysis_Dataset2["final_text"] = Sample_Sentiment_Analysis_Dataset2["lemma_text"].apply(cleaner.remove_stopwords)

        #Create the answer key for the AI
        Sample_Sentiment_Analysis_Dataset2["label"] = Sample_Sentiment_Analysis_Dataset2["rating"].apply(lambda r: 0 if r in (1,2) else (1 if r == 3 else 2))
        Sample_Sentiment_Analysis_Dataset2 = Sample_Sentiment_Analysis_Dataset2[["review", "final_text", "label"]]
        logging.info("Data successfully cleaned...")
        Sample_Sentiment_Analysis_Dataset2.head(5)
        Sample_Sentiment_Analysis_Dataset2.to_csv(Cleaned_Data)
        return Sample_Sentiment_Analysis_Dataset2
    except Exception as e:
        logging.error(f"error occurred while cleaning the data {e}")

