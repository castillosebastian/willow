import polars as pl
import os 
import numpy as np
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
import string
import spacy

# Stopword and stemer
stemmer = SnowballStemmer("spanish")
nlp = spacy.load("es_core_news_sm")
spanish_stopwords_spacy = spacy.lang.es.stop_words.STOP_WORDS

# Helper functions
# Define a mapping of accented characters to their unaccented counterparts
accent_mapping = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    'ü': 'u', 'ñ': 'n',
    'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
    'Ü': 'U', 'Ñ': 'N'
}

def load_keywords(topic = 'narcotráfico'):
    df = pl.read_csv('data/topics.csv')
    df = df.filter(pl.col('topic') == topic)
    return df['keywords'].to_list()

def load_urls(topic = 'narcotráfico'):
    df = pl.read_csv('data/portals.csv')
    df = df.filter(pl.col('topic') == topic)
    return df['newsportalurl'].to_list()

from gensim.models.keyedvectors import KeyedVectors

def load_embeddings(path='models/wiki.es.vec', limit=100000):
    """
    Load the word embeddings from the specified path.    
    Args:
    - path (str): Path to the embeddings model.
    - limit (int): Limit the number of word vectors loaded.
    
    Returns:
    - KeyedVectors: Loaded word vectors.
    """
    return KeyedVectors.load_word2vec_format(path, limit=limit)

def import_data(data):
    return pl.read_csv(data)

def remove_accents(text):
    """Remove accents from the given text."""
    return ''.join(accent_mapping.get(char, char) for char in text)

def normalize_text(url):
    """Normalize, remove accents, and stem the words in the given text."""
    # Convert to lowercase
    text = url.lower()
    
    # Split URLs into components
    text = re.sub(r'https?://', '', text)  # remove http/https
    text = re.sub(r'[\W_]+', ' ', text)    # replace non-alphanumeric characters with space
    text = re.sub(r'[0-9]', ' ', text)    # replace non-alphanumeric characters with space

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove accents
    text = remove_accents(text)
    
    # Stem, remove stopwords, and split into words
    return [stemmer.stem(word) for word in text.split() if word not in spanish_stopwords_spacy]
    
def evaluate_matches(urlshort, keywords):
    """Evaluate the similarity between a urlshort and a list of keywords."""
    # Normalize urlshort and keywords
    keywords_string = " ".join(keywords)    

    normalized_urlshort = normalize_text(urlshort)
    normalized_keywords = normalize_text(keywords_string)


    normalized_keywords = [normalize_text(keyword) for keyword in keywords]
    
    # Flatten the list of keywords (since some keywords can be multi-word phrases)
    flat_keywords = [word for sublist in normalized_keywords for word in sublist]
    
    # Find matches
    matches = [word for word in normalized_urlshort if word in flat_keywords]
    
    # Calculate the score
    score = round( len(matches) / len(normalized_urlshort), ndigits=2)
    
    return score, matches

def compute_max_similarity(url, topic_word, wordvec):
    """
    Compute the median similarity score between a topic word and all words in a URL.      
    Returns:
    - float: The median similarity score.
    """
    text = url.lower()
    text = re.sub(r'https?://', '', text)  # remove http/https
    text = re.sub(r'[\W_]+', ' ', text)    # replace non-alphanumeric characters with space
    text = re.sub(r'[0-9]', ' ', text)    # replace numeric characters with space    
    url_words = [word for word in text.split() if word not in spanish_stopwords_spacy]
    
    # Compute cosine similarity scores
    scores = []
    for word in url_words:
        if word in wordvec:            
            v_url_word = wordvec[word]
            v_topic = wordvec[topic_word]        
            similarity = wordvec.similarity(word, topic_word)
            scores.append(similarity)   
    
    # Return median score    
    return np.max(scores) if scores else 0.0