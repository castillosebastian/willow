import polars as pl
import numpy as np
import re
import shutil
import os
from nltk.stem.snowball import SnowballStemmer
from gensim.models.keyedvectors import KeyedVectors
from bs4 import BeautifulSoup
import string
import spacy

# Stopword and stemer
stemmer = SnowballStemmer("spanish")
nlp = spacy.load("es_core_news_sm")
spanish_stopwords_spacy = spacy.lang.es.stop_words.STOP_WORDS

# Helper functions
# Define a mapping of accented characters to their unaccented counterparts
accent_mapping = {
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "ü": "u",
    "ñ": "n",
    "Á": "A",
    "É": "E",
    "Í": "I",
    "Ó": "O",
    "Ú": "U",
    "Ü": "U",
    "Ñ": "N",
}


def load_keywords(topic="narcotráfico"):
    df = pl.read_csv("data/topics.csv")
    df = df.filter(pl.col("topic") == topic)
    return df["keywords"].to_list()

def load_bd_source(topic="narcotráfico", state=None):
    df = pl.read_csv("data/portals.csv")
    # Filter by topic
    df = df.filter(pl.col("topic") == topic)
    # Filter by state if provided
    if state is not None:
        df = df.filter(df["state"] == state)    
    return df


def load_embeddings(path="models/wiki.es.vec", limit=200000):
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
    return "".join(accent_mapping.get(char, char) for char in text)


def normalize_text(url):
    """Normalize, remove accents, and stem the words in the given text."""
    # Convert to lowercase
    text = url.lower()

    # Split URLs into components
    text = re.sub(r"https?://", "", text)  # remove http/https
    text = re.sub(
        r"[\W_]+", " ", text
    )  # replace non-alphanumeric characters with space
    text = re.sub(r"[0-9]", " ", text)  # replace non-alphanumeric characters with space

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove accents
    text = remove_accents(text)

    # Stem, remove stopwords, and split into words
    return [
        stemmer.stem(word)
        for word in text.split()
        if word not in spanish_stopwords_spacy
    ]


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
    score = round(len(matches) / len(normalized_urlshort), ndigits=2)

    return score, matches


def compute_similarity(string, keywords, wordvec):
    """
    Compute the average similarity score between a list of topic words and all words in a string.
    Returns:
    - float: The average similarity score.
    """
    text = string.lower()
    text = re.sub(r"https?://", "", text)  # remove http/https
    text = re.sub(r"[\W_]+", " ", text)  # replace non-alphanumeric characters with space
    text = re.sub(r"[0-9]", " ", text)  # replace numeric characters with space
    string_words = [word for word in text.split() if word not in spanish_stopwords_spacy]

    total_scores = []
    for keyword in keywords:
        # Compute cosine similarity scores
        scores = []
        for word in string_words:
            if word in wordvec:
                similarity = wordvec.similarity(word, keyword)
                scores.append(similarity)

        max_scores = np.max(scores) if scores else 0.0
        total_scores.append(max_scores)

    # Return average score
    average_score = sum(total_scores) / len(total_scores)
    rounded_average_score = round(average_score, 4)
    return  rounded_average_score if total_scores else 0.0



# Function to extract text from HTML using BeautifulSoup
def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    return text

def view_string(long_string, chunk_size=100):     
    return [long_string[i:i+chunk_size] for i in range(0, len(long_string), chunk_size)]



def move_to_archive(filename):
    source_path = os.path.join('output', filename)
    destination_path = os.path.join('archive', filename)

    try:
        # Create the 'archive' directory if it doesn't exist
        os.makedirs('archive', exist_ok=True)

        # Move the file
        shutil.move(source_path, destination_path)
        print(f"File {filename} moved to archive successfully!")
    except FileNotFoundError:
        print(f"File {filename} not found in the output directory.")
    except PermissionError:
        print(f"Permission denied while moving the file {filename}.")
    except Exception as e:
        print(f"An unexpected error occurred while moving the file {filename}: {e}")


