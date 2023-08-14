import polars as pl
import os
import numpy as np
import re
from urllib.parse import urlparse
from nltk.stem.snowball import SnowballStemmer
from gensim.models.keyedvectors import KeyedVectors
import datetime
import newspaper
from newspaper import Article
from src.utils import *

# etl helper object and helper functios


def build_newspaper_from_url(url, config):
    source = newspaper.Source(url, config)
    source.download()
    source.parse()
    source.set_categories()
    source.download_categories()  
    source.parse_categories()
    source.generate_articles()
    return source


def extract_articles_with_regex(
    source, keywords, evaluate_mode_for_matches, evaluate_mode_for_matches_term
):
    urls_matches = []
    found_matches = []

    for article in source.articles:
        urlm = article.url
        match_score, found_match = evaluate_matches(urlm, keywords)

        if evaluate_mode_for_matches:
            match = re.search(evaluate_mode_for_matches_term, urlm)
            if match:
                print(urlm), print(match_score), print(found_match), print(keywords)

        if match_score > 0:
            urls_matches.append(urlm)
            found_matches.append(found_match)

    return urls_matches, found_matches


def filter_articles_with_similarity(
    urls_matches, topic, wordvectors, similarity_treshold
):
    max_similarity_scores = []
    urls_second_match = []

    for url_match in urls_matches:
        max_similarity_score = compute_max_similarity(url_match, topic, wordvectors)
        max_similarity_scores.append(max_similarity_score)
        if max_similarity_score > similarity_treshold:
            urls_second_match.append(url_match)

    return urls_second_match


def download_and_parse_article(url):
    article = Article(url=url)
    article.download()
    article.parse()

    # Extract the text from the article's HTML
    text = extract_text(article.html)

    # Only update lists if the text is valid and the URL has a scheme
    if text and urlparse(article.url).scheme:
        # Store the article's publication date, text, and URL
        date = article.publish_date if article.publish_date else datetime.datetime.now()
        author = article.authors if article.authors else 'na'
        title = article.title if article.title else 'na'
        keywords = article.keywords if article.keywords else 'na'
        str_list = [str(item) for item in author]
        author = '-'.join(str_list)        
        link = article.url if article.url else url
        content = text if len(text) > 0 else None

    return date, content, link, author, title, keywords
