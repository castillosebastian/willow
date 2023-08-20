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
    
    for article in source.articles:
        urlm = article.url
        match_score, found_match = evaluate_matches(urlm, keywords)

        if evaluate_mode_for_matches:
            match = re.search(evaluate_mode_for_matches_term, urlm)
            if match:
                print(urlm), print(match_score), print(found_match), print(keywords)

        if match_score > 0:
            urls_matches.append(urlm)
            
    return urls_matches


def string_with_similarity(
    strings, keywords, wordvectors, similarity_treshold
):
    
    s_second_match = []
    s_second_match_score = []

    for string in strings:

        similarity_score = compute_similarity(string, keywords, wordvectors)
        
        if similarity_score > similarity_treshold:
            s_second_match.append(string)
            s_second_match_score.append(similarity_score)

    return s_second_match, s_second_match_score

def download_and_parse_article(url):
    article = Article(url=url)
    article.download()
    article.parse()
    article.nlp()

    # Extract the text from the article's HTML
    text = extract_text(article.html)

    # Only update lists if the text is valid and the URL has a scheme
    if text and urlparse(article.url).scheme:
        # Store the article's publication date, text, and URL
        date = article.publish_date if article.publish_date else datetime.datetime.now()
        author = article.authors if article.authors else 'na'
        title = article.title if article.title else 'na'
        summary = article.summary if article.summary else 'na' # todo
        str_list = [str(item) for item in author]
        author = '-'.join(str_list)        
        link = article.url if article.url else url
        content = text if len(text) > 0 else None

    return date, content, link, author, title, summary

def save_dataframes(stat_df, news_df, config, topic=None, mode="local", db_params=None):
    # For local saving
    if mode == "local":
        # Generate the filename
        news_outputfilename = "news_" + topic + "_related_"
        stat_outputfilename = "stat_" + topic + "_related_"

        filename_news_topic_related = os.path.join(
            config.get("main", "output_dir"),
            news_outputfilename
            + datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
            + ".csv",
        )
        filename_stat_etl_topic_related = os.path.join(
            config.get("main", "output_dir"),
            stat_outputfilename
            + datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
            + ".csv",
        )
        # Save the DataFrame to a file
        news_df.write_csv(filename_news_topic_related)
        stat_df.write_csv(filename_stat_etl_topic_related)

    # For database saving (you can expand this with your database logic)
    elif mode == "database" and db_params:
        # Save to database logic here
        pass  # replace with actual database-saving logic

    else:
        raise ValueError(
            "Invalid mode provided or missing db_params for database mode."
        )
