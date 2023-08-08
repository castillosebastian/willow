import polars as pl
import datetime
import numpy as np
import os
import re
import time
import string
import spacy
import logging
import configparser
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from gensim.models.keyedvectors import KeyedVectors
from bs4 import BeautifulSoup
from newspaper import Article, Config, build
from urllib.parse import urlparse
from src.utils import *

# hyperparameter---------------------------------------------
topic = 'narcotráfico'
#keywords = load_keywords(topic='narcotráfico')
keywords = [topic]
# urls = load_urls(topic=topic)
urls = ['https://www.elonce.com/', 'https://www.analisisdigital.com.ar/' ]
output_dir = 'output'
confignews = Config() # newspaper configuration
confignews.fetch_images = False
confignews.memoize_articles = False
#confignews.request_timeout = 30


# Prepare env------------------------------------------------
# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')
home_dir = config['main']['HOME_DIR']
os.chdir(home_dir)
# Set up logging
logging.basicConfig(filename=config.get('main', 'log_file'), level=logging.INFO)


# Tools -------------------------------------------------------
# Stopword and stemer
stemmer = SnowballStemmer("spanish")
nlp = spacy.load("es_core_news_sm")
spanish_stopwords_spacy = spacy.lang.es.stop_words.STOP_WORDS

# Function to extract text from HTML using BeautifulSoup
def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Load embeddings
wordvectors = load_embeddings(path='models/wiki.es.vec', limit=100000)

# Initialize the lists for the DataFrame
url_list = []
total_articles_list = []
drug_related_articles_list = []
total_text_drug_related_list = []
dates = []
contents = []
links = []

# Go through each URL in the list
for url in tqdm(urls, desc='Processing URLs'):    
    try:
        # Initialize the counters
        total_articles = 0
        drug_related_articles = 0        
        total_text_drug_related = 0

        # Build the newspaper
        source = build(url, config=confignews) 

        if not source:
            logging.warning(f"Failed to build source for URL: {url}")
            continue    

        # Total Articles
        total_articles = source.size()

        # First match processing with REGEX
        urls_matches = []
        found_matches = []   

        for article in source.articles:
            
            url = article.url
            match_score, found_match = evaluate_matches(url, keywords)
            
            if match_score > 0:
                urls_matches.append(url) 
                found_matches.append(found_match)     

        print(f'First regex_match of len {len(urls_matches)}')                

        # Second match processing with SIMILARITY 
        # Check if urls_matches is not empty
        if len(urls_matches) > 0:
            
            # Second match processing with SIMILARITY        
            max_sin_scores = []
            urls_second_match = []
            is_similar = []

            for url_match in urls_matches:
                max_sin_score = compute_max_similarity(url_match, topic, wordvectors) 
                max_sin_scores.append(max_sin_score)
                urls_second_match.append(url_match)
                is_similar.append(max_sin_score>0.4) 

            #df = pl.DataFrame({
            #    "max_sin_scores": max_sin_scores,
            #    "urls_second_match": urls_second_match,
            #    "is_similar": is_similar
            #}).filter(
            #    df['is_similar'] == 1
            #)           

            drug_related_articles = len(urls_second_match)
            print(f'Second similarity_match of len {drug_related_articles}')    

            # Go through each articles of the urls with double math (REGEX + SIMILARITY)        
            for url in urls_second_match:
                
                article = Article(url=url)
                
                # Download and parse the article
                article.download()            
                article.parse()

                # Extract the text from the article's HTML
                text = extract_text(article.html)

                # Validate the article text and URL
                if not text or not urlparse(article.url).scheme:
                    continue

                # Increment drug related text            
                total_text_drug_related += len(text)

                # Store the article's publication date, text, and URL
                date = article.publish_date if article.publish_date else datetime.datetime.now()
                url = article.url if article.url else url
                dates.append(date)
                contents.append(text)
                links.append(url)       

                time.sleep(int(config.get('main', 'sleep_time')))

            # Add the counts to the lists
            url_list.append(url)
            total_articles_list.append(total_articles)
            drug_related_articles_list.append(drug_related_articles)       
            total_text_drug_related_list.append(total_text_drug_related)

            # Backup intermediate results
            backup_df = pl.DataFrame({
                'URL': url_list,
                'Total Articles': total_articles_list,
                'Drug-Related Articles': drug_related_articles_list,            
                'Total Text Drug Related': total_text_drug_related_list
            })
            backup_df.write_csv(config.get('main', 'backup_file'))

        else:
            print(f"No valid matches found for URL: {url}. Skipping to next URL.")
            continue

    except Exception as e:
        logging.error(f'An error occurred while processing {url}: {e}')

# Create a DataFrame
count_drug_related = pl.DataFrame({
    'URL': url_list,
    'Total Articles': total_articles_list,
    'Drug-Related Articles': drug_related_articles_list,    
    'Total Text Drug Related': total_text_drug_related_list
})

# Create a Polars DataFrame with the data
news_drug_related = pl.DataFrame({
    'date': dates,
    'content': contents,
    'link': links
})

# Generate the filename
filename_news_drug_related = os.path.join(config.get('main', 'output_dir'), 'news_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M') + '.csv')
filename_count_drug_related = os.path.join(config.get('main', 'output_dir'), 'count_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M') + '.csv')

# Save the DataFrame to a file
news_drug_related.write_csv(filename_news_drug_related)
count_drug_related.write_csv(filename_count_drug_related)

logging.info(f'ETL saved data')