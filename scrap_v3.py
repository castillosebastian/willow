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
from newspaper import Article, Config, build
from urllib.parse import urlparse
from src.utils import *

# hyperparameter---------------------------------------------
topic = 'narcotráfico'
similarity_treshold = 0.4
confignews = Config() # newspaper configuration
confignews.fetch_images = False
confignews.memoize_articles = False
#confignews.request_timeout = 30
evaluate_mode_for_matches = False # evaluate match functions
evaluate_mode_for_matches_term = ''

# Prepare env------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')
home_dir = config['main']['HOME_DIR']
os.chdir(home_dir)
filename = os.path.basename(__file__)
namelog = 'logs/' + topic + '_extract.log'
logging.basicConfig(
    filename=namelog, 
    level=logging.INFO,
    format=f'%(asctime)s-{filename}-%(levelname)s-%(message)s'
)
start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
log_messages = [f"-START:{start_time}"]

# Tools -------------------------------------------------------
stemmer = SnowballStemmer("spanish")
nlp = spacy.load("es_core_news_sm")
spanish_stopwords_spacy = spacy.lang.es.stop_words.STOP_WORDS
wordvectors = load_embeddings(path='models/wiki.es.vec', limit=100000)

# Data -------------------------------------------------------
keywords = load_keywords(topic=topic)
# urls = load_urls(topic=topic)
# urls = ['https://www.elonce.com/', 'https://www.analisisdigital.com.ar/' ]
urls = ['https://www.infobae.com/']

# Initialize the lists for the DataFrame
url_list = []
fail_build_source_list = []
total_articles_list = []
match1_regex_list = []
match2_similarity_list = []
total_text_topic_related_list = []
dates = []
contents = []
links = []

# Go through each URL in the list
for url in tqdm(urls, desc='Processing URLs'):    
    try:
        # Initialize the counters        
        total_articles = 0      #       
        total_text_topic_related = 0 #
        match1_regex = 0 #
        match2_similarity = 0 #   
        fail_build_source = False             
        
        # Build the newspaper
        source = build(url, config=confignews) 

        if not source:
            fail_build_source = True
            continue    

        # Total Articles
        total_articles = source.size()

        # First match processing with REGEX
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

        match1_regex += len(urls_matches)            

        # Second match processing with SIMILARITY         
        if len(urls_matches) > 0:   
                  
            max_sin_scores = []
            urls_second_match = []
            is_similar = []

            for url_match in urls_matches:
                max_sin_score = compute_max_similarity(url_match, topic, wordvectors) 
                max_sin_scores.append(max_sin_score)
                urls_second_match.append(url_match)
                is_similar.append(max_sin_score>similarity_treshold) 
           
            match2_similarity += len(urls_second_match)            

            # Go through each articles of the urls with double math (REGEX + SIMILARITY)        
            for u in urls_second_match:                
                article = Article(url=u)      
                article.download()            
                article.parse()
                # Extract the text from the article's HTML
                text = extract_text(article.html)
                # Validate the article text and URL
                if not text or not urlparse(article.url).scheme:
                    continue
                # Increment topic related text            
                total_text_topic_related += len(text)
                # Store the article's publication date, text, and URL
                date = article.publish_date if article.publish_date else datetime.datetime.now()
                link = article.url if article.url else u
                text = text if len(text) > 0 else None
                dates.append(date)
                contents.append(text)
                links.append(link)       
                time.sleep(int(config.get('main', 'sleep_time')))

            # Add the counts to the lists
            url_list.append(url)
            fail_build_source_list.append(fail_build_source)
            total_articles_list.append(total_articles)  
            match1_regex_list.append(match1_regex)       
            match2_similarity_list.append(match2_similarity)         
            total_text_topic_related_list.append(total_text_topic_related)
            
            # Backup intermediate results
            backup_df = pl.DataFrame({
                'url': url_list,
                'build_source': fail_build_source_list,
                'total_articles': total_articles_list,
                'match1_url': match1_regex_list,
                'match2_url_topic_related': match2_similarity_list,     
                'total_text_topic_related': total_text_topic_related_list
            })
            backup_df.write_csv(config.get('main', 'backup_file'))

        else:            
            # Add the counts to the lists for 0 match
            url_list.append(url)
            fail_build_source_list.append(fail_build_source)
            total_articles_list.append(total_articles)     
            match1_regex_list.append(match1_regex)               
            match2_similarity_list.append(match2_similarity)               
            total_text_topic_related_list.append(total_text_topic_related)
            continue

    except Exception as e:
        logging.error(f'An error occurred while processing {url}: {e}')


try: 
    # Create a DataFrame
    stat_etl_topic_related = pl.DataFrame({
        'date_extract': start_time,
        'url': url_list,
        'fail_build': fail_build_source_list,
        'total_articles': total_articles_list,
        'match1_url': match1_regex_list,
        'match2_url_topic_related': match2_similarity_list,     
        'total_text_topic_related': total_text_topic_related_list
    })

    # Create a Polars DataFrame with the data
    if len(dates) == 0: dates.append('No news')
    if len(contents) == 0: contents.append('No news')
    if len(links) == 0: links.append('No news')

    news_topic_related = pl.DataFrame({
        'date_extract': start_time,             
        'date_article': dates,
        'content': contents,
        'link': links
    })

    # Generate the filename
    news_outputfilename = 'news_' + topic + '_related_'
    stat_outputfilename = 'stat_' + topic + '_related_'

    filename_news_topic_related = os.path.join(config.get('main', 'output_dir'), news_outputfilename + datetime.datetime.now().strftime('%Y-%m-%d_%H%M') + '.csv')
    filename_stat_etl_topic_related = os.path.join(config.get('main', 'output_dir'), stat_outputfilename + datetime.datetime.now().strftime('%Y-%m-%d_%H%M') + '.csv')

    # Save the DataFrame to a file
    news_topic_related.write_csv(filename_news_topic_related)
    stat_etl_topic_related.write_csv(filename_stat_etl_topic_related)

except Exception as e:
    logging.error(f'An error occurred while saving dataframes: {e}')

# Capture the script end time
end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
log_messages.append(f"END:{end_time}")

# Log a single message containing all the accumulated information
logging.info(" - ".join(log_messages))