from newspaper import Article, Config, build
from bs4 import BeautifulSoup
import polars as pl
import datetime
import os
import time
import logging
import configparser
from tqdm import tqdm
from urllib.parse import urlparse
from src import utils

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Set up logging
logging.basicConfig(filename=config.get('main', 'log_file'), level=logging.INFO)

# Parameters 
keywords = ['narcotráfico']
#keywords = ['crimen organizado', 'narcotráfico', 'narcotrafico', 'cocaina', 'traficante', 'crak', 'fentanilo']
urls = ['https://www.elonce.com/']
output_dir = 'output'

# Function to extract text from HTML using BeautifulSoup
def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Function to check if a string contains any of the keywords
def is_drug_related(string):
    return any(keyword in string.lower() for keyword in keywords)

# Configure newspaper
confignews = Config()
confignews.fetch_images = False
confignews.memoize_articles = False
#confignews.request_timeout = 30

# Initialize the lists for the DataFrame
url_list = []
total_articles_list = []
drug_related_articles_list = []
total_text_analysed_list = []
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
        total_text_analysed = 0
        total_text_drug_related = 0

        # Build the newspaper
        paper = build(url, config=confignews)        

        # Go through each article in the newspaper
        for article in paper.articles:
            # Download and parse the article
            article.download()
            article.parse()

            # Extract the text from the article's HTML
            text = extract_text(article.html)

            # Validate the article text and URL
            if not text or not urlparse(article.url).scheme:
                continue

            # Increment the total articles counter and total text analysed
            total_articles += 1
            total_text_analysed += len(text)

            # Check if the article's text contains any of the keywords
            if is_drug_related(text):
                # Increment the drug-related articles counter and total text drug related
                drug_related_articles += 1
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
        total_text_analysed_list.append(total_text_analysed)
        total_text_drug_related_list.append(total_text_drug_related)

        # Backup intermediate results
        backup_df = pl.DataFrame({
            'URL': url_list,
            'Total Articles': total_articles_list,
            'Drug-Related Articles': drug_related_articles_list,
            'Total Text Analysed': total_text_analysed_list,
            'Total Text Drug Related': total_text_drug_related_list
        })
        backup_df.write_csv(config.get('main', 'backup_file'))
                
    except Exception as e:
        logging.error(f'An error occurred while processing {url}: {e}')

# Create a DataFrame
count_drug_related = pl.DataFrame({
    'URL': url_list,
    'Total Articles': total_articles_list,
    'Drug-Related Articles': drug_related_articles_list,
    'Total Text Analysed': total_text_analysed_list,
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