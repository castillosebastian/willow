from newspaper import Article, Config, build
from bs4 import BeautifulSoup
import polars as pl
import datetime
import os
import time
from src import utils

# Parameters 
keywords = ['narcotráfico']
#keywords = ['crimen organizado', 'narcotráfico', 'narcotrafico', 'cocaina', 'traficante', 'crak', 'fentanilo']
urls = ['https://www.analisisdigital.com/']
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
config = Config()
config.fetch_images = False
config.memoize_articles = False

# Initialize the lists for the DataFrame
url_list = []
total_articles_list = []
drug_related_articles_list = []

# Initialize lists to store the data
dates = []
contents = []
links = []

# Go through each URL in the list
for url in urls:    
    try:
        # Initialize the counters
        total_articles = 0
        drug_related_articles = 0
        # Build the newspaper
        paper = build(url, config=config)        

        # Go through each article in the newspaper
        for article in paper.articles:
            # Download and parse the article
            article.download()
            article.parse()

            # Extract the text from the article's HTML
            text = extract_text(article.html)

            # Increment the total articles counter
            total_articles += 1

            # Check if the article's text contains any of the keywords
            if is_drug_related(text):
                # Increment the drug-related articles counter
                drug_related_articles += 1
                # Store the article's publication date, text, and URL
                date = article.publish_date if article.publish_date else datetime.datetime.now()
                url = article.url if article.url else url
                dates.append(date)
                contents.append(text)
                links.append(url)       

            time.sleep(2)

        # Add the counts to the lists
        url_list.append(url)
        total_articles_list.append(total_articles)
        drug_related_articles_list.append(drug_related_articles)
                
    except Exception as e:
        print(f'An error occurred while processing {url}: {e}')

# Create a DataFrame
count_drug_related = pl.DataFrame({
    'URL': url_list,
    'Total Articles': total_articles_list,
    'Drug-Related Articles': drug_related_articles_list
})

# Create a Polars DataFrame with the data
news_drug_related = pl.DataFrame({
    'date': dates,
    'content': contents,
    'link': links
})

# Generate the filename
filename_news_drug_related = os.path.join(output_dir, 'news_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M') + '.csv')
filename_count_drug_related = os.path.join(output_dir, 'count_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M') + '.csv')

# Save the DataFrame to a file
news_drug_related.write_csv(filename_news_drug_related)
count_drug_related.write_csv(filename_count_drug_related)

print(f'ETL saved data')