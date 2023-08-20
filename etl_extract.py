import polars as pl
import datetime
import os
import time
import logging
import configparser
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from gensim.models.keyedvectors import KeyedVectors
from newspaper import Config
from urllib.parse import urlparse
from src.utils import *
from src.extraction import *

# Prepare env------------------------------------------------
config = configparser.ConfigParser()
config.read("config.ini")
home_dir = config["main"]["HOME_DIR"]
os.chdir(home_dir)
filename = os.path.basename(__file__)


def main(
    topic="narcotráfico",
    similarity_threshold=0.45, 
    info_source_bd= None,
    sleep_time=1,
    evaluate_mode_for_matches=False,
    evaluate_mode_for_matches_term="",
    request_timeout=7,
):
    """
    This function scrapes news articles related to a specified topic from a list of URLs.
    It leverages REGEX and similarity checks to determine the relevance of the articles.

    Parameters:
    -----------
    topic : str
        The topic of interest for which news articles are to be extracted.
        Default is 'narcotráfico'.

    similarity_threshold : float
        A threshold for the cosine similarity between word vectors of the
        topics keyword and string been considered, in this case: url. Articles exceeding this threshold
        are considered relevant. Default is 0.45, value that what testing: dev/test_wordvec_similarity.ipynb

    info_source_bd : pl.DataFrame with 'topic' and 'newsportalurl' or list of urls
        (news websites) from which articles are to be scraped.
        Ej: ['https://www.infobae.com/'].

    sleep_time : int
        The amount of time (in seconds) the scraper waits between consecutive requests
        to avoid overloading the server and getting banned. Default is 1 second.

    evaluate_mode_for_matches : bool
        If set to True, the function evaluates and prints the matches found
        using the specified evaluate_mode_for_matches_term. Default is False.

    evaluate_mode_for_matches_term : str
        A term or REGEX pattern to search for within the URLs during the evaluation mode.
        This is used only if evaluate_mode_for_matches is set to True. Default is an empty string.

    Returns:
    --------
    tuple:
        - stat_etl_topic_related : DataFrame
            A DataFrame containing statistical data related to the extraction process.

        - news_topic_related : DataFrame
            A DataFrame containing the extracted news articles related to the topic.

    Notes:
    ------
    - Keywords for the specified topic are loaded using the `load_keywords` function.
    - The function uses the `Config()` class to set up the news extractor's configuration.
      By default, image fetching is turned off, and articles are not memoized to save memory.
    - Word vectors for similarity computation are loaded from the specified path using the
      `load_embeddings` function. Only a limited number of vectors (100,000 by default) are loaded for efficiency.

    Example:
    --------
    >>> stat_data, news_data = main(topic="technology", urls=["https://www.techcrunch.com/"])
    """

    # HYPERPARAMETERS and MODELS
    keywords = load_keywords(topic=topic)
    # Bd source
    if isinstance(info_source_bd, pl.DataFrame):
        urls = info_source_bd['newsportalurl'].to_list()
        states = info_source_bd['state'].to_list()
        cities = info_source_bd['city'].to_list()      
    else:
        raise TypeError("info_source_bd must be DataFrame") 

    # Extractor
    config_news_extractor = Config()
    config_news_extractor.fetch_images = False
    config_news_extractor.memoize_articles = False
    config_news_extractor.request_timeout = request_timeout

    # Models
    word_vectors = load_embeddings(path="models/wiki.es.vec", limit=200000)


    # Initialize the lists for the DataFrame
    url_list = []
    state_list = []
    cities_list = []
    fail_build_source_list = []
    total_articles_list = []
    match1_regex_list = []
    match2_similarity_list = []    
    dates = []
    contents = []
    links = []
    similarities = []
    authors = []
    titles = []
    sumaries = []

    # Your main loop starts here
    for url, state, city in tqdm(zip(urls, states, cities), total=len(urls), desc="Processing URLs"):
        try:
            # Initialize the counters            
            total_articles = 0            
            match1_regex = 0
            match2_similarity = 0
            fail_build_source = False

            # Build the newspaper
            source = build_newspaper_from_url(url, config_news_extractor)

            if not source:
                fail_build_source = True
                continue

            # Total Articles
            total_articles = source.size()
            
            # First match processing with REGEX
            urls_matches = extract_articles_with_regex(
                source,
                keywords,
                evaluate_mode_for_matches,
                evaluate_mode_for_matches_term,
            )         

            match1_regex += len(urls_matches)

            # Second match processing with SIMILARITY
            urls_second_match, urls_second_match_score = string_with_similarity(
                urls_matches, keywords, word_vectors, similarity_threshold
            )
            
            match2_similarity += len(urls_second_match)            

            # Go through each articles of the urls with double match (REGEX + SIMILARITY)
            for u, s in zip(urls_second_match, urls_second_match_score):
                date, content, link, author, title, summary = download_and_parse_article(u)                            

                dates.append(date)
                contents.append(content)
                links.append(link)
                similarities.append(s)
                authors.append(author)
                titles.append(title)
                sumaries.append(summary)
                state_list.append(state)
                cities_list.append(city)
                time.sleep(int(sleep_time)) 
            
            # Store results
            url_list.append(url)            
            fail_build_source_list.append(fail_build_source)
            total_articles_list.append(total_articles)
            match1_regex_list.append(match1_regex)
            match2_similarity_list.append(match2_similarity)                     

            try:
                # Create a DataFrame
                stat_etl_topic_related = pl.DataFrame(
                    {
                        "date_extract": start_time,
                        "topic": topic,
                        "url": url_list,                        
                        "fail_build": fail_build_source_list,
                        "total_articles": total_articles_list,
                        "match1_url": match1_regex_list,
                        "match2_url_topic_related": match2_similarity_list                        
                    }
                )

                # Create a Polars DataFrame with the data
                outputlists = [dates, contents, links, authors, 
                               titles, state_list, cities_list,
                               sumaries, similarities]
                
                for lst in outputlists:
                    if not lst:
                        lst.append(None)               

                news_topic_related = pl.DataFrame(
                    {
                        "date_extract": start_time,
                        "date_article": dates,
                        "topic": topic,
                        "content": contents,
                        "link": links,
                        "link_sim_score": similarities,
                        "title": titles,
                        "summary": sumaries,
                        "authors": authors,
                        "portal": url,
                        "state": state_list,
                        "city": cities_list
                    }
                )
                # Add content hash and exclude duplicated articles
                news_topic_related = (
                    news_topic_related.with_columns(
                    pl.col("content").hash().alias("content_hash"),
                    pl.col("content").str.n_chars().alias("content_nchar"),
                    )
                ).unique(
                    subset=['content_hash']
                ).filter(
                    pl.col('content_nchar') > 400
                )            

                

            except Exception:
                logging.exception(f"An error occurred while generating dataframes")

        except Exception:
            logging.exception(f"An error occurred while processing {url}")
    
    return stat_etl_topic_related, news_topic_related

if __name__ == "__main__":
    topic = "narcotráfico"

    name_log = "logs/" + topic + "_extract.log"

    logging.basicConfig(
        filename=name_log,
        level=logging.INFO,
        format=f"%(asctime)s-{filename}-%(levelname)s-%(message)s",
    )
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_messages = [f"-START:{start_time}"]

    info_source_bd = load_bd_source(topic=topic)
    #info_source_bd = load_bd_source(topic=topic, state = 'Buenos Aires')
    
    stat_etl_topic_related, news_topic_related = main(info_source_bd=info_source_bd)
    
    save_dataframes(
        stat_etl_topic_related, news_topic_related, config, topic=topic, mode="local"
    )

    # Capture the script end time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_messages.append(f"END:{end_time}")

    # Log a single message containing all the accumulated information
    logging.info(" - ".join(log_messages))
    
