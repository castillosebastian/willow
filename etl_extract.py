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
    similarity_threshold=0.4,
    urls=["https://www.infobae.com/"],
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
        topic and content in the articles. Articles exceeding this threshold
        are considered relevant. Default is 0.4.

    urls : list of str
        A list of URLs (news websites) from which articles are to be scraped.
        Default is ['https://www.infobae.com/'].

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

    # Extractor
    config_news_extractor = Config()
    config_news_extractor.fetch_images = False
    config_news_extractor.memoize_articles = False
    config_news_extractor.request_timeout = request_timeout

    # Models
    word_vectors = load_embeddings(path="models/wiki.es.vec", limit=100000)

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
    authors = []

    # Your main loop starts here
    for url in tqdm(urls, desc="Processing URLs"):
        try:
            # Initialize the counters
            total_articles = 0
            total_text_topic_related = 0
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
            urls_matches, found_matches = extract_articles_with_regex(
                source,
                keywords,
                evaluate_mode_for_matches,
                evaluate_mode_for_matches_term,
            )

            match1_regex += len(urls_matches)

            # Second match processing with SIMILARITY
            urls_second_match = filter_articles_with_similarity(
                urls_matches, topic, word_vectors, similarity_threshold
            )

            match2_similarity += len(urls_second_match)

            # Go through each articles of the urls with double match (REGEX + SIMILARITY)
            for u in urls_second_match:
                date, content, link, author = download_and_parse_article(u)

                dates.append(date)
                contents.append(content)
                links.append(link)
                authors.append(author)
                time.sleep(int(sleep_time))

            # Store results
            url_list.append(url)
            fail_build_source_list.append(fail_build_source)
            total_articles_list.append(total_articles)
            match1_regex_list.append(match1_regex)
            match2_similarity_list.append(match2_similarity)
            total_text_topic_related_list.append(total_text_topic_related)

            try:
                # Create a DataFrame
                stat_etl_topic_related = pl.DataFrame(
                    {
                        "date_extract": start_time,
                        "url": url_list,
                        "fail_build": fail_build_source_list,
                        "total_articles": total_articles_list,
                        "match1_url": match1_regex_list,
                        "match2_url_topic_related": match2_similarity_list,
                        "total_text_topic_related": total_text_topic_related_list,
                    }
                )

                # Create a Polars DataFrame with the data
                if len(dates) == 0:
                    dates.append("No news")
                if len(contents) == 0:
                    contents.append("No news")
                if len(links) == 0:
                    links.append("No news")

                news_topic_related = pl.DataFrame(
                    {
                        "date_extract": start_time,
                        "date_article": dates,
                        "content": contents,
                        "link": links,
                        "authors": authors
                    }
                )
            except Exception:
                logging.exception(f"An error occurred while generating dataframes")

        except Exception:
            logging.exception(f"An error occurred while processing {url}")

    return stat_etl_topic_related, news_topic_related


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

    stat_etl_topic_related, news_topic_related = main(urls=['https://www.infobae.com/'])

    save_dataframes(
        stat_etl_topic_related, news_topic_related, config, topic=topic, mode="local"
    )

    # Capture the script end time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_messages.append(f"END:{end_time}")

    # Log a single message containing all the accumulated information
    logging.info(" - ".join(log_messages))
    
