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
from src.extaction import *

# HYPERPARAMETERS and MODELS----------------------------------
# topic = 'narcotráfico'
# similarity_threshold = 0.4
# keywords = load_keywords(topic=topic)
# urls data base
# urls = load_urls(topic=topic)
# urls = ['https://www.elonce.com/', 'https://www.analisisdigital.com.ar/' ]
# urls = ['https://www.infobae.com/']
# Extractor
# config_news_extractor = Config()
# config_news_extractor.fetch_images = False
# config_news_extractor.memoize_articles = False
# sleep_time = 1
# config_news_extractor.request_timeout = 30
evaluate_mode_for_matches = False  # evaluate match functions
evaluate_mode_for_matches_term = ""
# Models
word_vectors = load_embeddings(path="models/wiki.es.vec", limit=100000)

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
                date, content, link = download_and_parse_article(u)

                dates.append(date)
                contents.append(content)
                links.append(link)
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

    stat_etl_topic_related, news_topic_related = main()

    save_dataframes(
        stat_etl_topic_related, news_topic_related, config, topic=topic, mode="local"
    )

    # Capture the script end time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_messages.append(f"END:{end_time}")

    # Log a single message containing all the accumulated information
    logging.info(" - ".join(log_messages))
