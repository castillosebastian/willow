# Import required libraries
import polars as pl
import time
import logging
import configparser
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pymongo import MongoClient, DESCENDING
from tqdm import tqdm
from src.transform import *
from src.database import *

# Prepare env------------------------------------------------
config = configparser.ConfigParser()
config.read("config.ini")
home_dir = config["main"]["HOME_DIR"]
os.chdir(home_dir)
filename = os.path.basename(__file__)

def main(
        files_to_process,        
        summary_model_str,
        ner_model,        
        wordvector,
        keywords,
        collection_news,
        collection_ner,
        summary_length = 400
):           

    if files_to_process:

        for file in tqdm(files_to_process, desc="Processing Files"):

            try:

                # 1. Read the input DataFrame
                df = load_file(file)

                # 2. Clean the DataFrame
                df_clean = clean_dataframe(df, replace_white_lines=True)

                # 3.1 Summarize the articles
                print('Starting sumarization')
                df_clean = summarize_articles(df_clean, summary_model_str,summary_length)
                
                # 3.1 Compute similarity btwen summary and keywords
                df_clean = compute_similarity(df_clean, column_to_eval = 'summary_llm', keywords=keywords, word_vectors=wordvector)

                # Get bd index 
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
                               
                # Find the document with the maximum 'index' value 
                max_index = get_max_index(collection=collection_news)
                
                # 4. Perform NER calculation
                print('Starting NER')
                ner_function = lambda text: ner_on_large_document(text, model=ner_model) # Customize as needed
                ner_news_df = calculate_ner(df_clean, max_index, ner_function)

                # 5. Arrange both datasets and Save to DB
                news_df, ner_df = arrange_datasets(max_index, df_clean, ner_news_df)
                
                # 6.1. Load to Mongodb News
                print('Loading to MDB')
                news = news_df.to_pandas()
                news.reset_index(inplace=True)
                news = news.to_dict("records") # Change to dict
                collection_news.insert_many(news)    
                # 6.2. Load to Mongodb NewsNER
                newsner = ner_df.to_pandas()
                newsner.reset_index(inplace=True)
                newsner = newsner.to_dict("records") # Change to dict
                
                collection_ner.insert_many(newsner)

                # 6. Add vector embedings
                process_documents(collection_news, model=model)
                                
                # 7 Move file from output (before process) to archive (after process)
                move_to_archive(file)

                # Print or log completion message
                print("ML pipeline execution complete!")

            except Exception:

                logging.exception(f"An error occurred while processing {file}")
    
    else:
        print("No files left to process")

if __name__ == "__main__":

    topic = "narcotráfico"
    name_log = "logs/" + topic + "_extract.log"
    logging.basicConfig(filename=name_log,level=logging.INFO,
        format=f"%(asctime)s-{filename}-%(levelname)s-%(message)s",
    )
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_messages = [f"-START:{start_time}"]

    # Define the model paths
    summary_model_str = "IIC/mt5-spanish-mlsum"
    ner_model = "mrm8488/bert-spanish-cased-finetuned-ner"
    wordvector = load_embeddings(path="models/wiki.es.vec", limit=200000)
    keywords = load_keywords(topic='narcotráfico')   

    files_to_process = list_rawdata(topic=topic)

    collection_news = get_collection(host='mongodb://localhost:27017/',
                                                db_name='wdocuments', 
                                                collection_name='news')
    
    collection_ner = get_collection(host='mongodb://localhost:27017/',
                                                db_name='wdocuments', 
                                                collection_name='newsner')        
    main(
        files_to_process,        
        summary_model_str=summary_model_str,
        ner_model=ner_model,
        summary_length=400,
        wordvector=wordvector,
        keywords=keywords, 
        collection_news=collection_news,
        collection_ner=collection_ner
        )
        
    # Capture the script end time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_messages.append(f"END:{end_time}")

    # Log a single message containing all the accumulated information
    logging.info(" - ".join(log_messages))