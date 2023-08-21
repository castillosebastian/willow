# Import required libraries
import polars as pl
from datetime import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pymongo import MongoClient, DESCENDING
from tqdm import tqdm
from src.transform import *
from src.database import *

# Define the model paths
summary_model_str = "IIC/mt5-spanish-mlsum"
ner_model = "mrm8488/bert-spanish-cased-finetuned-ner"

wordvector = load_embeddings(path="models/wiki.es.vec", limit=200000)
keywords = load_keywords(topic='narcotráfico')

def main(
        topic = 'narcotráfico',        
        summary_length = 400,
        wordvector = wordvector,
        keywords = keywords
    ):
        
    # 1. Read the input DataFrame
    df = load_rawdata(topic=topic)

    # 2. Clean the DataFrame
    df_clean = clean_dataframe(df, replace_white_lines=True)

    # 3.1 Summarize the articles
    df_clean = summarize_articles(df_clean, summary_model_str,summary_length)
    
    # 3.1 Compute similarity btwen summary and keywords
    df_clean = compute_similarity(df_clean, column_to_eval = 'summary_llm', keywords=keywords, word_vectors=wordvector)

    # Get bd index 
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    collection_news = get_collection(host='mongodb://localhost:27017/',
                                     db_name='wdocuments', 
                                     collection_name='news')
     
    # Find the document with the maximum 'index' value 
    max_index = get_max_index(collection=collection_news)
    
    # 4. Perform NER calculation
    ner_function = lambda text: ner_on_large_document(text, model=ner_model) # Customize as needed
    ner_news_df = calculate_ner(df_clean, max_index, ner_function)

    # 5. Arrange both datasets and Save to DB
    news_df, ner_df = arrange_datasets(max_index, df_clean, ner_news_df)
    
    # 6.1. Load to Mongodb News
    news = news_df.to_pandas()
    news.reset_index(inplace=True)
    news = news.to_dict("records") # Change to dict
    collection_news.insert_many(news)    
    # 6.2. Load to Mongodb NewsNER
    newsner = ner_df.to_pandas()
    newsner.reset_index(inplace=True)
    newsner = newsner.to_dict("records") # Change to dict
    collection_ner = db['newsner']
    collection_ner.insert_many(newsner)

    # 6. Add vector embedings
    process_documents(collection_news, model=model)
    
    # Print or log completion message
    print("ML pipeline execution complete!")

if __name__ == "__main__":
    main()