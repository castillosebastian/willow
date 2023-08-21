import polars as pl
import glob
from tqdm import tqdm
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import gc
from src.extraction import *
from src.utils import *

def load_rawdata(topic = 'narcotráfico'):
   
    # Step o
    dir = 'output/news_' + topic + '_related_*.csv'
    
    # Step 1: Get the list of all files with the specific pattern
    files = glob.glob(dir)

    # Step 2: Extract dates and sort them
    files_sorted = sorted(files, key=lambda x: datetime.datetime.strptime(x.split('_')[-2] + '_' + x.split('_')[-1][:-4], '%Y-%m-%d_%H%M'))

    # Step 3: Read the latest file
    oldest_file = files_sorted[0]

    df = pl.read_csv(oldest_file, dtypes={"content_hash": pl.UInt64})

    return df

def clean_dataframe(df, replace_white_lines=True):
    try:
        if replace_white_lines:
            content_replace_step = (
                pl.col("content")
                .str.replace_all(r"[\n\t]+", " ")
                .str.replace_all(r"\s{2,}", " ")
                .str.strip()
                .alias("content")
            )
        else:
            content_replace_step = pl.col("content")

        df_clean = (
            df.with_columns([
                pl.col('date_extract').str.strptime(pl.Date, format='%Y-%m-%d %H:%M:%S', strict=True),  # Fixed format
                pl.col('date_article').str.slice(0, 10).str.strptime(pl.Date, format='%Y-%m-%d'),
                content_replace_step
            ])
        )

        return df_clean

    except Exception as e:
        # Log or print the exception
        print(f"An error occurred during dataframe cleaning: {e}")
        # You can also return the original dataframe or handle the error in other ways
        return df


def summarize_articles(df_clean, model_str, summary_length = 400):
    try:
        # 1
        articles = df_clean['content'].to_list()

        # 2
        # Models
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_str)

        # 3
        # Process all articles
        articles_summaries = []
        for article in articles:
            input_ids = tokenizer(article, return_tensors="pt").input_ids    
            output_ids = model.generate(input_ids, max_length=summary_length, num_beams=4)[0]  # Adjusted parameters
            summary = tokenizer.decode(output_ids, skip_special_tokens=True)
            articles_summaries.append(summary)
        
        del tokenizer
        del model        
        gc.collect()
       
        # 4 Add summary for each article
        df_clean = df_clean.with_columns(
            pl.Series("summary_llm", articles_summaries)
        )

        return df_clean

    except Exception as e:
        # Log or print the exception
        print(f"An error occurred during summarization: {e}")
        # You can return the original dataframe or handle the error in other ways
        return df_clean


def compute_similarity(df, column_to_eval, keywords, word_vectors):
    try:
        # 1
        articles_summaries = df[column_to_eval].to_list()
        _, sim_score = string_with_similarity(articles_summaries, keywords, word_vectors, 0.0)
        # 2 Add summary for each article
        df = df.with_columns(pl.Series("summary_sim_score", sim_score))

        # 3.1. delete model
        del word_vectors
        del keywords
        gc.collect()

        return df

    except Exception as e:
        print(f"An error occurred during similarity computation: {e}")
        return df

def langchain_chunk_text(text):
    try:
        # Define custom text splitter
        custom_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
            length_function=len
        )

        # Create the chunks using langchain
        documents = custom_text_splitter.create_documents([text])

        # Extract the text content from the resulting Document objects
        chunks = [doc.page_content for doc in documents]

        return chunks
    except Exception as e:
        # Log or print the exception
        print(f"An error occurred during text chunking: {e}")
        # You can return an empty list or handle the error in other ways
        return []



def ner_on_large_document(text, 
                          model="mrm8488/bert-spanish-cased-finetuned-ner", 
                          aggregation_strategy="max"):
    try:
        nlp_ner = pipeline(
            "ner",
            model=model,
            aggregation_strategy=aggregation_strategy # Adjust as needed
        )

        chunks = langchain_chunk_text(text)
        all_ner_results = []

        for chunk in chunks:
            ner_results = nlp_ner(chunk)
            all_ner_results.extend(ner_results)

        return all_ner_results
    except Exception as e:
        # Log or print the exception
        print(f"An error occurred during NER processing: {e}")
        # You can return an empty list or handle the error in other ways
        return []


def calculate_ner(news_df, max_index, ner_function):
    try:
        index = list(range(max_index + 1, max_index + 1 + news_df.shape[0]))
        articles = news_df['content'].to_list()

        ner_news_df = pl.DataFrame()

        for article, i in tqdm(zip(articles, index)):
            ner = ner_function(article)
            df = pl.DataFrame(ner)
            df_len = df.shape[0]
            df = df.with_columns(pl.Series("index", [i] * df_len))
            if set(df.columns) == set(ner_news_df.columns) or ner_news_df.shape[1] == 0:
                ner_news_df = pl.concat([ner_news_df, df])

        return ner_news_df
    except Exception as e:
        print(f"An error occurred during NER calculation: {e}")
        return None


def arrange_datasets(max_index, news_df, ner_news_df):
    # Refactor this function
    try:
        index = list(range(max_index + 1, max_index + 1 + news_df.shape[0]))
        news_df = news_df.with_columns(pl.Series("index", index))

        ner_news_df = ner_news_df.join(news_df[['link', 'content_hash', 'index']], on='index', how='left')

        arranged_news_df = news_df.select([
            'index', 'topic', 'date_extract', 'date_article', 'content', 'portal', 'link',
            'link_sim_score', 'title', 'summary', 'summary_llm',"summary_sim_score", 'authors',
            'state', 'city','content_hash', 'content_nchar'
        ])

        arranged_news_df = (
            arranged_news_df.with_columns([
                pl.col("content_hash").cast(pl.Utf8), 
                pl.col("date_extract").cast(pl.Utf8), 
                pl.col("date_article").cast(pl.Utf8), 
                pl.concat_str(
                    [
                        pl.col('state'),
                        pl.col('city'),
                        pl.col("title"),
                        pl.col("summary_llm"),
                    ],
                    separator=" ",
                ).alias("tit_summary"),
            ])    
        )

        arranged_ner_df = ner_news_df.select(
            [
                'index', 'link', 'content_hash', 'entity_group', 'score', 'word', 'start', 'end'
            ]
        )

        arranged_ner_df = (
            arranged_ner_df.with_columns([
                pl.col("content_hash").cast(pl.Utf8)
            ])    
        )

        return arranged_news_df, arranged_ner_df
    except Exception as e:
        print(f"An error occurred during dataset arrangement: {e}")
        return None, None

def compute_embeddings(text, model):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model.encode(text)
    except Exception as e:
        print(f"An error occurred while computing embeddings: {e}")
        return None

def process_documents(collection,  model):
    try:
        
        # Query to find documents without the 'embeddings' field
        query = {'embeddings': {'$exists': False}}

        # Find the documents without the 'embeddings' field
        documents = collection.find(query)

        for document in documents:
            # Extract the text you want to embed
            text = document['tit_summary']

            # Compute the embeddings
            embeddings = compute_embeddings(text, model)

            if embeddings is not None:
                # Update the document with the embeddings
                update_query = {'_id': document['_id']}
                new_values = {'$set': {'embeddings': embeddings.tolist()}}
                collection.update_one(update_query, new_values)
    except Exception as e:
        print(f"An error occurred while processing documents: {e}")