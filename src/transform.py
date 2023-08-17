import polars as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import gc
from src.extraction import *
from src.utils import *
from tqdm import tqdm

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


def calculate_ner(news_df, ner_function):
    try:
        index = list(range(1, news_df.shape[0] + 1))
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

def arrange_datasets(news_df, ner_news_df):
    try:
        index = list(range(1, news_df.shape[0] + 1))
        news_df = news_df.with_columns(pl.Series("index", index))

        ner_news_df = ner_news_df.join(news_df[['link', 'content_hash', 'index']], on='index', how='left')

        arranged_news_df = news_df.select([
            'index', 'topic', 'date_extract', 'date_article', 'content', 'portal', 'link',
            'link_sim_score', 'title', 'summary', 'summary_llm',"summary_sim_score", 'authors',
            'state', 'city','content_hash', 'content_nchar'
        ])

        arranged_ner_df = ner_news_df.select(
            [
                'index', 'link', 'content_hash', 'entity_group', 'score', 'word', 'start', 'end'
            ]
        )

        return arranged_news_df, arranged_ner_df
    except Exception as e:
        print(f"An error occurred during dataset arrangement: {e}")
        return None, None

