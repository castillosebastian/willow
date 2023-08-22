import polars as pl
from pymongo import MongoClient, DESCENDING


def get_collection(host='mongodb://localhost:27017/', db_name='wdocuments', collection_name='news'):
    client = MongoClient(host)
    db = client[db_name]
    collection = db[collection_name]
    return collection

def get_max_index(collection):
    document_with_max_index = collection.find().sort('index', DESCENDING).limit(1)
    max_index = None
    for doc in document_with_max_index:
        max_index = doc['index']
    return max_index


def format_news(df):
    try:
        # Transform 'date_extract' to Date type
        df = df.with_columns(pl.col('date_extract').str.strptime(pl.Date, format='%Y-%m-%d'))
    except Exception as e:
        print("An error occurred while transforming 'date_extract':", str(e))
        return None

    try:
        # Transform 'date_article' to Date type
        df = df.with_columns(pl.col('date_article').str.strptime(pl.Date, format='%Y-%m-%d'))
    except Exception as e:
        print("An error occurred while transforming 'date_article':", str(e))
        return None

    try:
        # Transform 'content_hash' to UInt64 type
        df = df.with_columns(pl.col('content_hash').cast(pl.UInt64))
    except Exception as e:
        print("An error occurred while transforming 'content_hash':", str(e))
        return None

    return df


def get_news(collection, start_date, end_date, topic=None, embed=False):
    
    # Construct the query
    if topic:
        query = {
            "topic": {'$regex': topic, '$options': 'i'},
            "date_extract": {'$gte': start_date, '$lt': end_date}
        }
    else: 
        query = {           
            "date_extract": {'$gte': start_date, '$lt': end_date}
        }

    # Construct the projection to exclude '_id', and optionally 'embeddings'
    projection = {'_id': 0, 'level_0': 0}
    if not embed:
        projection['embeddings'] = 0

    # Execute the query
    try:
        result = list(collection.find(query, projection))
        result = pl.DataFrame(result)
        return format_news(result) # Assuming format_data is a function that formats the result
    except Exception as e:
        print("An error occurred while querying the documents:", str(e))
        return None

def get_news_byindex(collection, index_start, index_end, topic=None, embed=False):
    
    # Convert the date strings to datetime objects
    #start_date = datetime.strptime(start_date, "%Y-%m-%d")
    #end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Construct the query
    if topic:
        query = {
            "topic": {'$regex': topic, '$options': 'i'},
            "index": {'$gte': index_start, '$lte': index_end}
        }
    else: 
        query = {           
            "index": {'$gte': index_start, '$lte': index_end}
        }

    # Construct the projection to exclude '_id', and optionally 'embeddings'
    projection = {'_id': 0, 'level_0': 0}
    if not embed:
        projection['embeddings'] = 0

    # Execute the query
    try:
        result = list(collection.find(query, projection))
        result = pl.DataFrame(result)
        return format_news(result)
    except Exception as e:
        print("An error occurred while querying the documents:", str(e))
        return None

def get_ner_byindex(collection, df):
    # Extract 'index' values from polars 'df' to query collection
    index_values = df['index'].to_list()
    
    # Construct the query using the 'index' values
    query = {"index": {'$in': index_values}}

    # Projection can be adjusted as needed, for now, excluding '_id'
    projection = {'_id': 0, 'level_0': 0}

    # Execute the query
    try:
        result = list(collection.find(query, projection))
        result = pl.DataFrame(result)
        return result
    except Exception as e:
        print("An error occurred while querying the documents:", str(e))
        return None