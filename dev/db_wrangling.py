from pymongo import MongoClient
from datetime import datetime
import polars as pl
import pandas as pd

# Start: mongod --dbpath /home/sebacastillo/willow/mongodb
# Shutdown: mongo admin --eval "db.shutdownServer()"

client = MongoClient('mongodb://localhost:27017/')
db = client['wdocuments']

news = pl.read_csv('/home/sebacastillo/willow/output/news.csv',
                    dtypes={"content_hash": pl.UInt64} )
newsner = pl.read_csv('/home/sebacastillo/willow/output/newsner.csv',
                    dtypes={"content_hash": pl.UInt64} )

# Convert content_hash to str
news = (
    news.with_columns([
        pl.col("content_hash").cast(pl.Utf8)
    ])    
)

newsner = (
    newsner.with_columns([
        pl.col("content_hash").cast(pl.Utf8)
    ])    
)
# drop hash
#news = news.drop('content_hash')
#newsner = newsner.drop('content_hash')
# Convert to pandas
newsp = news.to_pandas()
newsnerp = newsner.to_pandas()

# Export to mongo News
newsp.reset_index(inplace=True)
newsp = newsp.to_dict("records") # Change to dict
collection = db['news'] # export to mongo collection news
collection.insert_many(newsp)

# Export to mongo NewsNer
newsnerp.reset_index(inplace=True)
newsnerp = newsnerp.to_dict("records")
collection_ner = db['newsner']
collection_ner.insert_many(newsnerp)