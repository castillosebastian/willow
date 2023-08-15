from pymongo import MongoClient
from datetime import datetime

client = MongoClient('mongodb://localhost:27017/')
db = client['wdocuments']
collection = db['articles']

# Now you can perform operations on the collection
# From here https://chat.openai.com/c/862cdf34-089f-4b67-84a8-c8dae49e6422
{
  "_id": ObjectId("60f7aefb4d2f5c3b3e4a0f5b"),
  "portal": "Example News",
  "url": "https://www.example.com/news/article-123",
  "url_sha": "hash_of_url",
  "content": "This is the content of the example article...",
  "content_sha": "hash_of_content",
  "knnVector": [0.123, 0.456, -0.789, ...], // The embedding vector
  "date": ISODate("2023-08-14T00:00:00Z"),
  "title": "Breaking News: Example Article",
  "tags": ["breaking news", "example"],
  "author": "John Doe"
}
