from pymongo import MongoClient

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