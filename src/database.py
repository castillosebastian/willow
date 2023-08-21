from pymongo import MongoClient, DESCENDING
import subprocess
import time

def start_mongodb():
    try:
        command = "mongod --dbpath /home/sebacastillo/willow/mongodb &"
        subprocess.Popen(command, shell=True)
        print("MongoDB server started successfully!")
    except Exception as e:
        print(f"Failed to start MongoDB server: {e}")

def stop_mongodb():
    try:
        command = 'mongo admin --eval "db.shutdownServer()" &'
        subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print("MongoDB server stopped successfully!")

        # Optionally, add a delay to ensure that the shutdown has time to take effect
        time.sleep(2)
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop MongoDB server. Error message: {e.stderr.decode()}")

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