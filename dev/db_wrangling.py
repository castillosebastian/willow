from pymongo import MongoClient
from datetime import datetime

# START MONGO
# sudo systemctl start mongod
# STOP MONGO
# sudo systemctl stop mongod
# REMOVE MONGO https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
# shutdown
# mongo admin --eval "db.shutdownServer()"

client = MongoClient('mongodb://localhost:27017/')
db = client['wdocuments']
collection = db['news']

# Save to mongo
from pymongo import MongoClient
from datetime import datetime

