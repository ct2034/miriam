import pymongo
import datetime

client = pymongo.MongoClient(
    "mongodb://experiment:2VP8ewY2qn" +
    "@ds050539.mlab.com:50539/" +
    "robotarium-results"
)
db = client["robotarium-results"]

collection = db.test_collection
data = {"test": datetime.datetime.now()}
print(data)

id = collection.insert_one(data).inserted_id
print(id)
