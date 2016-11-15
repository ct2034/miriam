import pymongo

client = pymongo.MongoClient(
    "mongodb://experiment:2VP8ewY2qn" +
    "@ds050539.mlab.com:50539/" +
    "robotarium-results"
)
db = client["robotarium-results"]
collection = db.test_collection
id = collection.insert_one({"test": 1}).inserted_id
print(id)
