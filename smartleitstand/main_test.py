import pymongo
import datetime
import subprocess

client = pymongo.MongoClient(
    "mongodb://experiment:2VP8ewY2qn" +
    "@ds050539.mlab.com:50539/" +
    "robotarium-results"
)
db = client["robotarium-results"]

collection = db.test_collection


def test_write_mongodb():
    bashCommand = "hostname"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    hostname = process.communicate()[0]
    data = {
        "time": datetime.datetime.now(),
        "host": hostname.decode()
    }
    print(data)

    id = collection.insert_one(data).inserted_id
    print(id)

test_write_mongodb()
