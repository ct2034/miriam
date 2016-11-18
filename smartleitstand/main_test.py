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

def run_command(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    return process.communicate()[0].decode()

def test_write_mongodb():
    data = {
        "time": datetime.datetime.now(),
        "host": run_command("hostname")
    }
    print(data)

    id = collection.insert_one(data).inserted_id
    print(id)
