import logging
import os
import pymongo

client = pymongo.MongoClient(
    "mongodb://testing:6R8IimXpg0TqVDwm" +
    "@ds033607.mlab.com:33607/smartleitstand-results"
)
db = client["smartleitstand-results"]
cursor = db.test_collection.find({}).sort({'time': -1})

n_results = 10
i = 0

for res in cursor:
    if i < n_results:
        res = cursor.next()
        print(res)
        i += 1
