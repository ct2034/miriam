import datetime
import numpy as np
import matplotlib.pyplot as plt

from pymongo import MongoClient
from bson.objectid import ObjectId


client = MongoClient(
    "mongodb://experiment:2VP8ewY2qn" +
    "@ds050539.mlab.com:50539/" +
    "robotarium-results"
)
db = client["robotarium-results"]
collection = db.test_collection
doc = collection.find_one({"_id": ObjectId("582c1a4afd94ce24372e7725")})
# print(entry)

keys = doc.keys()
keylist = list(keys)

keylist.remove("_id")
keylist.remove("experiment")

floatlist = []
for key in keylist:
    floatlist.append(float(key.replace("_", ".")))
floatlist.sort()

y = np.array(floatlist)


def plot_line(entry):
    xlist = []
    for key in floatlist:
        xlist.append(doc[str(key).replace(".", "_")]["0"][entry])

    x = np.array(xlist)

    plt.figure()
    plt.plot(y, x)
    plt.title(entry)
    plt.legend(["x", "y", "ph"])


plot_line("x")
plot_line("x_goal")
plot_line("dx")


def plot_xy(entry):
    xlist = []
    for key in floatlist:
        xlist.append(doc[str(key).replace(".", "_")]["0"][entry])

    x = np.array(xlist)

    plt.figure()
    plt.plot(x[:, 1], x[:, 0])
    plt.gca().set_aspect('equal')
    plt.title("X-Y " + entry)


plot_xy("x")

plt.show()
