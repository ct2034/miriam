import sys
import numpy as np
import matplotlib.pyplot as plt

from pymongo import MongoClient
from bson.objectid import ObjectId

assert "mongodb" in sys.argv[-1], "please pass mongodb connection string"

client = MongoClient(str(sys.argv[-1]))
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


def get_array(entry):
    xlist = []
    for key in floatlist:
        xlist.append(doc[str(key).replace(".", "_")]["0"][entry])
    return np.array(xlist)


def plot_line(entry):
    plt.figure()
    plt.plot(y, get_array(entry))
    plt.title(entry)
    plt.legend(["x", "y", "ph"])


plot_line("x")
plot_line("x_goal")
plot_line("dx")


def plot_xy(entry):
    x = get_array(entry)
    plt.figure()
    plt.plot(x[:, 1], x[:, 0])
    plt.gca().set_aspect('equal')
    plt.title("X-Y " + entry)


plot_xy("x")

plt.show()
