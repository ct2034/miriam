import sys
import numpy as np
import matplotlib.pyplot as plt

from pymongo import MongoClient
from bson.objectid import ObjectId

assert "mongodb:" in sys.argv[-1], "please pass mongodb connection string"

client = MongoClient(str(sys.argv[-1]))
db = client["robotarium-results"]
collection = db.test_collection
doc = collection.find_one({"_id": ObjectId("582c1a4afd94ce24372e7725")})

keys = doc.keys()
keylist = list(keys)

keylist.remove("_id")
keylist.remove("experiment")

floatlist = []
for key in keylist:
    floatlist.append(float(key.replace("_", ".")))
floatlist.sort()

y = np.array(floatlist)

fig = plt.figure()
p = 1


def get_array(entry):
    xlist = []
    for key in floatlist:
        xlist.append(doc[str(key).replace(".", "_")]["0"][entry])
    return np.array(xlist)


def plot_line(entry):
    fig.add_subplot(2, 2, p)
    plt.plot(y, get_array(entry))
    plt.title(entry)
    plt.legend(["x", "y", "ph"])


plot_line("x")
p += 1
plot_line("x_goal")
p += 1
plot_line("dx")
p += 1


def plot_xy(entry):
    fig.add_subplot(2, 2, p)
    x = get_array(entry)
    plt.plot(x[:, 1], x[:, 0])
    plt.gca().set_aspect('equal')
    plt.title("X-Y " + entry)


plot_xy("x")

plt.show()
