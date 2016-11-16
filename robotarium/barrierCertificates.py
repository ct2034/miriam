import datetime
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from robotarium import Robotarium, transformations, controllers

# Get Robotarium object used to communicate with the robots/simulator.
r = Robotarium()

# Get the number of available agents from the Robotarium. We don't need a
# specific value from this algorithm.
n = r.get_available_agents()

# Number of iterations.
iterations = 1500

# Initialize the Robotarium object with the desired number of agents.
r.initialize(n)

# Initialize velocity vector for agents. Each agent expects a 2x1 velocity
# vector containing the linear and angular velocity, respectively.
dx = np.zeros((2, n))

x_goal = (np.random.rand(2, n) * .8 - .4)

start_time = datetime.datetime.now()

data = {}
data["experiment"] = {
    "n": n,
    "iterations": iterations,
    "start": start_time
}

# iterate for the previously specified number of iterations.
for it in range(iterations):
    # Retrieve teh most recent poses from teh Robotarium. The time delay is
    # approximately 0.033 seconds.
    x = r.get_poses()
    x_temp = x[0:2, :]

    # New goal?
    if np.linalg.norm(x_goal - x_temp, ord=1) < 0.08:
        x_goal = np.random.rand(2, n) - .5
        print("New goal")

    dx = controllers.position_int(x, x_goal, 0.9)

    # Saturation of controls
    dx_max = 0.1
    for i in range(0, n):
        if np.linalg.norm(dx[:, i]) > dx_max:
            dx[:, i] = dx[:, i] / np.linalg.norm(dx[:, i]) * dx_max

    # Ensure the robots don't collide
    dx = transformations.barrier_certificate(dx, x, ds=0.1)

    # Transform the single-integrator dynamics to unicycle dynamics using a
    # diffeomorphism, which can be found in the utilities.
    dx = transformations.int_to_uni2(dx, x, 0.75, np.pi)

    # SAVE SOME METRICS ...
    if it % 10 is 0:
        now = datetime.datetime.now()
        now_str = str(
            (now - start_time).total_seconds()
        ).replace('.', '_')
        data[now_str] = {}
        data[now_str]["t"] = now
        for agent in range(n):
            data[now_str][str(agent)] = {
                "x": x[:, agent].tolist(),
                "x_goal": x_goal[:, agent].tolist(),
                "dx": dx[:, agent].tolist()
            }

        # Set velocities of agents 1,...,n
    r.set_velocities(range(0, n), dx)

    perc = (it / iterations * 100)
    if perc % 10 == 0:
        print(perc, "%")

    # Send the previously set velocities to the agents.
    # This function must be called.
    r.step()

legend = []
for agent in range(n):
    legend.append(str(agent))

# plt.figure()
# plt.plot(np.transpose(nearest_neighbors))
# plt.legend(legend)
# plt.savefig(str(
#     time.ctime())
#     .replace(' ', '')
#     .replace(':', '')
#     + ".png")

data["experiment"].update({"end": datetime.datetime.now()})

client = pymongo.MongoClient(
    "mongodb://experiment:2VP8ewY2qn" +
    "@ds050539.mlab.com:50539/" +
    "robotarium-results"
)
db = client["robotarium-results"]
collection = db.test_collection
print(data)

id = collection.insert_one(data).inserted_id
print("Saved reults with id: ", id)
