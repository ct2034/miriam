import numpy as np
from pyflann import FLANN

# the base points
dataset = np.array([[1.0, 1, 1, 2, 3], [10, 10, 10, 3, 2], [100, 100, 2, 30, 1]])
# the points to measure
testset = np.array([[1.0, 1, 1, 1, 1], [90, 90, 10, 10, 1]])
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16
)
# the result is for each point in the testset the 2 (because on the config)
# closest points from the dataset
print(result)
print(dists)

print("-----")
dataset = np.random.rand(10000, 128)  # 10 000 points with 128 dimensions
testset = np.random.rand(1000, 128)  # 1 000 points with 128 dimensions
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 5, algorithm="kmeans", branching=32, iterations=7, checks=16
)
print(result)
print(dists)
print(np.shape(dists))
