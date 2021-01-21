#!/usr/bin/env python3
# source: https://github.com/StatQuest/pca_demo/blob/master/pca_demo.py
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt  # tested with v2.1.0

#########################
# Data Generation Code
#########################
candidates = ['person' + str(i) for i in range(1, 1001)]
features = (['height',
             'weight',
             'age',
             'head circumference'])

data = np.zeros([len(features), len(candidates)])

# coming up with data
data[features.index('height'), :] = np.random.uniform(
    0.8, 2.2, len(candidates))
data[features.index('weight'), :] = (
    np.power(data[features.index('height'), :], 3) * 13 *  # people are cubical
    np.random.normal(1, .3, len(candidates)))  # some variance
data[features.index('age'), :] = (
    data[features.index('weight')] * .5 *
    np.random.normal(.9, .3, len(candidates)))  # some variance
data[features.index('head circumference'), :] = (
    np.power(data[features.index('height'), :], 2) * .1 *
    np.random.normal(1, .1, len(candidates)))  # less variance

print(data.shape)
plt.figure()
x_str = 'height'
for i, s in enumerate(features[1:]):
    plt.subplot(131 + i)
    plt.scatter(data[features.index(x_str), :],
                data[features.index(s), :])
    plt.xlabel(x_str)
    plt.ylabel(s)

#########################
# Perform PCA on the data
#########################
# First center and scale the data
scaled_data = preprocessing.scale(np.transpose(data))
print(scaled_data.shape)

pca = PCA()  # create a PCA object
pca.fit(scaled_data)  # do the math
pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data
print(len(pca.components_))

#########################
# Draw a scree plot and a PCA plot
#########################

# The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
print(per_var.shape)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.figure()
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')

# the following code makes a fancy looking plot using PC1 and PC2
plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

#########################
# Determine which features had the biggest influence on PC1-4
#########################


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


fig = plt.figure()
for c in range(len(pca.components_)):
    # first, get the loading scores
    loading_scores = pca.components_[c]
    print(loading_scores)
    abs_loading_scores = np.abs(loading_scores)

    ax = fig.add_subplot(221 + c)
    ax.set_title("PC" + str(c+1))
    rects = ax.bar(x=range(1, len(loading_scores)+1),
                   height=abs_loading_scores, tick_label=features)
    autolabel(ax, rects)

plt.show()
