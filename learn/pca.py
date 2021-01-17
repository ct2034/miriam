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
candidates = ['person' + str(i) for i in range(1, 11)]
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
    np.random.normal(1, .1, len(candidates)))  # some variance
data[features.index('age'), :] = (
    data[features.index('weight')] *
    np.random.normal(.9, .1, len(candidates)))  # some variance
data[features.index('head circumference'), :] = (
    np.power(data[features.index('height'), :], 2) * .1 *
    np.random.normal(1, .01, len(candidates)))  # less variance

print(data)

#########################
# Perform PCA on the data
#########################
# First center and scale the data
scaled_data = preprocessing.scale(np.transpose(data))
print(scaled_data)

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

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# the following code makes a fancy looking plot using PC1 and PC2
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

plt.show()

#########################
# Determine which genes had the biggest influence on PC1
#########################

# get the name of the top 10 measurements (genes) that contribute
# most to pc1.
# first, get the loading scores
loading_scores = pca.components_[0]
# now sort the loading scores based on their magnitude
sorted_loading_scores = np.sort(np.abs(loading_scores))

# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10]

# print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes])
