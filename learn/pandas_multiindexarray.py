#!/usr/bin/env python3
import numpy as np
import pandas as pd

print("A DataFrame " + "="*30)
df = pd.DataFrame(data={
    'col1': range(8),
    'col2': np.linspace(9, 9.1, 8),
    'col3': np.linspace(90, 20, 8)})
print(df)
print(df.info())

print("Multiindex " + "="*30)
index_arrays = [[1, 2], ['red', 'blue'], ['sweet', 'sour']]
idx = pd.MultiIndex.from_product(
    index_arrays, names=('number', 'color', 'taste'))

df = pd.DataFrame(data={
    'col1': range(8),
    'col2': np.linspace(9, 9.1, 8),
    'col3': np.linspace(90, 20, 8)},
    index=idx
)
print(df)
print(df.info())
print(df.loc[1])
print(df.loc[(1, 'blue'), :])
df.loc[(1, 'blue', 'sweet'), 'col3'] = 66
print(df)
