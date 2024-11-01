#!/usr/bin/env python3
import numpy as np
import pandas as pd

print("A DataFrame " + "=" * 60)
df = pd.DataFrame(
    data={
        "col1": range(8),
        "col2": np.linspace(9, 9.1, 8),
        "col3": np.linspace(90, 20, 8),
    }
)
print(df)
print(df.info())

print("Multiindex " + "=" * 60)
index_arrays = [[1, 2], ["red", "blue"], ["sweet", "sour"]]
idx = pd.MultiIndex.from_product(index_arrays, names=("number", "color", "taste"))

df = pd.DataFrame(
    data={
        "col1": range(8),
        "col2": np.linspace(9, 9.1, 8),
        "col3": np.linspace(90, 20, 8),
    },
    index=idx,
)
print("full df")
print(df)
print("-" * 80)

print("info")
print(df.info())
print("-" * 80)

print("nr: 1")
print(df.loc[1])
print("-" * 80)

print("color")
print(df.index.get_level_values(level="color").to_numpy())
print("-" * 80)

print("color: blue")
print(df.xs("blue", level="color"))
print((df.xs("blue", level="color")).__class__)
print("-" * 80)

print("color: blue and nr: 1")
print(df.loc[(1, "blue"), :])
print("-" * 80)

print("editing")
df.loc[(1, "blue", "sweet"), "col3"] = 66
print(df)
print("-" * 80)

print("Merging Columns " + "=" * 60)

df1 = pd.DataFrame(data={"col1": range(8), "col2": np.linspace(9, 9.1, 8)}, index=idx)
print("df1")
print(df1)
print("-" * 80)

df2 = pd.DataFrame(data={"col3": np.linspace(90, 20, 8)}, index=idx)
print("df2")
print(df2)
print("-" * 80)

assert len(df2.columns) == 1
assert df2.columns[0] not in df1.columns
df1[df2.columns[0]] = df2
print("df1 after merge")
print(df1)
print("-" * 80)
