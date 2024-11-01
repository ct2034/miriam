import pandas as pd

if __name__ == "__main__":
    data = pd.DataFrame.from_dict({"col1": [1, 2], "col2": [3, 2]})
    print(data.head())
    print("=" * 40)

    # get with index
    print(f"{data.loc[0]=}")
    print(f'{data["col1"]=}')
    print("=" * 40)

    # set with index
    data["col2"].loc[1] = 99
    print(data.head())
    print("=" * 40)

    # add column
    # data.insert(
    #     len(data.columns),
    #     "col3",
    #     ["o"] * len(data),
    # )
    data.at[1, "col3"] = 5
    print(data.head())
    print(f"{len(data)=}")
    print(f"{len(data.columns)=}")
    print("=" * 40)

    # add row
    data.at[2, "col1"] = 0
    print(data.head())
    print(data.describe())
