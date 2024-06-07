import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, fetch_covtype
from perpetual import convert_input_frame, transform_input_frame


if __name__ == "__main__":
    df = sns.load_dataset("titanic")
    df.to_csv("resources/titanic.csv", index=False)
    
    X = df.select_dtypes("number").drop(columns=["survived"]).astype(float)
    y = df["survived"].astype(float)

    pd.Series(X.fillna(0).to_numpy().ravel(order="F")).to_csv(
        "resources/contiguous_no_missing.csv",
        index=False,
        header=False,
    )

    pd.Series(X.to_numpy().ravel(order="F")).to_csv(
        "resources/contiguous_with_missing.csv",
        index=False,
        header=False,
    )

    y.to_csv(
        "resources/performance.csv",
        index=False,
        header=False,
    )

    X.fare.to_csv(
        "resources/performance-fare.csv",
        index=False,
        header=False,
    )

    dfb = df.sample(
        100_000,
        random_state=0,
        replace=True,
    ).reset_index(drop=True)

    Xb = dfb.select_dtypes("number").drop(columns=["survived"]).astype(float)
    yb = dfb["survived"].astype(float)

    pd.Series(Xb.fillna(0).to_numpy().ravel(order="F")).to_csv(
        "resources/contiguous_no_missing_100k_samp_seed0.csv",
        index=False,
        header=False,
    )

    yb.to_csv(
        "resources/performance_100k_samp_seed0.csv",
        index=False,
        header=False,
    )

    data = fetch_california_housing(as_frame=True)
    data_train, data_test = train_test_split(data.frame, test_size=0.2, random_state=42)
    data_train.to_csv("resources/cal_housing_train.csv", index=False)
    data_test.to_csv("resources/cal_housing_test.csv", index=False)

    data = fetch_covtype(as_frame=True)
    data_train, data_test = train_test_split(data.frame, test_size=0.2, random_state=42)
    data_train.to_csv("resources/cover_types_train.csv", index=False)
    data_test.to_csv("resources/cover_types_test.csv", index=False)

    

    # fetch dataset: https://archive.ics.uci.edu/dataset/2/adult
    adult = fetch_ucirepo(id=2)
    data = adult.data.features.copy()
    data["sex"] = pd.get_dummies(adult.data.features["sex"], drop_first=True, dtype=float).to_numpy()
    cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    data[cols] = data[cols].astype('category')
    y = adult.data.targets["income"].str.contains("<").to_numpy().astype(int)
    
    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    features_, adult_train_flat, rows, cols, categorical_features_, cat_mapping = convert_input_frame(data_train, "auto")
    features_, adult_test_flat, rows, cols = transform_input_frame(data_test, cat_mapping)

    pd.Series(adult_train_flat).to_csv("resources/adult_train_flat.csv", index=False, header=False)
    pd.Series(adult_test_flat).to_csv("resources/adult_test_flat.csv", index=False, header=False)
    pd.Series(y_train).to_csv("resources/adult_train_y.csv", index=False, header=False)
    pd.Series(y_test).to_csv("resources/adult_test_y.csv", index=False, header=False)
