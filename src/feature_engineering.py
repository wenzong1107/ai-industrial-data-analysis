import pandas as pd


data = pd.read_csv("data/machine_data.csv")

data["temperature_change"] = (
    data["temperature"].diff()
)

data["vibration_change"] = (
    data["vibration"].diff()
)

data["current_change"] = (
    data["current"].diff()
)

data["temperature_rolling_mean"] = (
    data["temperature"]
    .rolling(window=10)
    .mean()
)

data["vibration_rolling_mean"] = (
    data["vibration"]
    .rolling(window=10)
    .mean()
)

data["current_rolling_mean"] = (
    data["current"]
    .rolling(window=10)
    .mean()
)

data = data.dropna()

print(data.head())

print()
print(data.columns)

data.to_csv(
    "data/machine_features.csv",
    index=False
)