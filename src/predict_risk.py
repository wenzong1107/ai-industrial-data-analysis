import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv(
    "data/machine_features.csv"
)

data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)

data = data.sort_values(
    "timestamp"
)


features = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "temperature_change",
    "vibration_change",
    "current_change",
    "temperature_rolling_mean",
    "vibration_rolling_mean",
    "current_rolling_mean"
]


X = data[features]

y = data["failure"]


split_index = int(len(data) * 0.8)


X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


latest_data = X_test.iloc[[-1]]


failure_probability = model.predict_proba(
    latest_data
)[0][1]


print()
print("==============================")
print("Industrial AI Risk Assessment")
print("==============================")

print()

print(
    "Machine:",
    data.iloc[-1]["machine_id"]
)

print(
    "Timestamp:",
    data.iloc[-1]["timestamp"]
)

print()

print(
    "Failure Probability:",
    round(failure_probability * 100, 2),
    "%"
)


if failure_probability >= 0.8:

    risk_level = "HIGH RISK"

elif failure_probability >= 0.5:

    risk_level = "MEDIUM RISK"

else:

    risk_level = "LOW RISK"


print(
    "Risk Level:",
    risk_level
)

print()

print("Current Sensor Values")

for feature in [
    "temperature",
    "pressure",
    "vibration",
    "current"
]:

    print(
        feature,
        ":",
        round(
            latest_data.iloc[0][feature],
            4
        )
    )