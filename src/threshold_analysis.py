import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from xgboost import XGBClassifier


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


models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
}


thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]


all_results = []


for name, model in models.items():

    print()
    print("Analyzing:", name)

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    for threshold in thresholds:

        y_pred = (
            probabilities >= threshold
        ).astype(int)


        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        all_results.append({

            "Model": name,

            "Threshold": threshold,

            "Precision": precision,

            "Recall": recall,

            "F1": f1
        })


results = pd.DataFrame(
    all_results
)


print()
print(results)


results.to_csv(
    "data/threshold_results.csv",
    index=False
)


for metric in ["Precision", "Recall", "F1"]:

    plt.figure(figsize=(8, 5))

    for name in models.keys():

        subset = results[
            results["Model"] == name
        ]

        plt.plot(
            subset["Threshold"],
            subset[metric],
            marker="o",
            label=name
        )


    plt.title(
        metric + " vs Threshold"
    )

    plt.xlabel(
        "Threshold"
    )

    plt.ylabel(
        metric
    )

    plt.legend()

    plt.grid()

    plt.show()