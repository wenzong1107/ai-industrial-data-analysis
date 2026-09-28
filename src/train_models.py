import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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


results = []


for name, model in models.items():

    print()
    print("Training:", name)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


results_df = pd.DataFrame(
    results
)


print()
print("Model Comparison")
print(results_df)