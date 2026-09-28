import os

import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv("data/machine_features.csv")

data["timestamp"] = pd.to_datetime(data["timestamp"])

data = data.sort_values("timestamp")


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


model.fit(X_train, y_train)


explainer = shap.Explainer(
    model,
    X_train
)


shap_values = explainer(X_test)


os.makedirs("results", exist_ok=True)


plt.figure()

shap.plots.bar(
    shap_values,
    max_display=10,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure()

shap.plots.beeswarm(
    shap_values,
    max_display=10,
    show=False
)

plt.title("SHAP Feature Impact")

plt.tight_layout()

plt.savefig(
    "results/shap_feature_impact.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("SHAP analysis completed.")

print()
print("Saved files:")

print("results/shap_feature_importance.png")
print("results/shap_feature_impact.png")