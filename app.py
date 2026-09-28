import streamlit as st
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier


st.set_page_config(
    page_title="Industrial AI",
    layout="wide"
)


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
y_train = y.iloc[:split_index]


model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


latest = data.iloc[-1]

latest_features = X.iloc[[-1]]

failure_probability = model.predict_proba(
    latest_features
)[0][1]


if failure_probability >= 0.8:
    risk_level = "HIGH RISK"

elif failure_probability >= 0.5:
    risk_level = "MEDIUM RISK"

else:
    risk_level = "LOW RISK"


st.title(
    "Industrial AI Predictive Maintenance"
)

st.write(
    "AI-based equipment failure risk monitoring system"
)


st.divider()


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Failure Probability",
        f"{failure_probability * 100:.2f}%"
    )


with col2:

    st.metric(
        "Risk Level",
        risk_level
    )


st.divider()


st.subheader(
    "Current Sensor Values"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Temperature",
        f"{latest['temperature']:.2f}"
    )


with col2:

    st.metric(
        "Pressure",
        f"{latest['pressure']:.2f}"
    )


with col3:

    st.metric(
        "Vibration",
        f"{latest['vibration']:.3f}"
    )


with col4:

    st.metric(
        "Current",
        f"{latest['current']:.2f}"
    )


st.divider()


st.subheader(
    "Sensor Trends"
)


chart_data = data.set_index(
    "timestamp"
)[
    [
        "temperature",
        "vibration",
        "current"
    ]
]


st.line_chart(
    chart_data
)


st.divider()


st.subheader(
    "Recent Sensor Data"
)


st.dataframe(
    data.tail(20)[
        [
            "timestamp",
            "machine_id",
            "temperature",
            "pressure",
            "vibration",
            "current",
            "failure"
        ]
    ]
)