import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 12000

timestamp = pd.date_range(
    start="2026-01-01",
    periods=n_samples,
    freq="min"
)

machine_id = np.array(["M001"] * n_samples)

cycle_length = 1000

cycle_position = np.arange(n_samples) % cycle_length

degradation = cycle_position / cycle_length


temperature = (
    70
    + 8 * degradation
    + np.random.normal(0, 1.5, n_samples)
)

pressure = (
    5
    + 0.3 * degradation
    + np.random.normal(0, 0.15, n_samples)
)

vibration = (
    0.35
    + 0.25 * degradation
    + np.random.normal(0, 0.025, n_samples)
)

current = (
    12
    + 1.5 * degradation
    + np.random.normal(0, 0.3, n_samples)
)


failure_score = (
    0.4 * (temperature - 70) / 8
    + 0.4 * (vibration - 0.35) / 0.25
    + 0.2 * (current - 12) / 1.5
)


failure = (
    failure_score > 0.75
).astype(int)


data = pd.DataFrame({
    "timestamp": timestamp,
    "machine_id": machine_id,
    "temperature": temperature,
    "pressure": pressure,
    "vibration": vibration,
    "current": current,
    "failure": failure
})


print(data.head())

print()

print("Dataset shape:")
print(data.shape)

print()

print("Failure distribution:")
print(data["failure"].value_counts())

print()

print("Failure ratio:")
print(data["failure"].mean())


data.to_csv(
    "data/machine_data.csv",
    index=False
)