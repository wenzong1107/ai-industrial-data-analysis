import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data/machine_data.csv")

data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)


plt.figure(figsize=(10, 5))

plt.plot(
    data["timestamp"],
    data["temperature"]
)

plt.title("Temperature Over Time")
plt.xlabel("Time")
plt.ylabel("Temperature")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    data["timestamp"],
    data["vibration"]
)

plt.title("Vibration Over Time")
plt.xlabel("Time")
plt.ylabel("Vibration")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    data["timestamp"],
    data["current"]
)

plt.title("Current Over Time")
plt.xlabel("Time")
plt.ylabel("Current")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()