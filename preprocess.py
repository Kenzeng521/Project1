import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("nytaxi2022.csv")

cols = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", 
    "passenger_count", "trip_distance", "RatecodeID", 
    "PULocationID", "DOLocationID", "payment_type", "extra","total_amount"
]
df = df[cols]

df = df.dropna(axis=0)

df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors='coerce')
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors='coerce')

df = df.dropna(axis=0)

df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

df = df[(df['total_amount'] >= 0) & (df['total_amount'] <= 1000)]

numeric_cols = ["passenger_count", "trip_distance", "extra", 
                "total_amount", "trip_duration"]

for col in numeric_cols:
    df = df[df[col] >= 0]


df = df.drop_duplicates()


df = df.drop(["tpep_pickup_datetime", "tpep_dropoff_datetime"], axis=1)

num_cols = ["passenger_count", "trip_distance", "extra", "trip_duration",
            "RatecodeID", "PULocationID", "DOLocationID", "payment_type"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.to_csv("process3_data.csv", index=False)