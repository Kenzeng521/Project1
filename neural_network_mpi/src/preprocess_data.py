#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import json

def preprocess(file_path, output_path, stats_path=None, sample_size=-1):
    print(f"Loading data from {file_path} ...")
    if sample_size is None or sample_size < 0:
        df = pd.read_csv(file_path, low_memory=True)
    else:
        df = pd.read_csv(file_path, nrows=sample_size, low_memory=True)
    print(f"Loaded {len(df)} rows")

    # 自变量（严格按照要求）
    feature_columns = [
        "tpep pickup datetime",
        "tpep dropoff datetime",
        "passenger count",
        "trip distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment type",
        "extra"
    ]
    target_column = "total amount"

    # 保留需要的列并丢弃缺失值
    df = df[feature_columns + [target_column]].dropna()
    print(f"After dropping NA: {len(df)} rows")

    # 转换所有特征为数值型（datetime 会转为时间戳）
    for col in ["tpep pickup datetime", "tpep dropoff datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    print(f"After cleaning: {len(df)} rows")

    # 标准化
    stats = {"feature_columns": feature_columns, "target_column": target_column,
             "feature_means": {}, "feature_stds": {}}

    for col in feature_columns:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0: std = 1.0
        df[col] = (df[col] - mean) / std
        stats["feature_means"][col] = float(mean)
        stats["feature_stds"][col] = float(std)

    target_mean = df[target_column].mean()
    target_std = df[target_column].std()
    if target_std == 0: target_std = 1.0
    stats["target_mean"] = float(target_mean)
    stats["target_std"] = float(target_std)

    df[target_column] = (df[target_column] - target_mean) / target_std

    # 保存
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

    if stats_path:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Normalization stats saved to {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess NYC taxi data")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output preprocessed CSV file")
    parser.add_argument("--stats", default=None, help="Optional: save normalization stats JSON")
    parser.add_argument("--sample-size", type=int, default=-1, help="Number of rows to sample (-1 for full)")
    args = parser.parse_args()

    preprocess(args.input, args.output, args.stats, args.sample_size)

if __name__ == "__main__":
    main()
