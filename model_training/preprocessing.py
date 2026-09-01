import numpy as np
import pandas as pd

from data_loader import load_feature_data


# ======================================================
# Feature Preprocessing
# ======================================================

def preprocess_data(df):

    # --------------------------------------------------
    # Create a copy
    # --------------------------------------------------

    data = df.copy()

    # --------------------------------------------------
    # Ensure correct ordering
    # --------------------------------------------------

    data["time"] = pd.to_datetime(data["time"])

    data = data.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Drop unused / low-value features
    # --------------------------------------------------

    columns_to_drop = [
        "visibility",
        "precipitation",
        "cloud_cover",
        "day",
        "temperature_change",
        "humidity_change"
    ]

    data = data.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # --------------------------------------------------
    # Cyclical encoding: hour
    # --------------------------------------------------

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    # --------------------------------------------------
    # Cyclical encoding: month
    # --------------------------------------------------

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    # Original hour/month are no longer required
    data = data.drop(
        columns=["hour", "month"]
    )

    # --------------------------------------------------
    # Encode weekday
    # --------------------------------------------------

    data = pd.get_dummies(
        data,
        columns=["weekday"],
        prefix="weekday",
        dtype=int
    )

    # --------------------------------------------------
    # Encode city
    # --------------------------------------------------

    data = pd.get_dummies(
        data,
        columns=["city"],
        prefix="city",
        dtype=int
    )

    # --------------------------------------------------
    # Separate targets
    # --------------------------------------------------

    targets = {
        "target_24h": data["target_24h"].copy(),
        "target_48h": data["target_48h"].copy(),
        "target_72h": data["target_72h"].copy()
    }

    # Remove targets from feature set
    data = data.drop(
        columns=[
            "target_24h",
            "target_48h",
            "target_72h"
        ]
    )

    return data, targets


# ======================================================
# Test Preprocessing
# ======================================================

if __name__ == "__main__":

    df = load_feature_data()

    processed_df, targets = preprocess_data(df)

    print("\nOriginal shape:")
    print(df.shape)

    print("\nProcessed feature shape:")
    print(processed_df.shape)

    print("\nProcessed columns:")
    print(list(processed_df.columns))

    print("\nTarget shapes:")
    for name, target in targets.items():
        print(f"{name}: {target.shape}")