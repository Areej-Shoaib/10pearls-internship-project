import pandas as pd


# ======================================================
# Feature Selection
# ======================================================

SELECTED_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",

    "european_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    "aqi_change_rate",

    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    "weekday_Friday",
    "weekday_Monday",
    "weekday_Saturday",
    "weekday_Sunday",
    "weekday_Thursday",
    "weekday_Tuesday",
    "weekday_Wednesday",

    "city_Faisalabad",
    "city_Gujranwala",
    "city_Hyderabad",
    "city_Islamabad",
    "city_Karachi",
    "city_Lahore",
    "city_Multan",
    "city_Peshawar",
    "city_Quetta",
    "city_Rawalpindi",
    "city_Sialkot",
    "city_Sukkur"
]


TARGETS = [
    "target_24h",
    "target_48h",
    "target_72h"
]


# ======================================================
# Chronological Train / Validation / Test Split
# ======================================================

def split_data(processed_df, targets):

    data = processed_df.copy()

    # --------------------------------------------------
    # Keep time for chronological splitting
    # --------------------------------------------------

    data["time"] = pd.to_datetime(data["time"])

    # Combine features and targets temporarily
    for target_name in TARGETS:
        data[target_name] = targets[target_name].values

    # Sort chronologically
    data = data.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------
    # Remove rows with missing model inputs
    # --------------------------------------------------

    data = data.dropna(
        subset=SELECTED_FEATURES
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Chronological split
    #
    # 70% Training
    # 15% Validation
    # 15% Testing
    # --------------------------------------------------

    total_rows = len(data)

    train_end = int(total_rows * 0.70)
    validation_end = int(total_rows * 0.85)

    train_df = data.iloc[:train_end].copy()
    validation_df = data.iloc[
        train_end:validation_end
    ].copy()
    test_df = data.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------
    # Feature matrices
    # --------------------------------------------------

    X_train = train_df[SELECTED_FEATURES].copy()
    X_val = validation_df[SELECTED_FEATURES].copy()
    X_test = test_df[SELECTED_FEATURES].copy()

    # --------------------------------------------------
    # Targets
    # --------------------------------------------------

    y_train = {
        target: train_df[target].copy()
        for target in TARGETS
    }

    y_val = {
        target: validation_df[target].copy()
        for target in TARGETS
    }

    y_test = {
        target: test_df[target].copy()
        for target in TARGETS
    }

    # --------------------------------------------------
    # Return everything
    # --------------------------------------------------

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    from data_loader import load_feature_data
    from preprocessing import preprocess_data

    df = load_feature_data()

    processed_df, targets = preprocess_data(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(
        processed_df,
        targets
    )

    print("\nTRAINING SET")
    print("X:", X_train.shape)

    print("\nVALIDATION SET")
    print("X:", X_val.shape)

    print("\nTEST SET")
    print("X:", X_test.shape)

    print("\nTARGET SHAPES")

    for target in TARGETS:
        print(
            target,
            "train:", y_train[target].shape,
            "val:", y_val[target].shape,
            "test:", y_test[target].shape
        )