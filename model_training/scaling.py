import pandas as pd

from sklearn.preprocessing import StandardScaler


# ======================================================
# Columns to Scale
# ======================================================

NUMERICAL_FEATURES = [
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
    "month_cos"
]


# ======================================================
# Scale Features
# ======================================================

def scale_features(X_train, X_val, X_test):

    # --------------------------------------------------
    # Create copies
    # --------------------------------------------------

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # --------------------------------------------------
    # Initialize scaler
    # --------------------------------------------------

    scaler = StandardScaler()

    # --------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------

    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(
        X_train[NUMERICAL_FEATURES]
    )

    # --------------------------------------------------
    # Transform validation and test using
    # the training-fitted scaler
    # --------------------------------------------------

    X_val_scaled[NUMERICAL_FEATURES] = scaler.transform(
        X_val[NUMERICAL_FEATURES]
    )

    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(
        X_test[NUMERICAL_FEATURES]
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    from data_loader import load_feature_data
    from preprocessing import preprocess_data
    from splitting import split_data

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

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    print("\nScaling completed.")

    print("\nTrain shape:", X_train_scaled.shape)
    print("Validation shape:", X_val_scaled.shape)
    print("Test shape:", X_test_scaled.shape)

    print("\nScaled numerical feature statistics:")
    print(
        X_train_scaled[NUMERICAL_FEATURES]
        .describe()
        .loc[["mean", "std"]]
    )

    print("\nScaler fitted successfully on training data.")