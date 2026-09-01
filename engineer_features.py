import pandas as pd


# ======================================================
# Feature Engineering
# ======================================================

def engineer_features(df):

    # Work on a copy
    df = df.copy()

    # ==================================================
    # Validate Required Columns
    # ==================================================

    required_columns = [
        "city",
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "european_aqi"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ==================================================
    # Convert Time
    # ==================================================

    df["time"] = pd.to_datetime(df["time"])

    # ==================================================
    # Sort Chronologically
    # ==================================================

    df = (
        df
        .sort_values(["city", "time"])
        .reset_index(drop=True)
    )

    # ==================================================
    # Time-Based Features
    # ==================================================

    df["hour"] = df["time"].dt.hour.astype("int64")
    df["day"] = df["time"].dt.day.astype("int64")
    df["month"] = df["time"].dt.month.astype("int64")
    df["weekday"] = df["time"].dt.day_name()

    # ==================================================
    # Derived Features
    # ==================================================

    # AQI Change
    df["aqi_change_rate"] = (
        df.groupby("city")["european_aqi"]
        .diff()
    )

    # Temperature Change
    df["temperature_change"] = (
        df.groupby("city")["temperature_2m"]
        .diff()
    )

    # Humidity Change
    df["humidity_change"] = (
        df.groupby("city")["relative_humidity_2m"]
        .diff()
    )

    # ==================================================
    # Future AQI Targets
    # ==================================================

    # AQI after 24 hours
    df["target_24h"] = (
        df.groupby("city")["european_aqi"]
        .shift(-24)
    )

    # AQI after 48 hours
    df["target_48h"] = (
        df.groupby("city")["european_aqi"]
        .shift(-48)
    )

    # AQI after 72 hours
    df["target_72h"] = (
        df.groupby("city")["european_aqi"]
        .shift(-72)
    )

    # ==================================================
    # Column Order
    # ==================================================

    column_order = [

        # Identification
        "city",
        "time",

        # Time-Based Features
        "hour",
        "day",
        "month",
        "weekday",

        # Weather Features
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "wind_speed_10m",
        "precipitation",
        "cloud_cover",
        "visibility",

        # AQI Features
        "european_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",

        # Engineered Features
        "aqi_change_rate",
        "temperature_change",
        "humidity_change",

        # Targets
        "target_24h",
        "target_48h",
        "target_72h"
    ]

    # Check that all expected columns exist
    missing_output_columns = [
        column for column in column_order
        if column not in df.columns
    ]

    if missing_output_columns:
        raise ValueError(
            f"Missing columns required for final features: "
            f"{missing_output_columns}"
        )

    df = df[column_order]

    # ==================================================
    # Summary
    # ==================================================

    print("\nFeature engineering completed successfully!")
    print("Final feature shape:", df.shape)

    return df