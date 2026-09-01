
import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
import hopsworks

from datetime import datetime
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ======================================================
# Configuration
# ======================================================

PROJECT_NAME = "areej_aqi_project"

MODEL_NAMES = {
    "24h": "AQI_GB_24h",
    "48h": "AQI_GB_48h",
    "72h": "AQI_GB_72h"
}

CITIES = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Islamabad": (33.6844, 73.0479),
    "Rawalpindi": (33.5651, 73.0169),
    "Faisalabad": (31.4504, 73.1350),
    "Multan": (30.1575, 71.5249),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Hyderabad": (25.3960, 68.3578),
    "Gujranwala": (32.1877, 74.1945),
    "Sialkot": (32.4945, 74.5229),
    "Sukkur": (27.7052, 68.8574)
}


# ======================================================
# API
# ======================================================

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
AQI_API = "https://air-quality-api.open-meteo.com/v1/air-quality"


# ======================================================
# Features Used During Training
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


# ======================================================
# Numerical Features
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
# HTTP Session
# ======================================================

def create_session():

    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter
    )

    session.mount(
        "http://",
        adapter
    )

    return session


# ======================================================
# Fetch Current Weather
# ======================================================

def fetch_current_weather(
    session,
    latitude,
    longitude
):

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover",
            "visibility"
        ]),

        "past_hours": 6,
        "forecast_hours": 1,

        "timezone": "auto"
    }

    response = session.get(
        FORECAST_API,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ======================================================
# Fetch Recent AQI
# ======================================================

def fetch_recent_aqi(
    session,
    latitude,
    longitude
):

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": ",".join([
            "european_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone"
        ]),

        "past_hours": 6,
        "forecast_hours": 1,

        "timezone": "auto"
    }

    response = session.get(
        AQI_API,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ======================================================
# Prepare Live Data
# ======================================================

def fetch_live_data(city):

    if city not in CITIES:
        raise ValueError(
            f"Unsupported city: {city}"
        )

    latitude, longitude = CITIES[city]

    print(f"\nFetching live data for {city}...")

    session = create_session()

    weather_json = fetch_current_weather(
        session,
        latitude,
        longitude
    )

    aqi_json = fetch_recent_aqi(
        session,
        latitude,
        longitude
    )

    weather_df = pd.DataFrame(
        weather_json["hourly"]
    )

    aqi_df = pd.DataFrame(
        aqi_json["hourly"]
    )

    weather_df["time"] = pd.to_datetime(
        weather_df["time"]
    )

    aqi_df["time"] = pd.to_datetime(
        aqi_df["time"]
    )

    # Merge weather and AQI
    df = pd.merge(
        weather_df,
        aqi_df,
        on="time",
        how="inner"
    )

    if df.empty:
        raise ValueError(
            "No matching weather/AQI timestamps found."
        )

    df["city"] = city

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    print(
        f"Live data received: {len(df)} rows"
    )

    return df


# ======================================================
# Engineer Live Features
# ======================================================

def prepare_features(df, city):

    data = df.copy()

    # --------------------------------------------------
    # Time features
    # --------------------------------------------------

    data["hour"] = data["time"].dt.hour
    data["month"] = data["time"].dt.month
    data["weekday"] = data["time"].dt.day_name()

    # --------------------------------------------------
    # AQI change rate
    # --------------------------------------------------

    data["aqi_change_rate"] = (
        data["european_aqi"].diff()
    )

    # --------------------------------------------------
    # Cyclical hour encoding
    # --------------------------------------------------

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    # --------------------------------------------------
    # Cyclical month encoding
    # --------------------------------------------------

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    # --------------------------------------------------
    # Weekday one-hot encoding
    # --------------------------------------------------

    data = pd.get_dummies(
        data,
        columns=["weekday"],
        prefix="weekday",
        dtype=int
    )

    # --------------------------------------------------
    # City one-hot encoding
    # --------------------------------------------------

    data = pd.get_dummies(
        data,
        columns=["city"],
        prefix="city",
        dtype=int
    )

    # --------------------------------------------------
    # Ensure ALL training columns exist
    # --------------------------------------------------

    for column in SELECTED_FEATURES:

        if column not in data.columns:
            data[column] = 0

    # --------------------------------------------------
    # Select exact feature order
    # --------------------------------------------------

    features = data[
        SELECTED_FEATURES
    ].copy()

    # --------------------------------------------------
    # Handle missing numerical values
    # --------------------------------------------------

    if features[NUMERICAL_FEATURES].isna().any().any():

        print(
            "\nWarning: Missing numerical values detected."
        )

        features[
            NUMERICAL_FEATURES
        ] = features[
            NUMERICAL_FEATURES
        ].ffill().bfill()

    # --------------------------------------------------
    # Select latest observation
    # --------------------------------------------------

    latest_features = features.iloc[
        [-1]
    ].copy()

    latest_raw = data.iloc[
        [-1]
    ].copy()

    return latest_features, latest_raw


# ======================================================
# Connect to Hopsworks Model Registry
# ======================================================

def get_model_registry():

    load_dotenv()

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found in .env"
        )

    print(
        "\nConnecting to Hopsworks..."
    )

    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=api_key
    )

    mr = project.get_model_registry()

    print(
        "Connected to Hopsworks Model Registry."
    )

    return mr


# ======================================================
# Load Registered Model
# ======================================================

def load_registered_model(
    mr,
    model_name
):

    print(
        f"\nLoading model: {model_name}"
    )

    # Get latest version
    model = mr.get_model(
        name=model_name
    )

    latest_version = model.version

    print(
        f"Latest version: {latest_version}"
    )

    # Download model artifacts
    model_dir = model.download()

    print(
        f"Model downloaded to: {model_dir}"
    )

    # Load model
    model_path = os.path.join(
        model_dir,
        "model.pkl"
    )

    scaler_path = os.path.join(
        model_dir,
        "scaler.pkl"
    )

    feature_path = os.path.join(
        model_dir,
        "feature_names.json"
    )

    metadata_path = os.path.join(
        model_dir,
        "metadata.json"
    )

    trained_model = joblib.load(
        model_path
    )

    scaler = joblib.load(
        scaler_path
    )

    with open(
        feature_path,
        "r"
    ) as f:

        feature_names = json.load(f)

    with open(
        metadata_path,
        "r"
    ) as f:

        metadata = json.load(f)

    return (
        trained_model,
        scaler,
        feature_names,
        metadata
    )


# ======================================================
# Scale Features
# ======================================================

def scale_input(
    features,
    scaler
):

    scaled_features = features.copy()

    scaled_features[
        NUMERICAL_FEATURES
    ] = scaler.transform(
        scaled_features[
            NUMERICAL_FEATURES
        ]
    )

    return scaled_features


# ======================================================
# Run Inference
# ======================================================

def predict_aqi(city):

    print("\n" + "=" * 60)
    print("AQI MODEL INFERENCE")
    print("=" * 60)

    print(f"City: {city}")

    # --------------------------------------------------
    # Fetch live data
    # --------------------------------------------------

    live_df = fetch_live_data(
        city
    )

    # --------------------------------------------------
    # Prepare features
    # --------------------------------------------------

    features, latest_raw = prepare_features(
        live_df,
        city
    )

    print(
        "\nLatest observation:"
    )

    print(
        "Time:",
        latest_raw["time"].iloc[0]
    )

    print(
        "AQI:",
        latest_raw["european_aqi"].iloc[0]
    )

    print(
        "PM2.5:",
        latest_raw["pm2_5"].iloc[0]
    )

    print(
        "Temperature:",
        latest_raw["temperature_2m"].iloc[0]
    )

    print(
        "Humidity:",
        latest_raw["relative_humidity_2m"].iloc[0]
    )

    # --------------------------------------------------
    # Connect to registry
    # --------------------------------------------------

    mr = get_model_registry()

    predictions = {}

    # --------------------------------------------------
    # Predict for each horizon
    # --------------------------------------------------

    for horizon, model_name in MODEL_NAMES.items():

        print(
            f"\nRunning {horizon} prediction..."
        )

        (
            model,
            scaler,
            feature_names,
            metadata
        ) = load_registered_model(
            mr,
            model_name
        )

        # Verify feature compatibility
        if feature_names != SELECTED_FEATURES:

            raise ValueError(
                f"Feature mismatch for {model_name}."
            )

        # Scale
        scaled_features = scale_input(
            features,
            scaler
        )

        # Prediction
        prediction = model.predict(
            scaled_features
        )[0]

        predictions[horizon] = float(
            prediction
        )

    # --------------------------------------------------
    # Display predictions
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AQI PREDICTIONS")
    print("=" * 60)

    print(
        f"24-hour AQI: {predictions['24h']:.2f}"
    )

    print(
        f"48-hour AQI: {predictions['48h']:.2f}"
    )

    print(
        f"72-hour AQI: {predictions['72h']:.2f}"
    )

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return {
        "city": city,
        "timestamp": latest_raw["time"].iloc[0],
        "current_aqi": float(
            latest_raw[
                "european_aqi"
            ].iloc[0]
        ),
        "predictions": predictions
    }


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    # Change this city to test another city
    CITY = "Karachi"

    result = predict_aqi(
        CITY
    )

