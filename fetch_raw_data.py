import time
import requests
import pandas as pd

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ======================================================
# Supported Cities
# ======================================================

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
# API URLs
# ======================================================

WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
AQI_API = "https://air-quality-api.open-meteo.com/v1/air-quality"


# ======================================================
# HTTP Session
# ======================================================

def create_session():

    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# ======================================================
# Fetch Weather Data
# ======================================================

def fetch_weather(session, latitude, longitude, start_date, end_date):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover",
            "visibility"
        ]),
        "timezone": "auto"
    }

    response = session.get(
        WEATHER_API,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ======================================================
# Fetch AQI Data
# ======================================================

def fetch_aqi(session, latitude, longitude, start_date, end_date):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "european_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone"
        ]),
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
# Convert API Response to DataFrame
# ======================================================

def weather_to_dataframe(weather_json):

    return pd.DataFrame(weather_json["hourly"])


def aqi_to_dataframe(aqi_json):

    return pd.DataFrame(aqi_json["hourly"])


# ======================================================
# Fetch One City
# ======================================================

def fetch_city_data(
    session,
    city,
    latitude,
    longitude,
    start_date,
    end_date
):

    print(f"\nFetching data for {city}...")

    weather_json = fetch_weather(
        session,
        latitude,
        longitude,
        start_date,
        end_date
    )

    aqi_json = fetch_aqi(
        session,
        latitude,
        longitude,
        start_date,
        end_date
    )

    weather_df = weather_to_dataframe(weather_json)
    aqi_df = aqi_to_dataframe(aqi_json)

    print(
        f"{city}: "
        f"{len(weather_df)} weather rows, "
        f"{len(aqi_df)} AQI rows"
    )

    return weather_df, aqi_df


# ======================================================
# Fetch All Cities
# ======================================================

def fetch_all_cities(
    start_date,
    end_date,
    cities=None
):

    if cities is None:
        cities = CITIES

    session = create_session()

    weather_data = {}
    aqi_data = {}

    for city, (latitude, longitude) in cities.items():

        try:

            weather_df, aqi_df = fetch_city_data(
                session,
                city,
                latitude,
                longitude,
                start_date,
                end_date
            )

            weather_data[city] = weather_df
            aqi_data[city] = aqi_df

        except Exception as e:

            print(f"{city} failed: {e}")

        time.sleep(2)

    print("\nData fetching completed.")

    return weather_data, aqi_data


# ======================================================
# Backfill Helper
# ======================================================

def fetch_historical_data(start_date, end_date):

    print("=" * 60)
    print("HISTORICAL DATA BACKFILL")
    print("=" * 60)

    print(f"Start date: {start_date}")
    print(f"End date:   {end_date}")

    return fetch_all_cities(
        start_date=start_date,
        end_date=end_date
    )



# ======================================================
# Hourly Helper
# ======================================================

def fetch_hourly_data(start_date, end_date):

    print("=" * 60)
    print("HOURLY DATA FETCH")
    print("=" * 60)

    print(f"Start date: {start_date}")
    print(f"End date:   {end_date}")

    return fetch_all_cities(
        start_date=start_date,
        end_date=end_date
    )



# ======================================================
# Live / Current Data (for inference, recently added by cluade)
# ======================================================

FORECAST_WEATHER_API = "https://api.open-meteo.com/v1/forecast"

def fetch_current_weather(session, latitude, longitude):

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

    response = session.get(FORECAST_WEATHER_API, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_current_aqi(session, latitude, longitude):

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



def fetch_current_data(city, latitude, longitude):
    """Fetch recent weather + AQI data for one city."""

    session = create_session()

    weather_json = fetch_current_weather(
        session,
        latitude,
        longitude
    )

    aqi_json = fetch_current_aqi(
        session,
        latitude,
        longitude
    )

    weather_df = weather_to_dataframe(weather_json)
    aqi_df = aqi_to_dataframe(aqi_json)

    weather_df["time"] = pd.to_datetime(
        weather_df["time"]
    )

    aqi_df["time"] = pd.to_datetime(
        aqi_df["time"]
    )

    return weather_df, aqi_df