import os

import pandas as pd
import hopsworks
from dotenv import load_dotenv


# ======================================================
# Configuration
# ======================================================

PROJECT_NAME = "areej_aqi_project"
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


# ======================================================
# Load Data from Hopsworks Online Feature Store
# ======================================================

def load_feature_data():

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found in .env"
        )

    # Connect to Hopsworks
    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=api_key
    )

    fs = project.get_feature_store()

    # Access Feature Group
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    print("Connected to Hopsworks.")
    print(f"Feature Group: {fg.name}")
    print(f"Version: {fg.version}")

    # Read data from ONLINE store
    df = fg.select_all().read(
        online=True
    )

    # Ensure timestamp is datetime
    df["time"] = pd.to_datetime(df["time"])

    return df


# ======================================================
# Test Loader
# ======================================================

if __name__ == "__main__":

    df = load_feature_data()

    print("\nData loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))