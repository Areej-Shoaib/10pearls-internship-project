import os

import pandas as pd
import hopsworks
from dotenv import load_dotenv


# ======================================================
# Configuration
# ======================================================

PROJECT_NAME = "areej_aqi_project"
FEATURE_GROUP_NAME = "aqi_features"


# ======================================================
# Connect to Hopsworks
# ======================================================

def get_feature_group():

    load_dotenv()

    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

    if not hopsworks_api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found in .env"
        )

    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=hopsworks_api_key
    )

    fs = project.get_feature_store()

    print("\nConnected to Hopsworks!")
    print("Project:", PROJECT_NAME)
    print("Feature Store:", fs.name)

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=1
    )

    print("\nFeature Group:", fg.name)
    print("Version:", fg.version)
    print("Online enabled:", fg.online_enabled)

    return project, fg


# ======================================================
# Validate Feature Data
# ======================================================

def validate_data(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot insert an empty DataFrame."
        )

    # Make sure time is datetime
    df["time"] = pd.to_datetime(df["time"])

    # Primary key validation
    if df[["city", "time"]].duplicated().any():
        raise ValueError(
            "Duplicate city + time combinations found!"
        )

    print("\nPrimary key validation passed.")
    print("city + time combinations are unique.")

    return df


# ======================================================
# Insert Data into Feature Store
# ======================================================

def ingest_features(df):

    df = validate_data(df)

    project, fg = get_feature_group()

    print("\nFeature data ready for ingestion.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nInserting feature data into Hopsworks...")

    try:

        result = fg.insert(
            df,
            storage="online"
        )

        print("\n========================================")
        print("FEATURE INGESTION SUCCEEDED!")
        print("========================================")

        print("Rows inserted:", len(df))
        print("Result:", result)

        return result

    except Exception as e:

        print("\n========================================")
        print("FEATURE INGESTION FAILED!")
        print("========================================")

        print("Error type:", type(e).__name__)
        print("Error:", e)

        raise