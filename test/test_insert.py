import os
from pathlib import Path

import pandas as pd
import hopsworks
from dotenv import load_dotenv


# ======================================================
# Configuration
# ======================================================

INPUT_FILE = Path("data/processed/features.csv")

PROJECT_NAME = "areej_aqi_project"
FEATURE_GROUP_NAME = "aqi_features"


# ======================================================
# Load Environment Variables
# ======================================================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env")


# ======================================================
# Load Only 4 Rows
# ======================================================

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"])

# IMPORTANT: Test only
df = df.head(4).copy()

print("Test DataFrame:")
print(df)

print("\nShape:", df.shape)
print("Time dtype:", df["time"].dtype)


# ======================================================
# Connect to Hopsworks
# ======================================================

project = hopsworks.login(
    project=PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()

print("\nConnected to Hopsworks!")
print("Feature Store:", fs.name)


# ======================================================
# Get Feature Group
# ======================================================

fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=1
)

print("Feature Group:", fg.name)
print("Version:", fg.version)
print("Online enabled:", fg.online_enabled)


# ======================================================
# TEST NORMAL INSERT
# ======================================================

print("\nAttempting NORMAL insert of 4 rows...")
print("(No storage='online')")

try:

    result = fg.insert(df)

    print("\n========================================")
    print("INSERT SUCCEEDED!")
    print("========================================")

    print("Result:", result)

except Exception as e:

    print("\n========================================")
    print("INSERT FAILED!")
    print("========================================")

    print("Error type:", type(e).__name__)
    print("Error:", e)