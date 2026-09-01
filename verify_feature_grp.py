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
# Connect to Hopsworks
# ======================================================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError(
        "HOPSWORKS_API_KEY not found in .env"
    )


print("\n" + "=" * 60)
print("FEATURE GROUP VERIFICATION")
print("=" * 60)


project = hopsworks.login(
    project=PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()

fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

print("\nConnected to Hopsworks!")
print("Project:", PROJECT_NAME)
print("Feature Store:", fs.name)
print("Feature Group:", fg.name)
print("Version:", fg.version)


# ======================================================
# Read Feature Group
# ======================================================

print("\nReading feature group data...")

df = fg.select_all().read(online=True)

print("\nData loaded successfully.")


# ======================================================
# Basic Dataset Information
# ======================================================

print("\n" + "-" * 60)
print("DATASET INFORMATION")
print("-" * 60)

print("Total records:", len(df))
print("Total columns:", len(df.columns))


# ======================================================
# Timestamp Information
# ======================================================

df["time"] = pd.to_datetime(df["time"])

print("\n" + "-" * 60)
print("TIMESTAMP RANGE")
print("-" * 60)

print("Earliest timestamp:", df["time"].min())
print("Latest timestamp:  ", df["time"].max())


# ======================================================
# Records Per City
# ======================================================

print("\n" + "-" * 60)
print("RECORDS PER CITY")
print("-" * 60)

records_per_city = (
    df.groupby("city")
      .size()
      .sort_index()
)

print(records_per_city)


# ======================================================
# Primary Key Check
# ======================================================

print("\n" + "-" * 60)
print("PRIMARY KEY CHECK")
print("-" * 60)

duplicate_count = df[
    ["city", "time"]
].duplicated().sum()

print("Duplicate city + time records:", duplicate_count)

if duplicate_count == 0:
    print("Primary key check: PASSED")
else:
    print("Primary key check: FAILED")


# ======================================================
# Target Availability
# ======================================================

print("\n" + "-" * 60)
print("TARGET AVAILABILITY")
print("-" * 60)

target_columns = [
    "target_24h",
    "target_48h",
    "target_72h"
]

for column in target_columns:

    if column in df.columns:

        available = df[column].notna().sum()
        missing = df[column].isna().sum()

        print(
            f"{column}: "
            f"{available} available, "
            f"{missing} missing"
        )


# ======================================================
# Missing Values
# ======================================================

print("\n" + "-" * 60)
print("MISSING VALUES")
print("-" * 60)

missing_values = df.isna().sum()

missing_values = missing_values[
    missing_values > 0
]

if missing_values.empty:

    print("No missing values found.")

else:

    print(missing_values)


# ======================================================
# Final Summary
# ======================================================

print("\n" + "=" * 60)
print("VERIFICATION COMPLETED")
print("=" * 60)

print("Total records:", len(df))
print("Cities:", df["city"].nunique())
print("Earliest:", df["time"].min())
print("Latest:", df["time"].max())
print("Duplicate primary keys:", duplicate_count)