import hopsworks
import os
from dotenv import load_dotenv


PROJECT_NAME = "areej_aqi_project"
FEATURE_GROUP_NAME = "aqi_features"


# ======================================================
# Connect to Hopsworks
# ======================================================

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError("HOPSWORKS_API_KEY not found in .env")


project = hopsworks.login(
    project=PROJECT_NAME,
    api_key_value=api_key
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

print("\nFeature Group:", fg.name)
print("Version:", fg.version)


# ======================================================
# Read Data
# ======================================================

print("\nReading feature data...")

df = fg.select_all().read(online=True)

print("\nData successfully read!")

print("Shape:", df.shape)

print("\nLatest rows:")

print(
    df.sort_values("time")
      .tail(15)
      .to_string(index=False)
)


# ======================================================
# Check Primary Keys
# ======================================================

print("\nPrimary key duplicates:")

duplicates = df[
    df[["city", "time"]].duplicated(
        keep=False
    )
]

print(
    "Duplicate rows:",
    len(duplicates)
)


# ======================================================
# Check Latest Timestamp Per City
# ======================================================

print("\nLatest timestamp per city:")

latest = (
    df.groupby("city")["time"]
      .max()
      .sort_index()
)

print(latest)


print("\n========================================")
print("FEATURE STORE VERIFICATION COMPLETED")
print("========================================")