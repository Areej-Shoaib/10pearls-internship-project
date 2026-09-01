import pandas as pd
import hopsworks

# Load only 2 rows
df = pd.read_csv(
    "data/processed/features.csv",
    parse_dates=["time"]
).head(2)

print("Test DataFrame:")
print(df)
print("\nTime dtype:", df["time"].dtype)

# Connect
project = hopsworks.login(
    project="areej_aqi_project",
    api_key_value="dANIuAo1dY8M7w67.CQ6LbSi6XJ1avmSrJzGnjHQLSTdJ61YzjxVO3XohDAWI70wEUUh8ugnJwQ3FqbLz",
    engine="python"
)

fs = project.get_feature_store()

print("\nConnected to Hopsworks!")
print("Feature Store:", fs.name)

# Get existing Feature Group
fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)

print("Feature Group:", fg.name)
print("Version:", fg.version)
print("Online enabled:", fg.online_enabled)

# IMPORTANT: test ONLINE storage only
print("\nAttempting ONLINE-only insert...")

try:
    job, validation = fg.insert(
        df,
        storage="online",
        wait=True
    )

    print("\n========================================")
    print("ONLINE INSERT SUCCEEDED!")
    print("========================================")

    print("Job:", job)
    print("Validation:", validation)

except Exception as e:
    print("\n========================================")
    print("ONLINE INSERT FAILED!")
    print("========================================")

    print("Error type:", type(e).__name__)
    print("Error:", e)