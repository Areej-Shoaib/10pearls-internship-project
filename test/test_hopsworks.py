import hopsworks
import hsfs.engine as engine



import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


project = hopsworks.login(
    project="areej_aqi_project",
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value= HOPSWORKS_API_KEY
)

print("\nConnected to Hopsworks!")
print("Project:", project.name)

fs = project.get_feature_store()

print("Feature Store:", fs.name)

print("\nExecution engine:")
print(engine._get_type())

fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)

print("\nFeature Group:")
print("Name:", fg.name)
print("Version:", fg.version)
print("Primary Key:", fg.primary_key)
print("Time Travel Format:", fg.time_travel_format)
print("Location:", fg.location)




print("\nTesting Feature Group read...")

try:
    df = fg.read()
    print("Read successful!")
    print("Shape:", df.shape)
    print(df.head())
except Exception as e:
    print("READ FAILED")
    print(type(e).__name__)
    print(e)