import pandas as pd


# ======================================================
# Convert Weather + AQI Data into Feature DataFrame
# ======================================================

def build_city_features(city, weather_df, aqi_df):

    # Make copies so original data is not modified
    weather_df = weather_df.copy()
    aqi_df = aqi_df.copy()

    # Ensure timestamps have the same type
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    aqi_df["time"] = pd.to_datetime(aqi_df["time"])

    # Merge weather and AQI using timestamp
    merged_df = pd.merge(
        weather_df,
        aqi_df,
        on="time",
        how="inner"
    )

    # Add city identifier
    merged_df["city"] = city

    return merged_df


# ======================================================
# Build Features for All Cities
# ======================================================

def build_all_features(weather_data, aqi_data):

    all_dataframes = []

    for city in weather_data.keys():

        print(f"Building features for {city}...")

        weather_df = weather_data[city]
        city_aqi_df = aqi_data[city]

        city_df = build_city_features(
            city,
            weather_df,
            city_aqi_df
        )

        all_dataframes.append(city_df)

    if not all_dataframes:
        raise ValueError("No city data available to build features.")

    # Combine all cities
    final_df = pd.concat(
        all_dataframes,
        ignore_index=True
    )

    # Put city first
    columns = [
        "city"
    ] + [
        col for col in final_df.columns
        if col != "city"
    ]

    final_df = final_df[columns]

    # Sort chronologically
    final_df = (
        final_df
        .sort_values(["city", "time"])
        .reset_index(drop=True)
    )

    print("\nFeature building completed successfully!")

    print("Shape:", final_df.shape)

    return final_df