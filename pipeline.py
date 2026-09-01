import argparse
from datetime import datetime, timedelta
import pandas as pd

from fetch_raw_data import (
    fetch_historical_data,
    fetch_hourly_data
)

from build_features import build_all_features
from engineer_features import engineer_features
from data_ingestion import ingest_features


# ======================================================
# Keep Latest Available Hour
# ======================================================

def keep_latest_hour(weather_data, aqi_data):

    latest_weather_data = {}
    latest_aqi_data = {}

    for city in weather_data:

        weather_df = weather_data[city].copy()
        aqi_df = aqi_data[city].copy()

        weather_df["time"] = pd.to_datetime(
            weather_df["time"]
        )

        aqi_df["time"] = pd.to_datetime(
            aqi_df["time"]
        )

        # Find latest common timestamp
        common_times = set(
            weather_df["time"]
        ).intersection(
            set(aqi_df["time"])
        )

        if not common_times:
            print(f"{city}: No matching timestamps found.")
            continue

        latest_time = max(common_times)

        # Keep only latest hour
        latest_weather_data[city] = weather_df[
            weather_df["time"] == latest_time
        ].copy()

        latest_aqi_data[city] = aqi_df[
            aqi_df["time"] == latest_time
        ].copy()

        print(
            f"{city}: keeping hour {latest_time}"
        )

    return latest_weather_data, latest_aqi_data

# ======================================================
# Pipeline
# ======================================================

def run_pipeline(start_date, end_date, mode):

    print("\n" + "=" * 60)
    print(f"AQI FEATURE PIPELINE - {mode.upper()}")
    print("=" * 60)

    print(f"Start date: {start_date}")
    print(f"End date:   {end_date}")

    # ==================================================
    # STEP 1: Fetch Raw Data
    # ==================================================

    print("\n" + "-" * 60)
    print("STEP 1: FETCHING WEATHER + AQI DATA")
    print("-" * 60)

    if mode == "backfill":

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Fetch 72 extra hours so future AQI targets
        # can be calculated correctly
        fetch_end_dt = end_dt + timedelta(days=3)

        fetch_start = start_dt.strftime("%Y-%m-%d")
        fetch_end = fetch_end_dt.strftime("%Y-%m-%d")

        weather_data, aqi_data = fetch_historical_data(
            start_date=fetch_start,
            end_date=fetch_end
        )

    elif mode == "hourly":

        weather_data, aqi_data = fetch_hourly_data(
            start_date=start_date,
            end_date=end_date
        )

        weather_data, aqi_data = keep_latest_hour(
            weather_data,
            aqi_data
        )

    else:

        raise ValueError(
            f"Unknown pipeline mode: {mode}"
        )

    # ==================================================
    # STEP 2: Build Features
    # ==================================================

    print("\n" + "-" * 60)
    print("STEP 2: BUILDING FEATURES")
    print("-" * 60)

    merged_df = build_all_features(
        weather_data,
        aqi_data
    )

    print("\nMerged DataFrame:")
    print("Shape:", merged_df.shape)

    # ==================================================
    # STEP 3: Engineer Features
    # ==================================================

    print("\n" + "-" * 60)
    print("STEP 3: ENGINEERING FEATURES")
    print("-" * 60)

    feature_df = engineer_features(
        merged_df
    )

    print("\nFinal Feature DataFrame:")
    print("Shape:", feature_df.shape)

    # ==================================================
    # Target Availability Check
    # ==================================================

    print("\nTarget availability:")

    print(
        "target_24h:",
        feature_df["target_24h"].notna().sum()
    )

    print(
        "target_48h:",
        feature_df["target_48h"].notna().sum()
    )

    print(
        "target_72h:",
        feature_df["target_72h"].notna().sum()
    )


    # ======================================================
    # Remove Look-Ahead Rows
    # ======================================================

    if mode == "backfill":

        feature_df = feature_df[
            (feature_df["time"] >= start_dt) &
            (feature_df["time"] < end_dt)
        ].copy()

        print("\nAfter removing look-ahead rows:")
        print("Shape:", feature_df.shape)

    else:

        print("\nHourly mode: keeping latest available hour.")
        print("Shape:", feature_df.shape)

    
    # ==================================================
    # STEP 4: Ingest into Hopsworks
    # ==================================================

    print("\n" + "-" * 60)
    print("STEP 4: INGESTING INTO HOPSWORKS")
    print("-" * 60)

    ingest_features(
        feature_df
    )

    # ==================================================
    # Complete
    # ==================================================

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(f"Mode: {mode}")
    print(f"Rows processed: {len(feature_df)}")



# ======================================================
# Hourly Date Range
# ======================================================

def get_hourly_range():

    now = datetime.now()

    today = now.strftime("%Y-%m-%d")

    return today, today


# ======================================================
# Command Line Interface
# ======================================================

def main():

    parser = argparse.ArgumentParser(
        description="AQI Feature Pipeline"
    )

    parser.add_argument(
        "--mode",
        choices=["backfill", "hourly"],
        required=True,
        help="Pipeline mode: backfill or hourly"
    )

    parser.add_argument(
        "--start-date",
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        help="End date in YYYY-MM-DD format"
    )

    args = parser.parse_args()

    # ==================================================
    # Backfill Mode
    # ==================================================

    if args.mode == "backfill":

        if not args.start_date or not args.end_date:

            raise ValueError(
                "Backfill mode requires --start-date and --end-date."
            )

        start_date = args.start_date
        end_date = args.end_date

    # ==================================================
    # Hourly Mode
    # ==================================================

    else:

        if args.start_date and args.end_date:

            start_date = args.start_date
            end_date = args.end_date

        else:

            start_date, end_date = get_hourly_range()

    # ==================================================
    # Run Pipeline
    # ==================================================

    run_pipeline(
        start_date=start_date,
        end_date=end_date,
        mode=args.mode
    )


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()

