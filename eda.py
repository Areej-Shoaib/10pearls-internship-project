import os
from pathlib import Path

import hopsworks
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv


# ======================================================
# Configuration
# ======================================================

PROJECT_NAME = "areej_aqi_project"
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

RESULTS_DIR = Path("eda_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = RESULTS_DIR / "eda_report.txt"


# ======================================================
# Helper: Write to Report
# ======================================================

def write_report(file, title, content=""):

    file.write("\n")
    file.write("=" * 60 + "\n")
    file.write(title + "\n")
    file.write("=" * 60 + "\n")

    if content:
        file.write(str(content))
        file.write("\n")


# ======================================================
# Load Feature Store Data
# ======================================================

def load_feature_data():

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found in .env"
        )

    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=api_key
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    df = fg.select_all().read(online=True)

    df["time"] = pd.to_datetime(df["time"])

    return df


# ======================================================
# Dataset Overview
# ======================================================

def dataset_overview(df, report):

    write_report(
        report,
        "DATASET INFORMATION"
    )

    report.write(f"Rows: {df.shape[0]}\n")
    report.write(f"Columns: {df.shape[1]}\n")

    report.write("\nColumn Names:\n")

    for column in df.columns:
        report.write(f"- {column}\n")

    write_report(
        report,
        "DATA TYPES",
        df.dtypes.to_string()
    )

    write_report(
        report,
        "DESCRIPTIVE STATISTICS",
        df.describe(include="all").to_string()
    )


# ======================================================
# Time Coverage
# ======================================================

def time_analysis(df, report):

    write_report(
        report,
        "TIME COVERAGE"
    )

    report.write(
        f"Earliest timestamp: {df['time'].min()}\n"
    )

    report.write(
        f"Latest timestamp:   {df['time'].max()}\n"
    )

    report.write(
        f"Total unique timestamps: {df['time'].nunique()}\n"
    )

    report.write(
        f"Total cities: {df['city'].nunique()}\n"
    )


# ======================================================
# Records per City
# ======================================================

def city_analysis(df, report):

    city_counts = (
        df["city"]
        .value_counts()
        .sort_index()
    )

    write_report(
        report,
        "RECORDS PER CITY",
        city_counts.to_string()
    )

    city_counts.to_csv(
        RESULTS_DIR / "records_per_city.csv"
    )


# ======================================================
# Missing Value Analysis
# ======================================================

def missing_value_analysis(df, report):

    missing = df.isnull().sum()

    missing = (
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    write_report(
        report,
        "MISSING VALUE ANALYSIS"
    )

    if missing.empty:

        report.write(
            "No missing values found.\n"
        )

        return

    report.write(
        missing.to_string()
    )

    report.write("\n")

    missing_percentage = (
        (missing / len(df)) * 100
    ).round(2)

    report.write(
        "\nMissing percentage:\n"
    )

    report.write(
        missing_percentage.to_string()
    )

    missing_summary = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": missing_percentage
    })

    missing_summary.to_csv(
        RESULTS_DIR / "missing_values.csv"
    )

    # Missing-value plot

    plt.figure(figsize=(10, 6))

    missing.plot(kind="bar")

    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Number of Missing Values")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "missing_values.png",
        dpi=300
    )

    plt.close()


# ======================================================
# AQI Distribution
# ======================================================

def aqi_distribution(df):

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["european_aqi"].dropna(),
        bins=50,
        kde=True
    )

    plt.title("European AQI Distribution")
    plt.xlabel("European AQI")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "aqi_distribution.png",
        dpi=300
    )

    plt.close()


# ======================================================
# AQI Statistics
# ======================================================

def aqi_statistics(df, report):

    aqi_stats = (
        df["european_aqi"]
        .describe()
    )

    write_report(
        report,
        "EUROPEAN AQI STATISTICS",
        aqi_stats.to_string()
    )


# ======================================================
# Average AQI by City
# ======================================================

def aqi_by_city(df, report):

    city_aqi = (
        df.groupby("city")["european_aqi"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std"
            ]
        )
        .sort_values(
            "mean",
            ascending=False
        )
    )

    write_report(
        report,
        "AQI STATISTICS BY CITY",
        city_aqi.to_string()
    )

    city_aqi.to_csv(
        RESULTS_DIR / "aqi_statistics_by_city.csv"
    )

    # Plot

    plt.figure(figsize=(12, 6))

    city_aqi["mean"].sort_values(
        ascending=True
    ).plot(kind="barh")

    plt.title("Average AQI by City")
    plt.xlabel("Average European AQI")
    plt.ylabel("City")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "aqi_by_city.png",
        dpi=300
    )

    plt.close()


# ======================================================
# AQI Over Time
# ======================================================

def aqi_over_time(df):

    daily_aqi = (
        df.set_index("time")
        .groupby("city")["european_aqi"]
        .resample("D")
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 7))

    for city in daily_aqi["city"].unique():

        city_data = daily_aqi[
            daily_aqi["city"] == city
        ]

        plt.plot(
            city_data["time"],
            city_data["european_aqi"],
            label=city
        )

    plt.title("Daily AQI Trends by City")
    plt.xlabel("Date")
    plt.ylabel("Average European AQI")

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "aqi_over_time.png",
        dpi=300
    )

    plt.close()


# ======================================================
# Hourly AQI Pattern
# ======================================================

def hourly_aqi_analysis(df, report):

    hourly_aqi = (
        df.groupby("hour")["european_aqi"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std"
            ]
        )
    )

    write_report(
        report,
        "AQI BY HOUR OF DAY",
        hourly_aqi.to_string()
    )

    hourly_aqi.to_csv(
        RESULTS_DIR / "hourly_aqi_statistics.csv"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        hourly_aqi.index,
        hourly_aqi["mean"],
        marker="o"
    )

    plt.title("Average AQI by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Average European AQI")

    plt.xticks(range(24))

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "hourly_aqi_pattern.png",
        dpi=300
    )

    plt.close()


# ======================================================
# Correlation Analysis
# ======================================================

def correlation_analysis(df, report):

    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = numeric_df.corr()

    write_report(
        report,
        "CORRELATION MATRIX",
        correlation.to_string()
    )

    correlation.to_csv(
        RESULTS_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "correlation_matrix.png",
        dpi=300
    )

    plt.close()


# ======================================================
# Target Analysis
# ======================================================

def target_analysis(df, report):

    targets = [
        "target_24h",
        "target_48h",
        "target_72h"
    ]

    target_stats = df[targets].describe()

    write_report(
        report,
        "TARGET STATISTICS",
        target_stats.to_string()
    )

    target_stats.to_csv(
        RESULTS_DIR / "target_statistics.csv"
    )

    target_correlation = (
        df[targets].corr()
    )

    write_report(
        report,
        "TARGET CORRELATION",
        target_correlation.to_string()
    )


# ======================================================
# Feature Variability
# ======================================================

def feature_variability(df, report):

    numeric_df = df.select_dtypes(
        include="number"
    )

    variability = pd.DataFrame({
        "unique_values": numeric_df.nunique(),
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max()
    })

    write_report(
        report,
        "NUMERIC FEATURE VARIABILITY",
        variability.to_string()
    )

    variability.to_csv(
        RESULTS_DIR / "feature_variability.csv"
    )


# ======================================================
# Main
# ======================================================

def main():

    df = load_feature_data()

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "AQI EXPLORATORY DATA ANALYSIS REPORT\n"
        )

        report.write(
            f"Feature Group: "
            f"{FEATURE_GROUP_NAME}\n"
        )

        report.write(
            f"Version: "
            f"{FEATURE_GROUP_VERSION}\n"
        )

        dataset_overview(
            df,
            report
        )

        time_analysis(
            df,
            report
        )

        city_analysis(
            df,
            report
        )

        missing_value_analysis(
            df,
            report
        )

        aqi_statistics(
            df,
            report
        )

        aqi_by_city(
            df,
            report
        )

        hourly_aqi_analysis(
            df,
            report
        )

        correlation_analysis(
            df,
            report
        )

        target_analysis(
            df,
            report
        )

        feature_variability(
            df,
            report
        )

        write_report(
            report,
            "EDA COMPLETED",
            "All EDA results and visualizations have been saved "
            "to the eda_results directory."
        )

    # Visualizations

    aqi_distribution(df)

    aqi_over_time(df)


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()

