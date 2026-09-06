# CityAQI Forecast

An end-to-end machine learning and MLOps application for forecasting Air Quality Index (AQI) across major cities in Pakistan for the next 24, 48, and 72 hours.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Areej-Shoaib/10pearls-internship-project.git
cd 10pearls-internship-project
```

### 2. Create Virtual Environments

Python **3.11** is recommended for local development. The project was developed and tested locally with **Python 3.11.7**.

Two separate virtual environments are used because the main application and the automated data pipeline have different dependency requirements. Keeping them separate prevents package-version conflicts.

#### Main Application Environment

The main local environment is used to run the **CityAQI Forecast application**, including the Streamlit dashboard, live data retrieval, feature preparation, model inference, SHAP explainability, and related application components.

**Windows:**

```bash
python -m venv .venv-dashboard
```

**macOS / Linux:**

```bash
python3 -m venv .venv-dashboard
```

#### Pipeline Environment

A separate environment is used for the **automated feature pipeline and CI/CD workflows**, keeping its pipeline-specific dependencies isolated from the main application environment.

**Windows:**

```bash
python -m venv .venv-pipeline
```

**macOS / Linux:**

```bash
python3 -m venv .venv-pipeline
```

### 3. Install Dependencies

Each environment has its own dependency file.

#### Main Application

Activate `.venv-dashboard`:

**Windows:**

```bash
.venv-dashboard\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv-dashboard/bin/activate
```

Then install:

```bash
pip install -r requirements.txt
```

#### Automated Pipeline

Deactivate the current environment:

```bash
deactivate
```

Activate `.venv-pipeline`:

**Windows:**

```bash
.venv-pipeline\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv-pipeline/bin/activate
```

Then install:

```bash
pip install -r requirements-pipeline.txt
```

Use `.venv-dashboard` for **local development and running the CityAQI Forecast application**. The `.venv-pipeline` environment is dedicated to the **automated data/feature pipeline and its CI/CD execution**.


### 4. Configure Hopsworks

The project uses Hopsworks for its Feature Store and Model Registry.

Create a Hopsworks API key and configure it as an environment variable:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
```

### 5. Activate the Application Environment

Before running the dashboard, activate the `.venv-dashboard` environment.

**Windows:**

```bash
.venv-dashboard\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv-dashboard/bin/activate
```

### 6. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

The dashboard allows users to select a supported city and view current air-quality and weather information, 24-hour, 48-hour, and 72-hour AQI forecasts, prediction explanations, and related insights.

---


# Project Structure

```text
10pearls-internship-project/
│
├── .github/
│   └── workflows/
│       ├── hourly_features.yml
│       └── daily_training.yml
│
├── .streamlit/
│   └── config.toml
│
├── dashboard/
│   ├── app.py
│   └── style.css
│
├── eda_results/
│   ├── *.png
│   ├── *.csv
│   └── eda_report.txt
│
├── model_training/
│   ├── artifacts/
│   ├── data_loader.py
│   ├── daily_training.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── inference.py
│   ├── model_registration.py
│   ├── model_testing.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── scaling.py
│   ├── splitting.py
│   ├── tune_gd.py
│   └── tuning_timeseries.py
│
├── build_features.py
├── data_ingestion.py
├── eda.py
├── engineer_features.py
├── fetch_raw_data.py
├── pipeline.py
│
├── requirements.txt
├── requirements-pipeline.txt
├── .gitignore
└── .python-version
```

### Directory and File Description

| Component                   | Description                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| `.github/workflows/`        | GitHub Actions workflows for automated feature updates and model training                    |
| `.streamlit/`               | Streamlit application configuration                                                          |
| `dashboard/`                | User-facing Streamlit dashboard                                                              |
| `eda_results/`              | Exploratory data analysis charts, statistics, and reports                                    |
| `model_training/`           | Model training, evaluation, inference, explainability, preprocessing, and model registration |
| `fetch_raw_data.py`         | Fetches weather and air-quality data from Open-Meteo                                         |
| `build_features.py`         | Builds the combined feature dataset                                                          |
| `engineer_features.py`      | Performs feature engineering and creates forecasting targets                                 |
| `data_ingestion.py`         | Handles feature ingestion into Hopsworks                                                     |
| `pipeline.py`               | Orchestrates the feature/data pipeline                                                       |
| `eda.py`                    | Performs exploratory data analysis                                                           |
| `requirements.txt`          | Main project dependencies                                                                    |
| `requirements-pipeline.txt` | Dependencies required for CI/CD automation                                                   |

---

# Environment Configuration

CityAQI Forecast uses Hopsworks for feature management and model registration.

Set the following environment variable before running components that require Hopsworks:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
```

The Hopsworks project and Feature Store configuration are defined within the project code.

For security:

* Do not hard-code API keys.
* Do not commit `.env` files.
* Store GitHub Actions credentials using GitHub Secrets.
* Use your own Hopsworks project/API key when reproducing the project.

---

# Running the Data Pipeline

The feature pipeline supports two primary modes.

## Backfill Mode

Backfill mode is used to process historical data and generate the features and future AQI targets required for model training.

```bash
python pipeline.py --mode backfill --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

Example:

```bash
python pipeline.py --mode backfill --start-date 2026-07-01 --end-date 2026-08-01
```

The pipeline retrieves additional future data when necessary so that the 24-hour, 48-hour, and 72-hour forecasting targets can be calculated correctly.

## Hourly Mode

Hourly mode retrieves the latest available weather and air-quality information and updates the feature store.

```bash
python pipeline.py --mode hourly
```

This mode is also used by the automated GitHub Actions workflow.

---

# Running Model Training

The model-training pipeline can be executed with:

```bash
python model_training/daily_training.py
```

The training process retrieves feature data from Hopsworks, performs preprocessing and chronological splitting, trains the forecasting models, evaluates them, and registers the resulting models.

---

# About CityAQI Forecast

**CityAQI Forecast** is an end-to-end machine learning and MLOps system designed to forecast Air Quality Index across major cities in Pakistan.

The system provides AQI forecasts for:

* **24 hours ahead**
* **48 hours ahead**
* **72 hours ahead**

It combines live weather and air-quality data with machine learning, feature engineering, cloud-based feature management, model registration, automated pipelines, explainability, and an interactive Streamlit dashboard.

The project is designed as a complete workflow rather than a standalone prediction script:

```text
Data Collection
      ↓
Feature Engineering
      ↓
Hopsworks Feature Store
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hopsworks Model Registry
      ↓
Live Inference
      ↓
SHAP Explainability
      ↓
Streamlit Dashboard
```

---

# Key Features

* AQI forecasting for the next **24, 48, and 72 hours**
* Support for **12 cities across Pakistan**
* Live weather and air-quality data from Open-Meteo
* Automated feature engineering
* Hopsworks Feature Store integration
* Three horizon-specific Gradient Boosting models
* Hopsworks Model Registry
* Chronological time-series train/validation/test splitting
* SHAP-based local prediction explanations
* Automated hourly feature pipeline
* Automated daily model-training pipeline
* Interactive Streamlit dashboard
* Live model inference
* Current weather and AQI information
* AQI forecast visualization
* Model evaluation using RMSE, MAE, and R²

---

# Supported Cities

CityAQI Forecast currently supports the following cities:

1. Karachi
2. Lahore
3. Islamabad
4. Rawalpindi
5. Faisalabad
6. Multan
7. Peshawar
8. Quetta
9. Hyderabad
10. Gujranwala
11. Sialkot
12. Sukkur

---

# Data Sources

The system uses **Open-Meteo** to retrieve weather and air-quality information.

The collected data includes environmental variables such as:

* Temperature
* Relative humidity
* Atmospheric pressure
* Wind speed
* European AQI
* PM2.5
* PM10
* Carbon monoxide (CO)
* Nitrogen dioxide (NO₂)
* Sulphur dioxide (SO₂)
* Ozone (O₃)

These variables are combined with temporal and derived features before being supplied to the forecasting models.

---

# Feature Engineering

The feature-engineering stage transforms raw environmental data into model-ready features.

### Temporal Features

The system derives temporal information including:

* Hour
* Month
* Weekday

Cyclical encoding is applied to periodic variables such as hour and month:

```text
hour → hour_sin, hour_cos
month → month_sin, month_cos
```

This allows the models to represent the cyclical nature of time.

### Environmental Features

Environmental and pollution-related variables include:

* Temperature
* Relative humidity
* Pressure
* Wind speed
* European AQI
* PM2.5
* PM10
* CO
* NO₂
* SO₂
* O₃
* AQI change rate

### Categorical Encoding

City and weekday information are encoded for use by the machine-learning models.

### Forecasting Targets

For each city, future AQI values are generated as:

```text
target_24h → AQI 24 hours into the future
target_48h → AQI 48 hours into the future
target_72h → AQI 72 hours into the future
```

---

# Machine Learning Models

CityAQI Forecast uses **three separate Gradient Boosting regression models**, with one model dedicated to each forecasting horizon.

| Model        | Forecast Horizon |
| ------------ | ---------------: |
| `AQI_GB_24h` |         24 hours |
| `AQI_GB_48h` |         48 hours |
| `AQI_GB_72h` |         72 hours |

The models use a fixed random state for reproducibility.

---

# Model Training Strategy

Because AQI forecasting is a time-dependent problem, the dataset is split chronologically rather than randomly.

The final split is approximately:

```text
70% → Training
15% → Validation
15% → Testing
```

This approach helps prevent future observations from being randomly introduced into the training data.

The preprocessing and scaling components used during training are also preserved with the model artifacts so that live inference uses the same transformations.

---

# Model Evaluation

The forecasting models are evaluated using **RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² (Coefficient of Determination)** on a chronological test set.

Separate models are trained for **24-hour, 48-hour, and 72-hour AQI forecasting horizons**. Model performance generally decreases as the forecast horizon increases, reflecting the greater uncertainty associated with predicting AQI further into the future.

---

# Model Registry

Trained models are registered in Hopsworks Model Registry.

Each forecasting horizon has its own registered model:

```text
AQI_GB_24h
AQI_GB_48h
AQI_GB_72h
```

Associated model artifacts include the trained model, scaler, feature names, and metadata.

During inference, the application retrieves the registered models and uses the corresponding preprocessing artifacts before generating predictions.

---

# Live Inference

The inference pipeline combines current environmental information with the registered models.

The workflow is:

```text
Selected City
     ↓
Live Weather Data
     +
Live Air-Quality Data
     ↓
Feature Engineering
     ↓
Latest Observation
     ↓
Registered Model Retrieval
     ↓
Feature Compatibility Check
     ↓
Scaling
     ↓
24h / 48h / 72h Predictions
     ↓
SHAP Explanation
     ↓
Prediction Result
```

The inference system also verifies that the features generated during inference match the features expected by the registered model before making a prediction.

---

# Explainability with SHAP

CityAQI Forecast uses **SHAP (SHapley Additive exPlanations)** to provide local explanations for individual predictions.

For each forecasting horizon, SHAP values are calculated for the current input and the most influential features are identified.

The dashboard presents the top contributing features and indicates whether their contribution increases or decreases the predicted AQI.

This provides users with additional context instead of presenting the prediction as an unexplained numerical value.

---

# Automated MLOps Pipelines

GitHub Actions is used to automate recurring pipeline tasks.

## Hourly Feature Pipeline

The hourly workflow runs automatically and executes:

```bash
python pipeline.py --mode hourly
```

Its purpose is to continuously update the feature data using the latest available environmental observations.

## Daily Model Training

The daily workflow executes:

```bash
python model_training/daily_training.py
```

This retrains and registers the forecasting models using the latest available feature data.

The workflows use GitHub Secrets for sensitive credentials such as the Hopsworks API key.

---

# Streamlit Dashboard

The user-facing application is built using Streamlit.

The dashboard provides:

### Current Conditions

* European AQI
* PM2.5
* PM10
* Temperature
* Humidity
* Wind speed
* AQI category

### Forecasts

* 24-hour AQI prediction
* 48-hour AQI prediction
* 72-hour AQI prediction

### Explainability

* Top SHAP features
* Positive/negative feature contribution
* Prediction-specific explanations

The dashboard dynamically runs the inference process and displays the resulting predictions rather than relying only on static prediction values.

---

# Technology Stack

| Category             | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python 3.11.7               |
| Data Source          | Open-Meteo                  |
| Data Processing      | Pandas, NumPy               |
| Machine Learning     | Scikit-learn                |
| Model                | Gradient Boosting Regressor |
| Feature Store        | Hopsworks                   |
| Model Registry       | Hopsworks                   |
| Explainability       | SHAP                        |
| Dashboard            | Streamlit                   |
| Visualization        | Plotly                      |
| Automation           | GitHub Actions              |
| Version Control      | Git / GitHub                |

---

# Exploratory Data Analysis

The repository includes the results of exploratory data analysis in the `eda_results/` directory.

These include:

* AQI distribution
* AQI by city
* AQI over time
* Hourly AQI patterns
* Correlation analysis
* Missing-value analysis
* Feature variability
* Records per city
* Target statistics

The EDA outputs were used to understand the dataset and identify relevant characteristics before model development.

---

# Deployment

The Streamlit dashboard is deployed and publicly accessible.

**Live Application:**
(https://city-aqi-forecast.streamlit.app/)

The deployed application provides the interactive CityAQI Forecast dashboard without requiring users to configure the local development environment.

---

# Limitations

Several factors should be considered when interpreting the forecasts:

* Forecast performance decreases at longer horizons.
* The 72-hour model has substantially lower R² than the 24-hour model.
* Predictions depend on the availability and quality of external weather and air-quality data.
* The system currently focuses on a predefined set of cities.
* Model performance may change as environmental conditions and data distributions change over time.

---

# Future Improvements

Potential future improvements include:

* Increasing the amount of historical training data
* Evaluating additional machine-learning and deep-learning approaches
* Incorporating additional meteorological and environmental variables
* Improving long-horizon forecasting performance
* Adding confidence intervals or prediction uncertainty estimates
* Expanding coverage to additional cities
* Adding historical forecast-vs-actual monitoring
* Introducing model drift monitoring
* Improving automated model selection and retraining
* Adding more advanced time-series forecasting architectures

---

# Project Status

**Status: Completed**

CityAQI Forecast currently includes:

* Data ingestion
* Feature engineering
* Exploratory data analysis
* Hopsworks Feature Store
* Model training
* Model evaluation
* Model Registry
* Live inference
* SHAP explainability
* Automated pipelines
* Streamlit dashboard
* Cloud deployment

---

# Author

**Areej Shoaib**

Bachelor of Computer Science — Artificial Intelligence
NED University of Engineering & Technology

---

## License

This project was developed as part of the **10Pearls Shine Internship Program**.

