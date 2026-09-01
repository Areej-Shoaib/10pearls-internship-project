import os
import json
import joblib
import hopsworks
import shap
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_feature_data
from preprocessing import preprocess_data
from splitting import split_data
from scaling import scale_features


# ======================================================
# Configuration
# ======================================================

PROJECT_NAME = "areej_aqi_project"

MODEL_NAMES = {
    "target_24h": "AQI_GB_24h",
    "target_48h": "AQI_GB_48h",
    "target_72h": "AQI_GB_72h"
}

OUTPUT_DIR = "model_training/explainability_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================================================
# Connect to Hopsworks Model Registry
# ======================================================

def connect_to_registry():

    project = hopsworks.login()

    mr = project.get_model_registry()

    print("Connected to Hopsworks Model Registry.")

    return mr


# ======================================================
# Load Registered Model
# ======================================================

def load_registered_model(mr, model_name):

    print(f"\nLoading registered model: {model_name}")

    model = mr.get_models(model_name)[0]

    model_dir = model.download()

    model_path = os.path.join(
        model_dir,
        "model.pkl"
    )

    trained_model = joblib.load(model_path)

    print("Model loaded successfully.")

    return trained_model


# ======================================================
# Generate SHAP Explanation
# ======================================================

def explain_model(
    model,
    X_test,
    feature_names,
    target_name
):

    print(
        f"\nGenerating SHAP explanation for "
        f"{target_name}..."
    )

    # Use a representative sample to keep
    # SHAP computation efficient.
    sample_size = min(2000, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    # TreeExplainer is appropriate for
    # Gradient Boosting models.
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_sample
    )

    # --------------------------------------------------
    # SHAP Summary Plot
    # --------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )

    plt.title(
        f"SHAP Feature Importance - {target_name}"
    )

    plot_path = os.path.join(
        OUTPUT_DIR,
        f"{target_name}_shap_summary.png"
    )

    plt.tight_layout()
    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"SHAP summary plot saved: {plot_path}"
    )

    # --------------------------------------------------
    # Global Feature Importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": (
            abs(shap_values).mean(axis=0)
        )
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{target_name}_feature_importance.csv"
    )

    importance.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Feature importance saved: {csv_path}"
    )

    # --------------------------------------------------
    # Display Top Features
    # --------------------------------------------------

    print("\nTop 10 features:")

    print(
        importance.head(10).to_string(
            index=False
        )
    )


# ======================================================
# Main
# ======================================================

def main():

    print("\n" + "=" * 60)
    print("SHAP MODEL EXPLAINABILITY")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 1: Load Feature Store Data
    # --------------------------------------------------

    print("\nSTEP 1: LOADING DATA")

    df = load_feature_data()

    print(
        f"Data loaded: {df.shape}"
    )

    # --------------------------------------------------
    # STEP 2: Preprocessing
    # --------------------------------------------------

    print("\nSTEP 2: PREPROCESSING")

    processed_df, targets = preprocess_data(df)

    print(
        f"Processed data: {processed_df.shape}"
    )

    # --------------------------------------------------
    # STEP 3: Splitting
    # --------------------------------------------------

    print("\nSTEP 3: DATA SPLITTING")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(
        processed_df,
        targets
    )

    # --------------------------------------------------
    # STEP 4: Scaling
    # --------------------------------------------------

    print("\nSTEP 4: FEATURE SCALING")

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    print(
        f"Test data ready: {X_test_scaled.shape}"
    )

    # --------------------------------------------------
    # Feature Names
    # --------------------------------------------------

    feature_names = list(
        X_train_scaled.columns
    )

    # --------------------------------------------------
    # STEP 5: Model Registry
    # --------------------------------------------------

    print("\nSTEP 5: CONNECTING TO MODEL REGISTRY")

    mr = connect_to_registry()

    # --------------------------------------------------
    # STEP 6: Explain Final Models
    # --------------------------------------------------

    print("\nSTEP 6: GENERATING SHAP EXPLANATIONS")

    for target_name, model_name in MODEL_NAMES.items():

        print("\n" + "-" * 60)
        print(f"TARGET: {target_name}")
        print(f"MODEL: {model_name}")
        print("-" * 60)

        model = load_registered_model(
            mr,
            model_name
        )

        explain_model(
            model,
            X_test_scaled,
            feature_names,
            target_name
        )

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETED")
    print("=" * 60)

    print(
        f"\nResults saved in: {OUTPUT_DIR}"
    )


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()

