import os
import json
import joblib
import hopsworks

from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor

from data_loader import load_feature_data
from preprocessing import preprocess_data
from splitting import split_data
from scaling import scale_features


# ======================================================
# Configuration
# ======================================================

PROJECT_NAME = "areej_aqi_project"

MODEL_CONFIGS = {
    "AQI_GB_24h": {
        "target": "target_24h",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 2,
        "rmse": 11.8631,
        "mae": 8.2559,
        "r2": 0.5723,
    },

    "AQI_GB_48h": {
        "target": "target_48h",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "rmse": 15.2360,
        "mae": 10.7644,
        "r2": 0.2872,
    },

    "AQI_GB_72h": {
        "target": "target_72h",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "rmse": 16.1693,
        "mae": 11.6157,
        "r2": 0.1995,
    },
}


# ======================================================
# Main
# ======================================================

def main():

    # ==================================================
    # STEP 1: Load Data
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 1: LOADING FEATURE STORE DATA")
    print("=" * 60)

    df = load_feature_data()

    print(f"Data loaded successfully: {df.shape}")


    # ==================================================
    # STEP 2: Preprocessing
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 60)

    processed_df, targets = preprocess_data(df)

    print(f"Preprocessed data: {processed_df.shape}")


    # ==================================================
    # STEP 3: Splitting
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 3: DATA SPLITTING")
    print("=" * 60)

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

    print(f"Training set:   {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set:       {X_test.shape}")


    # ==================================================
    # STEP 4: Scaling
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 4: FEATURE SCALING")
    print("=" * 60)

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

    print(f"Scaled training set: {X_train_scaled.shape}")


    # ==================================================
    # STEP 5: Connect to Model Registry
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 5: CONNECTING TO MODEL REGISTRY")
    print("=" * 60)

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

    mr = project.get_model_registry()

    print("Connected to Hopsworks Model Registry.")


    # ==================================================
    # STEP 6: Train + Register Models
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 6: TRAINING AND REGISTERING FINAL MODELS")
    print("=" * 60)


    # Create artifact directory
    artifact_root = "model_training/artifacts"
    os.makedirs(artifact_root, exist_ok=True)


    for model_name, config in MODEL_CONFIGS.items():

        target_name = config["target"]

        print("\n" + "-" * 60)
        print(f"REGISTERING: {model_name}")
        print("-" * 60)

        # ------------------------------------------------
        # Create final model
        # ------------------------------------------------

        model = GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=42
        )

        print("Training final model...")

        model.fit(
            X_train_scaled,
            y_train[target_name]
        )

        print("Model training completed.")


        # ------------------------------------------------
        # Create artifact directory
        # ------------------------------------------------

        model_dir = os.path.join(
            artifact_root,
            model_name
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )


        # ------------------------------------------------
        # Save model
        # ------------------------------------------------

        model_path = os.path.join(
            model_dir,
            "model.pkl"
        )

        joblib.dump(
            model,
            model_path
        )


        # ------------------------------------------------
        # Save scaler
        # ------------------------------------------------

        scaler_path = os.path.join(
            model_dir,
            "scaler.pkl"
        )

        joblib.dump(
            scaler,
            scaler_path
        )


        # ------------------------------------------------
        # Save feature names
        # ------------------------------------------------

        feature_names = list(
            X_train.columns
        )

        feature_path = os.path.join(
            model_dir,
            "feature_names.json"
        )

        with open(
            feature_path,
            "w"
        ) as f:

            json.dump(
                feature_names,
                f,
                indent=4
            )


        # ------------------------------------------------
        # Save metadata
        # ------------------------------------------------

        metadata = {
            "model_name": model_name,
            "target": target_name,
            "forecast_horizon": target_name.replace(
                "target_", ""
            ),
            "framework": "scikit-learn",
            "model_type": "GradientBoostingRegressor",
            "n_estimators": config["n_estimators"],
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"],
            "test_rmse": config["rmse"],
            "test_mae": config["mae"],
            "test_r2": config["r2"],
            "number_of_features": len(feature_names),
        }

        metadata_path = os.path.join(
            model_dir,
            "metadata.json"
        )

        with open(
            metadata_path,
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )


        
        # ------------------------------------------------
        # Create Hopsworks model
        # ------------------------------------------------

        hopsworks_model = mr.sklearn.create_model(
            name=model_name,
            description=(
                f"Gradient Boosting AQI forecasting model "
                f"for {target_name.replace('target_', '')} horizon."
            ),
            metrics={
                "rmse": config["rmse"],
                "mae": config["mae"],
                "r2": config["r2"],
            }
        )




        # ------------------------------------------------
        # Save model to registry
        # ------------------------------------------------

        print("Uploading model to Model Registry...")

        hopsworks_model.save(
            model_dir
        )

        print(f"Successfully registered: {model_name}")


    # ==================================================
    # Final Summary
    # ==================================================

    print("\n" + "=" * 60)
    print("MODEL REGISTRATION COMPLETED")
    print("=" * 60)

    print("\nRegistered models:")

    for model_name in MODEL_CONFIGS:
        print(f"  ✓ {model_name}")

    print("\nModels are now available in the Hopsworks Model Registry.")

    print("\n" + "=" * 60)


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()