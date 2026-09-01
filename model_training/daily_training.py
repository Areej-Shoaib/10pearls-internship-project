import os
import json
import joblib
import hopsworks
import pandas as pd

from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor

from data_loader import load_feature_data
from preprocessing import preprocess_data
from splitting import SELECTED_FEATURES
from evaluation import evaluate_model


# ======================================================
# Configuration
# ======================================================

PROJECT_NAME = "areej_aqi_project"

ARTIFACT_ROOT = "model_training/artifacts"


# ======================================================
# Model Configuration
# These are the already-selected hyperparameters
# from the original model training phase.
# ======================================================

MODEL_CONFIGS = {

    "AQI_GB_24h": {
        "target": "target_24h",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 2,
    },

    "AQI_GB_48h": {
        "target": "target_48h",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
    },

    "AQI_GB_72h": {
        "target": "target_72h",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
    },
}


# ======================================================
# Main
# ======================================================

def main():

    print("\n" + "=" * 60)
    print("DAILY AQI MODEL TRAINING")
    print("=" * 60)


    # ==================================================
    # STEP 1: Load Feature Store Data
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 1: LOADING FEATURE STORE DATA")
    print("=" * 60)

    df = load_feature_data()

    print(
        f"Total Feature Store rows: {len(df)}"
    )


    # ==================================================
    # STEP 2: Preprocessing
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 60)

    processed_df, targets = preprocess_data(df)

    print(
        f"Processed feature shape: "
        f"{processed_df.shape}"
    )


    # ==================================================
    # STEP 3: Connect to Model Registry
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 3: CONNECTING TO MODEL REGISTRY")
    print("=" * 60)

    load_dotenv()

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY not found in .env"
        )

    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=api_key
    )

    mr = project.get_model_registry()

    print(
        "Connected to Hopsworks Model Registry."
    )


    # ==================================================
    # STEP 4: Daily Training
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 4: DAILY MODEL TRAINING")
    print("=" * 60)


    for model_name, config in MODEL_CONFIGS.items():

        target_name = config["target"]

        print("\n" + "-" * 60)
        print(f"MODEL: {model_name}")
        print(f"TARGET: {target_name}")
        print("-" * 60)


        # ==================================================
        # STEP 4A: Combine Features + Target
        # ==================================================

        data = processed_df.copy()

        # IMPORTANT:
        # Reset both indexes so that feature rows and target
        # rows correspond exactly.
        data = data.reset_index(drop=True)

        target_series = (
            targets[target_name]
            .reset_index(drop=True)
        )

        data[target_name] = target_series


        # ==================================================
        # STEP 4B: Sort Chronologically
        # ==================================================

        data["time"] = pd.to_datetime(
            data["time"]
        )

        data = data.sort_values(
            "time"
        ).reset_index(drop=True)


        # ==================================================
        # STEP 4C: Remove Missing Features
        # ==================================================

        data = data.dropna(
            subset=SELECTED_FEATURES
        )


        # ==================================================
        # STEP 4D: Remove Unavailable Targets
        #
        # New/live rows do not have future targets yet.
        # They must NOT be used for training.
        # ==================================================

        data = data.dropna(
            subset=[target_name]
        ).reset_index(drop=True)


        print(
            f"Rows with available {target_name}: "
            f"{len(data)}"
        )


        # ==================================================
        # Minimum Data Check
        # ==================================================

        if len(data) < 100:

            print(
                f"WARNING: Not enough training data "
                f"for {target_name}."
            )

            print("Skipping model.")

            continue


        # ==================================================
        # STEP 4E: Chronological Split
        #
        # 70% Training
        # 15% Validation
        # 15% Testing
        # ==================================================

        total_rows = len(data)

        train_end = int(
            total_rows * 0.70
        )

        validation_end = int(
            total_rows * 0.85
        )


        train_df = data.iloc[
            :train_end
        ].copy()

        validation_df = data.iloc[
            train_end:validation_end
        ].copy()

        test_df = data.iloc[
            validation_end:
        ].copy()


        X_train = train_df[
            SELECTED_FEATURES
        ].copy()

        X_val = validation_df[
            SELECTED_FEATURES
        ].copy()

        X_test = test_df[
            SELECTED_FEATURES
        ].copy()


        y_train = train_df[
            target_name
        ].copy()

        y_val = validation_df[
            target_name
        ].copy()

        y_test = test_df[
            target_name
        ].copy()


        print(
            f"Training rows:   {len(X_train)}"
        )

        print(
            f"Validation rows: {len(X_val)}"
        )

        print(
            f"Test rows:       {len(X_test)}"
        )


        # ==================================================
        # STEP 5: LOAD EXISTING SCALER
        # ==================================================

        print(
            "\nLoading existing scaler..."
        )

        model_dir = os.path.join(
            ARTIFACT_ROOT,
            model_name
        )

        scaler_path = os.path.join(
            model_dir,
            "scaler.pkl"
        )

        if not os.path.exists(
            scaler_path
        ):

            raise FileNotFoundError(
                f"Existing scaler not found:\n"
                f"{scaler_path}"
            )


        scaler = joblib.load(
            scaler_path
        )

        print(
            "Existing scaler loaded successfully."
        )

        print(
            "Scaler will NOT be refitted."
        )


        # ==================================================
        # STEP 6: Apply Existing Scaler
        # ==================================================

        print(
            "\nApplying existing scaler..."
        )

        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()


        numerical_features = [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "european_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "aqi_change_rate",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
        ]


        X_train_scaled[
            numerical_features
        ] = scaler.transform(
            X_train[numerical_features]
        )

        X_val_scaled[
            numerical_features
        ] = scaler.transform(
            X_val[numerical_features]
        )

        X_test_scaled[
            numerical_features
        ] = scaler.transform(
            X_test[numerical_features]
        )


        print(
            "Feature scaling completed."
        )


        # ==================================================
        # STEP 7: Train Gradient Boosting
        # ==================================================

        print(
            "\nTraining Gradient Boosting model..."
        )

        model = GradientBoostingRegressor(

            n_estimators=config[
                "n_estimators"
            ],

            learning_rate=config[
                "learning_rate"
            ],

            max_depth=config[
                "max_depth"
            ],

            random_state=42
        )


        model.fit(
            X_train_scaled,
            y_train
        )


        print(
            "Model training completed."
        )


        # ==================================================
        # STEP 8: Evaluation
        # ==================================================

        print(
            "\nValidation performance:"
        )

        validation_metrics = evaluate_model(
            model,
            X_val_scaled,
            y_val
        )

        print(
            f"RMSE: "
            f"{validation_metrics['RMSE']:.4f}"
        )

        print(
            f"MAE:  "
            f"{validation_metrics['MAE']:.4f}"
        )

        print(
            f"R²:   "
            f"{validation_metrics['R2']:.4f}"
        )


        print(
            "\nTest performance:"
        )

        test_metrics = evaluate_model(
            model,
            X_test_scaled,
            y_test
        )

        print(
            f"RMSE: "
            f"{test_metrics['RMSE']:.4f}"
        )

        print(
            f"MAE:  "
            f"{test_metrics['MAE']:.4f}"
        )

        print(
            f"R²:   "
            f"{test_metrics['R2']:.4f}"
        )


        # ==================================================
        # STEP 9: Save Updated Artifacts
        # ==================================================

        print(
            "\nSaving updated model artifacts..."
        )


        os.makedirs(
            model_dir,
            exist_ok=True
        )


        # --------------------------------------------------
        # Save NEW model
        # --------------------------------------------------

        joblib.dump(
            model,
            os.path.join(
                model_dir,
                "model.pkl"
            )
        )


        # --------------------------------------------------
        # IMPORTANT:
        # Keep the ORIGINAL scaler.
        # We intentionally do NOT overwrite scaler.pkl.
        # --------------------------------------------------


        # --------------------------------------------------
        # Feature names
        # --------------------------------------------------

        with open(
            os.path.join(
                model_dir,
                "feature_names.json"
            ),
            "w"
        ) as f:

            json.dump(
                SELECTED_FEATURES,
                f,
                indent=4
            )


        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = {

            "model_name":
                model_name,

            "target":
                target_name,

            "forecast_horizon":
                target_name.replace(
                    "target_",
                    ""
                ),

            "framework":
                "scikit-learn",

            "model_type":
                "GradientBoostingRegressor",

            "n_estimators":
                config["n_estimators"],

            "learning_rate":
                config["learning_rate"],

            "max_depth":
                config["max_depth"],

            "scaler":
                "Existing original scaler reused",

            "training_rows":
                len(X_train),

            "validation_rows":
                len(X_val),

            "test_rows":
                len(X_test),

            "validation_rmse":
                validation_metrics["RMSE"],

            "validation_mae":
                validation_metrics["MAE"],

            "validation_r2":
                validation_metrics["R2"],

            "test_rmse":
                test_metrics["RMSE"],

            "test_mae":
                test_metrics["MAE"],

            "test_r2":
                test_metrics["R2"],

            "number_of_features":
                len(SELECTED_FEATURES),
        }


        with open(
            os.path.join(
                model_dir,
                "metadata.json"
            ),
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )


        # ==================================================
        # STEP 10: Register New Model Version
        # ==================================================

        print(
            "\nRegistering new model version..."
        )


        hopsworks_model = (
            mr.sklearn.create_model(

                name=model_name,

                description=(
                    "Daily retrained Gradient "
                    "Boosting AQI forecasting "
                    f"model for {target_name}."
                ),

                metrics={

                    "rmse":
                        test_metrics["RMSE"],

                    "mae":
                        test_metrics["MAE"],

                    "r2":
                        test_metrics["R2"],
                }
            )
        )


        hopsworks_model.save(
            model_dir
        )


        print(
            f"Successfully registered: "
            f"{model_name}"
        )

        print(
            "New model version created."
        )


    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 60)
    print("DAILY MODEL TRAINING COMPLETED")
    print("=" * 60)

    print(
        "\nExisting scaler was reused."
    )

    print(
        "New model versions were registered "
        "for eligible forecast horizons."
    )

    print("=" * 60)


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()