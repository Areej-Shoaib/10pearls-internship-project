from data_loader import load_feature_data
from preprocessing import preprocess_data
from splitting import split_data
from scaling import scale_features

from models import ( train_ridge, train_random_forest, train_gradient_boosting, tune_gradient_boosting )
from evaluation import evaluate_model


# ======================================================
# Model Training Pipeline Runner
# ======================================================

def main():

    # ==================================================
    # STEP 1: Load Feature Store Data
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 1: LOADING FEATURE STORE DATA")
    print("=" * 60)

    df = load_feature_data()

    print(f"Data loaded successfully: {df.shape}")


    # ==================================================
    # STEP 2: Preprocess Data
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 60)

    processed_df, targets = preprocess_data(df)

    print(
        f"Preprocessing completed: "
        f"{processed_df.shape}"
    )


    # ==================================================
    # STEP 3: Feature Selection + Data Splitting
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 3: FEATURE SELECTION + DATA SPLITTING")
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
    # STEP 4: Feature Scaling
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

    print(f"Scaled training set:   {X_train_scaled.shape}")
    print(f"Scaled validation set: {X_val_scaled.shape}")
    print(f"Scaled test set:       {X_test_scaled.shape}")


    # ==================================================
    # STEP 5: MODEL TRAINING + VALIDATION
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING + VALIDATION")
    print("=" * 60)

    models = {
        "Ridge Regression": train_ridge,
        "Random Forest": train_random_forest,
        "Gradient Boosting": train_gradient_boosting
    }

    validation_results = {}
    trained_models = {}

    for target_name in y_train:

        print("\n" + "-" * 60)
        print(f"TARGET: {target_name}")
        print("-" * 60)

        validation_results[target_name] = {}
        trained_models[target_name] = {}

        y_train_target = y_train[target_name]
        y_val_target = y_val[target_name]

        for model_name, train_function in models.items():

            print(f"\nTraining {model_name}...")

            # Train model using training data
            model = train_function(
                X_train_scaled,
                y_train_target
            )

            # Evaluate on validation data
            metrics = evaluate_model(
                model,
                X_val_scaled,
                y_val_target
            )

            validation_results[target_name][model_name] = metrics
            trained_models[target_name][model_name] = model

            print(
                f"RMSE: {metrics['RMSE']:.4f} | "
                f"MAE: {metrics['MAE']:.4f} | "
                f"R²: {metrics['R2']:.4f}"
            )


    # ==================================================
    # STEP 6: VALIDATION MODEL COMPARISON
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 6: VALIDATION MODEL COMPARISON")
    print("=" * 60)

    for target_name, results in validation_results.items():

        print(f"\n{target_name}")
        print("-" * 60)

        for model_name, metrics in results.items():

            print(
                f"{model_name:<25} "
                f"RMSE={metrics['RMSE']:.4f}  "
                f"MAE={metrics['MAE']:.4f}  "
                f"R²={metrics['R2']:.4f}"
            )


    # ==================================================
    # STEP 7: GRADIENT BOOSTING HYPERPARAMETER TUNING
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 7: GRADIENT BOOSTING HYPERPARAMETER TUNING")
    print("=" * 60)

    tuned_models = {}
    tuned_results = {}

    for target_name in y_train:

        print("\n" + "-" * 60)
        print(f"TUNING: {target_name}")
        print("-" * 60)

        print("Searching for best hyperparameters...")

        tuned_model, best_params = tune_gradient_boosting(
            X_train_scaled,
            y_train[target_name]
        )

        tuned_models[target_name] = tuned_model

        print("\nBest parameters:")
        for parameter, value in best_params.items():
            print(f"{parameter}: {value}")

        # Evaluate tuned model on validation set
        metrics = evaluate_model(
            tuned_model,
            X_val_scaled,
            y_val[target_name]
        )

        tuned_results[target_name] = metrics

        print("\nTuned validation performance:")
        print(
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"R²: {metrics['R2']:.4f}"
        )


    # ==================================================
    # STEP 8: BASELINE VS TUNED GRADIENT BOOSTING
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 8: BASELINE VS TUNED GRADIENT BOOSTING")
    print("=" * 60)

    for target_name in y_train:

        baseline = validation_results[target_name]["Gradient Boosting"]
        tuned = tuned_results[target_name]

        print(f"\n{target_name}")
        print("-" * 60)

        print(
            f"Baseline Gradient Boosting  | "
            f"RMSE={baseline['RMSE']:.4f}  "
            f"MAE={baseline['MAE']:.4f}  "
            f"R²={baseline['R2']:.4f}"
        )

        print(
            f"Tuned Gradient Boosting     | "
            f"RMSE={tuned['RMSE']:.4f}  "
            f"MAE={tuned['MAE']:.4f}  "
            f"R²={tuned['R2']:.4f}"
        )


    # ==================================================
    # Final Summary
    # ==================================================

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print(f"Total original rows: {len(df)}")

    print(f"\nTraining rows:   {len(X_train_scaled)}")
    print(f"Validation rows: {len(X_val_scaled)}")
    print(f"Test rows:       {len(X_test_scaled)}")

    print(
        f"\nNumber of model features: "
        f"{X_train_scaled.shape[1]}"
    )

    print("\nTarget datasets:")

    for target in y_train:
        print(
            f"{target}: "
            f"train={len(y_train[target])}, "
            f"val={len(y_val[target])}, "
            f"test={len(y_test[target])}"
        )

    print("\n" + "=" * 60)
    print("MODEL TRAINING AND VALIDATION COMPLETED")
    print("=" * 60)




# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()

