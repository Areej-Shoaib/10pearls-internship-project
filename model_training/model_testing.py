from sklearn.ensemble import GradientBoostingRegressor

from data_loader import load_feature_data
from preprocessing import preprocess_data
from splitting import split_data
from scaling import scale_features

from evaluation import evaluate_model


# ======================================================
# Final Test Evaluation
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

    print(f"Preprocessing completed: {processed_df.shape}")


    # ==================================================
    # STEP 3: Split Data
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
    # STEP 5: Train Final Selected Models
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 5: TRAINING FINAL SELECTED MODELS")
    print("=" * 60)


    # --------------------------------------------------
    # 24h Tuned Gradient Boosting
    # --------------------------------------------------

    print("\nTraining tuned Gradient Boosting for target_24h...")

    model_24h = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=2,
        random_state=42
    )

    model_24h.fit(
        X_train_scaled,
        y_train["target_24h"]
    )


    # --------------------------------------------------
    # 48h Baseline Gradient Boosting
    # --------------------------------------------------

    print("Training baseline Gradient Boosting for target_48h...")

    model_48h = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model_48h.fit(
        X_train_scaled,
        y_train["target_48h"]
    )


    # --------------------------------------------------
    # 72h Baseline Gradient Boosting
    # --------------------------------------------------

    print("Training baseline Gradient Boosting for target_72h...")

    model_72h = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model_72h.fit(
        X_train_scaled,
        y_train["target_72h"]
    )


    # ==================================================
    # STEP 6: Final Test Evaluation
    # ==================================================

    print("\n" + "=" * 60)
    print("STEP 6: FINAL TEST-SET EVALUATION")
    print("=" * 60)


    # --------------------------------------------------
    # 24h
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("TARGET: target_24h")
    print("-" * 60)

    results_24h = evaluate_model(
        model_24h,
        X_test_scaled,
        y_test["target_24h"]
    )

    print(
        f"RMSE: {results_24h['RMSE']:.4f} | "
        f"MAE: {results_24h['MAE']:.4f} | "
        f"R²: {results_24h['R2']:.4f}"
    )


    # --------------------------------------------------
    # 48h
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("TARGET: target_48h")
    print("-" * 60)

    results_48h = evaluate_model(
        model_48h,
        X_test_scaled,
        y_test["target_48h"]
    )

    print(
        f"RMSE: {results_48h['RMSE']:.4f} | "
        f"MAE: {results_48h['MAE']:.4f} | "
        f"R²: {results_48h['R2']:.4f}"
    )


    # --------------------------------------------------
    # 72h
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("TARGET: target_72h")
    print("-" * 60)

    results_72h = evaluate_model(
        model_72h,
        X_test_scaled,
        y_test["target_72h"]
    )

    print(
        f"RMSE: {results_72h['RMSE']:.4f} | "
        f"MAE: {results_72h['MAE']:.4f} | "
        f"R²: {results_72h['R2']:.4f}"
    )


    # ==================================================
    # STEP 7: Final Results Summary
    # ==================================================

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print("\nTarget      RMSE       MAE        R²")
    print("-" * 60)

    print(
        f"24h      "
        f"{results_24h['RMSE']:.4f}     "
        f"{results_24h['MAE']:.4f}     "
        f"{results_24h['R2']:.4f}"
    )

    print(
        f"48h      "
        f"{results_48h['RMSE']:.4f}     "
        f"{results_48h['MAE']:.4f}     "
        f"{results_48h['R2']:.4f}"
    )

    print(
        f"72h      "
        f"{results_72h['RMSE']:.4f}     "
        f"{results_72h['MAE']:.4f}     "
        f"{results_72h['R2']:.4f}"
    )


    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 60)


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()