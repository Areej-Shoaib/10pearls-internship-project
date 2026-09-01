import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from data_loader import load_feature_data
from preprocessing import preprocess_data
from splitting import split_data
from scaling import scale_features
from evaluation import evaluate_model


# ======================================================
# Gradient Boosting Hyperparameter Tuning
# ======================================================

def tune_model(X_train, y_train):
    """
    Tune Gradient Boosting using a small hyperparameter grid.
    """

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3]
    }

    model = GradientBoostingRegressor(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_
    )


# ======================================================
# Main
# ======================================================

def main():

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING HYPERPARAMETER TUNING")
    print("=" * 60)


    # ==================================================
    # STEP 1: Load data
    # ==================================================

    print("\nLoading data...")

    df = load_feature_data()

    print(f"Data loaded: {df.shape}")


    # ==================================================
    # STEP 2: Preprocess
    # ==================================================

    print("\nPreprocessing data...")

    processed_df, targets = preprocess_data(df)

    print(f"Processed data: {processed_df.shape}")


    # ==================================================
    # STEP 3: Split
    # ==================================================

    print("\nSplitting data...")

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
    # STEP 4: Scale
    # ==================================================

    print("\nScaling features...")

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

    print("Scaling completed.")


    # ==================================================
    # STEP 5: Tune Gradient Boosting
    # ==================================================

    print("\n" + "=" * 60)
    print("STARTING HYPERPARAMETER SEARCH")
    print("=" * 60)

    tuned_models = {}
    tuned_results = {}

    for target_name in y_train:

        print("\n" + "-" * 60)
        print(f"TARGET: {target_name}")
        print("-" * 60)

        print("\nSearching hyperparameters...")

        (
            best_model,
            best_params,
            best_cv_score
        ) = tune_model(
            X_train_scaled,
            y_train[target_name]
        )

        tuned_models[target_name] = best_model

        print("\nBest parameters:")

        for parameter, value in best_params.items():
            print(f"{parameter}: {value}")

        print(
            f"\nBest CV RMSE: "
            f"{-best_cv_score:.4f}"
        )


        # ==================================================
        # Validation evaluation
        # ==================================================

        validation_metrics = evaluate_model(
            best_model,
            X_val_scaled,
            y_val[target_name]
        )

        tuned_results[target_name] = validation_metrics

        print("\nValidation performance:")

        print(
            f"RMSE: {validation_metrics['RMSE']:.4f} | "
            f"MAE: {validation_metrics['MAE']:.4f} | "
            f"R²: {validation_metrics['R2']:.4f}"
        )


    # ==================================================
    # Final Results
    # ==================================================

    print("\n" + "=" * 60)
    print("TUNING COMPLETED")
    print("=" * 60)

    print("\nBest tuned Gradient Boosting models:")

    for target_name, metrics in tuned_results.items():

        print("\n" + "-" * 60)
        print(target_name)
        print("-" * 60)

        print(
            f"RMSE: {metrics['RMSE']:.4f}"
        )

        print(
            f"MAE:  {metrics['MAE']:.4f}"
        )

        print(
            f"R²:   {metrics['R2']:.4f}"
        )

    print("\n" + "=" * 60)
    print("READY FOR FINAL MODEL SELECTION")
    print("=" * 60)


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()

