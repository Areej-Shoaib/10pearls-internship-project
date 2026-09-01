import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_feature_data
from preprocessing import preprocess_data
from splitting import split_data
from scaling import scale_features


def main():

    # ==================================================
    # Load and prepare data
    # ==================================================

    print("\nLoading data...")

    df = load_feature_data()

    processed_df, targets = preprocess_data(df)

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

    # ==================================================
    # Target: 48h
    # ==================================================

    target_name = "target_48h"

    X = X_train_scaled
    y = y_train[target_name]

    print("\n" + "=" * 60)
    print("TIME SERIES HYPERPARAMETER SEARCH")
    print("=" * 60)

    print(f"\nTarget: {target_name}")

    # ==================================================
    # TimeSeriesSplit
    # ==================================================

    tscv = TimeSeriesSplit(n_splits=3)

    # ==================================================
    # Gradient Boosting
    # ==================================================

    model = GradientBoostingRegressor(
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3]
    }

    print("\nSearching hyperparameters...")
    print("Using TimeSeriesSplit with 3 folds.")
    print("Testing 8 parameter combinations...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    # ==================================================
    # Best parameters
    # ==================================================

    print("\n" + "-" * 60)
    print("BEST PARAMETERS")
    print("-" * 60)

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    best_cv_rmse = -grid_search.best_score_

    print(f"\nBest TimeSeries CV RMSE: {best_cv_rmse:.4f}")

    # ==================================================
    # Validation performance
    # ==================================================

    best_model = grid_search.best_estimator_

    y_val_pred = best_model.predict(X_val_scaled)

    rmse = np.sqrt(
        mean_squared_error(
            y_val[target_name],
            y_val_pred
        )
    )

    mae = mean_absolute_error(
        y_val[target_name],
        y_val_pred
    )

    r2 = r2_score(
        y_val[target_name],
        y_val_pred
    )

    print("\n" + "-" * 60)
    print("VALIDATION PERFORMANCE")
    print("-" * 60)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # ==================================================
    # Comparison with previous baseline
    # ==================================================

    baseline_rmse = 21.7833
    baseline_mae = 15.0486
    baseline_r2 = 0.2499

    print("\n" + "=" * 60)
    print("BASELINE VS TIME-SERIES TUNED MODEL")
    print("=" * 60)

    print("\nBaseline Gradient Boosting:")
    print(f"RMSE: {baseline_rmse:.4f}")
    print(f"MAE:  {baseline_mae:.4f}")
    print(f"R²:   {baseline_r2:.4f}")

    print("\nTime-Series Tuned Gradient Boosting:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    print("\n" + "=" * 60)

    if rmse < baseline_rmse:
        print("RESULT: TIME-SERIES TUNED MODEL WINS")
    else:
        print("RESULT: BASELINE MODEL REMAINS BETTER")

    print("=" * 60)


if __name__ == "__main__":
    main()