import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ======================================================
# Model Evaluation
# ======================================================

def evaluate_model(model, X, y):

    # Generate predictions
    predictions = model.predict(X)

    # Calculate evaluation metrics
    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    r2 = r2_score(
        y,
        predictions
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# ======================================================
# Print Evaluation Results
# ======================================================

def print_evaluation_results(
    model_name,
    target_name,
    metrics,
    dataset_name
):

    print(
        f"\n{model_name} - "
        f"{target_name} - "
        f"{dataset_name}"
    )

    print("-" * 50)

    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")

