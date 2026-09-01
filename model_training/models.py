from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ======================================================
# Ridge Regression
# ======================================================

def train_ridge(X_train, y_train):

    model = Ridge(
        alpha=1.0
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ======================================================
# Random Forest
# ======================================================

def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ======================================================
# Gradient Boosting
# ======================================================

def train_gradient_boosting(X_train, y_train):

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model

# ======================================================
# Gradient Boosting Hyperparameter Tuning
# ======================================================

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor


def tune_gradient_boosting(X_train, y_train):

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    model = GradientBoostingRegressor(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_
