"""Models and evaluation used to predict squat and deadlift."""

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


def evaluate(y_true, y_pred) -> dict[str, float]:
    """MAE in kg (average error) and R2 (1 = perfect, 0 = no better than the mean)."""
    return {"MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


def make_poly_pipeline(degree: int) -> Pipeline:
    """Polynomial features -> standardization -> linear regression."""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    )


def make_svr(
    C: float = 10, gamma: float = 0.2, epsilon: float = 0.1
) -> TransformedTargetRegressor:
    """RBF SVR with standardized inputs AND standardized target."""
    return TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("scale", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)),
            ]
        ),
        transformer=StandardScaler(),  # target scaled for training, converted back to kg for predictions
    )
