import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        'R2': float(r2),
        'RMSE': float(rmse),
    }
