import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer

__all__ = ["rmlse", "rmlse_scorer"]

def rmlse(y_true, y_pred):
    """
    Root Mean Logarithmic Squared Error

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns:
    --------
    score : float
        RMLSE score.
    """
    # Ensure inputs are positive (add small epsilon to avoid log(0))
    y_true = np.maximum(y_true, 1e-7)
    y_pred = np.maximum(y_pred, 1e-7)

    # Calculate RMLSE
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# Create the scorer using sklearn's make_scorer
rmlse_scorer = make_scorer(rmlse, greater_is_better=False)
