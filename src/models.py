"""Model definitions."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def make_logreg_baseline() -> Pipeline:
    """Scaled logistic regression with balanced class weights.

    The scaler sits inside the pipeline, so calling ``fit`` on training data
    learns the means and standard deviations from training rows only.
    ``class_weight="balanced"`` up-weights the rare fraud class so the model
    does not just predict "legit" everywhere.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                         random_state=SEED)),
        ]
    )
