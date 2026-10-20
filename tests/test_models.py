"""Tests for the evaluation metric and the model pipelines."""

import numpy as np
import pandas as pd

from ml_powerlifting_potential.models import evaluate, make_poly_pipeline, make_svr


def test_evaluate_perfect_prediction():
    y = [100.0, 150.0, 200.0]
    scores = evaluate(y, y)

    assert scores["MAE"] == 0
    assert scores["R2"] == 1


def test_evaluate_mae_in_kg():
    scores = evaluate([100.0, 200.0], [105.0, 185.0])

    assert scores["MAE"] == 10  # (5 + 15) / 2


def test_pipelines_fit_a_simple_relation():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"bench": rng.uniform(60, 160, 300)})
    y = 1.5 * X["bench"]  # squat = 1.5 x bench, no noise

    for model in [make_poly_pipeline(2), make_svr(C=30, gamma=0.02, epsilon=0.1)]:
        model.fit(X, y)
        assert evaluate(y, model.predict(X))["MAE"] < 2  # less than 2 kg of error
