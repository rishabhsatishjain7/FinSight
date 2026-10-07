import pandas as pd
import pytest

from scoring.xgboost_model import DistressScorer


def _toy_dataset(n=60, n_features=8):
    """Small but learnable synthetic dataset: label is driven by feature 0."""
    import random

    random.seed(0)
    rows = []
    labels = []
    for i in range(n):
        f0 = random.uniform(-3, 3)
        row = {f"f{j}": random.uniform(-1, 1) for j in range(1, n_features)}
        row["f0"] = f0
        rows.append(row)
        labels.append(1 if f0 < -1.0 else 0)
    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y


def test_train_returns_expected_metrics_shape():
    X, y = _toy_dataset()
    scorer = DistressScorer()
    metrics = scorer.train(X, y, n_estimators=20, cv_folds=3)

    assert metrics["n_samples"] == len(X)
    assert metrics["n_features"] == X.shape[1]
    assert 0.0 <= metrics["train_auc"] <= 1.0
    # cv_auc_mean should be present and meaningfully lower than train_auc in
    # general, though we only assert it's a valid probability-like score here.
    assert metrics["cv_auc_mean"] is not None
    assert 0.0 <= metrics["cv_auc_mean"] <= 1.0
    assert len(metrics["cv_auc_folds"]) == 3


def test_score_reindexes_missing_and_extra_columns():
    """RC-004 regression: scoring must not crash when the input DataFrame's
    columns don't exactly match the training feature set (schema drift)."""
    X, y = _toy_dataset()
    scorer = DistressScorer()
    scorer.train(X, y, n_estimators=20, cv_folds=3)

    # Simulate schema drift: drop a trained-on feature, add a new unseen one.
    drifted = X.drop(columns=["f1"]).copy()
    drifted["brand_new_feature"] = 0.5

    result = scorer.score(drifted)
    assert len(result) == len(drifted)
    assert result["distress_probability"].between(0, 1).all()


def test_explain_single_returns_shap_breakdown():
    X, y = _toy_dataset()
    scorer = DistressScorer()
    scorer.train(X, y, n_estimators=20, cv_folds=3)

    explanation = scorer.explain_single(X.iloc[0])
    assert 0.0 <= explanation["distress_probability"] <= 1.0
    assert isinstance(explanation["base_value"], float)
    assert len(explanation["contributions"]) == X.shape[1]
    for c in explanation["contributions"]:
        assert set(c.keys()) == {"feature", "shap_value", "feature_value"}


def test_save_and_load_round_trip(tmp_path):
    X, y = _toy_dataset()
    model_path = tmp_path / "test_model.json"
    scorer = DistressScorer(model_path=model_path)
    scorer.train(X, y, n_estimators=20, cv_folds=3)
    scorer.save()

    reloaded = DistressScorer(model_path=model_path)
    reloaded.load()
    assert reloaded.feature_names == scorer.feature_names

    result = reloaded.score(X.head(3))
    assert len(result) == 3


def test_cv_skipped_gracefully_when_too_few_minority_examples():
    """With only 2 positive examples and cv_folds=5, CV can't stratify --
    must degrade to None rather than raising."""
    X, y = _toy_dataset(n=30)
    y = pd.Series([1, 1] + [0] * (len(y) - 2))  # force minority count below cv_folds

    scorer = DistressScorer()
    metrics = scorer.train(X, y, n_estimators=10, cv_folds=5)
    assert metrics["cv_auc_mean"] is None
