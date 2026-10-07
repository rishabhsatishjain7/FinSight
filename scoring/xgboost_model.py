"""
FinSight — XGBoost financial distress scoring model.

Predicts probability of financial distress from the Z-scored ratio feature
set. Distress is proxied via forward-looking weak-fundamentals heuristics
at training time (see train.py) when no labeled default dataset is
available — swap in real labels (e.g. credit downgrades, bankruptcy
filings) by replacing `build_training_labels`.

SHAP values are computed alongside every prediction so the narrative engine
(narrative/gemini_client.py) can ground its RAG context in *why* the model
scored a company the way it did, not just the score itself.

A note on evaluation metrics: in-sample ("train") AUC is close to
meaningless here — with a few hundred trees at depth 4 over a training set
of only a few hundred rows, XGBoost will essentially memorize the training
labels regardless of whether the features actually generalize, which masks
things like label leakage rather than exposing them. `train()` therefore
also reports a stratified K-fold cross-validated AUC (`cv_auc_mean`) from
freshly-fit models on held-out folds, which is the number that actually
moves when you remove leaked features — the in-sample number mostly won't.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from config.settings import XGB_MODEL_PATH

logger = logging.getLogger("finsight.scoring.xgboost_model")


class DistressScorer:
    def __init__(self, model_path: Path = XGB_MODEL_PATH):
        self.model_path = model_path
        self.model: xgb.XGBClassifier | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        cv_folds: int = 5,
    ) -> dict:
        self.feature_names = list(X.columns)

        model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=42,
        )

        cv_auc_mean, cv_auc_std, cv_scores = self._cross_validated_auc(
            X, y, model_params, cv_folds
        )

        # Final model fit on ALL data -- used for production scoring/SHAP.
        # cv_auc_mean above, not this model's own in-sample fit, is the
        # trustworthy generalization estimate; see module docstring.
        self.model = xgb.XGBClassifier(**model_params)
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

        preds = self.model.predict_proba(X)[:, 1]
        metrics = {
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "positive_rate": float(y.mean()),
            "train_auc": self._safe_auc(y, preds),
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std,
            "cv_auc_folds": cv_scores,
        }
        logger.info("Trained distress model: %s", metrics)
        return metrics

    @staticmethod
    def _cross_validated_auc(
        X: pd.DataFrame, y: pd.Series, model_params: dict, cv_folds: int
    ) -> tuple[float | None, float | None, list[float] | None]:
        """
        Stratified K-fold CV AUC using freshly-initialized models per fold
        (never the model that later gets saved/deployed). This is the
        metric that actually reflects whether features generalize, since
        each fold's model never sees the rows it's scored against.
        """
        try:
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedKFold
        except ImportError:
            return None, None, None

        # Need at least 2 examples of the minority class per fold to stratify.
        min_class_count = int(y.value_counts().min()) if y.nunique() > 1 else 0
        if min_class_count < cv_folds:
            logger.warning(
                "Too few minority-class examples (%d) for %d-fold CV; skipping CV AUC.",
                min_class_count,
                cv_folds,
            )
            return None, None, None

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            fold_model = xgb.XGBClassifier(**model_params)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_preds = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
            try:
                scores.append(float(roc_auc_score(y.iloc[test_idx], fold_preds)))
            except ValueError:
                continue  # single-class fold, AUC undefined

        if not scores:
            return None, None, None
        return float(np.mean(scores)), float(np.std(scores)), scores

    @staticmethod
    def _safe_auc(y_true, y_pred) -> float | None:
        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(y_true, y_pred))
        except Exception:  # noqa: BLE001 — AUC undefined for single-class y, etc.
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        self.model.save_model(str(self.model_path))
        feature_path = self.model_path.with_suffix(".features.txt")
        feature_path.write_text("\n".join(self.feature_names))

    def load(self) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))
        feature_path = self.model_path.with_suffix(".features.txt")
        self.feature_names = feature_path.read_text().strip().split("\n")
        self.explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame indexed like X with columns:
            distress_probability, top_risk_factors (list of (feature, shap_value))
        """
        if self.model is None:
            raise RuntimeError("Model not loaded/trained.")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0)
        probs = self.model.predict_proba(X_aligned)[:, 1]
        shap_values = self.explainer.shap_values(X_aligned)

        rows = []
        for i, idx in enumerate(X_aligned.index):
            contributions = list(zip(self.feature_names, shap_values[i]))
            contributions.sort(key=lambda t: abs(t[1]), reverse=True)
            rows.append(
                {
                    "index": idx,
                    "distress_probability": float(probs[i]),
                    "top_risk_factors": contributions[:5],
                }
            )
        return pd.DataFrame(rows).set_index("index")

    def explain_single(self, x_row: pd.Series) -> dict:
        """Detailed SHAP breakdown for one company/year, for report generation."""
        X = pd.DataFrame([x_row]).reindex(columns=self.feature_names, fill_value=0.0)
        prob = float(self.model.predict_proba(X)[:, 1][0])
        shap_vals = self.explainer.shap_values(X)[0]
        base_value = float(self.explainer.expected_value)

        contributions = sorted(
            zip(self.feature_names, shap_vals, X.iloc[0].values),
            key=lambda t: abs(t[1]),
            reverse=True,
        )
        return {
            "distress_probability": prob,
            "base_value": base_value,
            "contributions": [
                {"feature": f, "shap_value": float(sv), "feature_value": float(fv)}
                for f, sv, fv in contributions
            ],
        }
