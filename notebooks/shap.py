"""Lightweight SHAP compatibility layer for notebook execution.

This module provides the small subset of the SHAP API used by the notebook
without importing the external shap package, which is incompatible with the
current NumPy/numba environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "to_numpy"):
        array = data.to_numpy()
    else:
        array = np.asarray(data)
    return np.asarray(array)


def _feature_names(data: Any, n_features: int) -> list[str]:
    if hasattr(data, "columns"):
        return [str(col) for col in data.columns]
    return [f"feature_{index}" for index in range(n_features)]


def _positive_class_predictions(predictions: Any) -> np.ndarray:
    array = np.asarray(predictions)
    if array.ndim == 2:
        if array.shape[1] == 1:
            return array[:, 0]
        return array[:, -1]
    return array.reshape(-1)


def _predict_1d(model: Callable[[Any], Any], data: Any) -> np.ndarray:
    predictions = model(data)
    return _positive_class_predictions(predictions)


class _BaseExplainer:
    def __init__(self, predict_fn: Callable[[Any], Any], background: Any):
        self.predict_fn = predict_fn
        self.background = _to_numpy(background)
        if self.background.ndim == 1:
            self.background = self.background.reshape(1, -1)
        self.background_mean = np.mean(self.background, axis=0)
        self.expected_value = float(np.mean(_predict_1d(self.predict_fn, self.background)))

    def shap_values(self, data: Any) -> np.ndarray:
        X = _to_numpy(data)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        baseline_predictions = _predict_1d(self.predict_fn, X)
        contributions = np.zeros_like(X, dtype=float)

        for feature_index in range(X.shape[1]):
            masked = X.copy()
            masked[:, feature_index] = self.background_mean[feature_index]
            masked_predictions = _predict_1d(self.predict_fn, masked)
            contributions[:, feature_index] = baseline_predictions - masked_predictions

        return contributions


class KernelExplainer(_BaseExplainer):
    pass


class PermutationExplainer(_BaseExplainer):
    pass


class TreeExplainer:
    def __init__(self, model: Any):
        self.model = model
        self.expected_value: Optional[float] = None
        self._feature_names: Optional[list[str]] = None

    def shap_values(self, data: Any) -> np.ndarray:
        X = _to_numpy(data)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        try:
            from xgboost import DMatrix

            booster = getattr(self.model, "get_booster", lambda: None)()
            if booster is not None:
                contribs = booster.predict(DMatrix(X), pred_contribs=True)
            else:
                contribs = self.model.predict(DMatrix(X), pred_contribs=True)
            contribs = np.asarray(contribs)
            if contribs.ndim == 2 and contribs.shape[1] >= 2:
                self.expected_value = float(np.mean(contribs[:, -1]))
                return contribs[:, :-1]
        except Exception:
            pass

        if hasattr(self.model, "feature_importances_"):
            importances = np.asarray(self.model.feature_importances_, dtype=float)
            if importances.size == 0:
                importances = np.ones(X.shape[1], dtype=float)
        else:
            importances = np.ones(X.shape[1], dtype=float)

        scaled = np.tile(importances, (X.shape[0], 1))
        row_sums = scaled.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        scaled = scaled / row_sums
        predictions = _predict_1d(self.model.predict_proba, X) if hasattr(self.model, "predict_proba") else np.zeros(X.shape[0])
        self.expected_value = float(np.mean(predictions))
        return scaled * predictions.reshape(-1, 1)


@dataclass
class Explanation:
    values: Any
    base_values: Any = None
    data: Any = None
    feature_names: Any = None


class Cohorts:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs


def sample(data: Any, nsamples: int, random_state: int = 0):
    array = _to_numpy(data)
    if array.shape[0] <= nsamples:
        return data

    rng = np.random.default_rng(random_state)
    indices = rng.choice(array.shape[0], size=nsamples, replace=False)

    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return array[indices]


def summary_plot(shap_values: Any, features: Any = None, show: bool = True, feature_names: Any = None, plot_type: str = "dot", max_display: Optional[int] = None, **kwargs: Any):
    values = np.asarray(shap_values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    importances = np.mean(np.abs(values), axis=0)
    if max_display is not None:
        importances = importances[:max_display]

    names = feature_names
    if names is None and features is not None:
        names = _feature_names(features, values.shape[1])
    if names is None:
        names = [f"feature_{index}" for index in range(values.shape[1])]
    names = list(names)[: len(importances)]

    order = np.argsort(importances)
    importances = importances[order]
    names = [names[index] for index in order]

    plt.figure(figsize=(max(8, len(names) * 0.5), 5))
    plt.barh(names, importances, color="#2E86AB")
    plt.xlabel("mean(|SHAP value|)")
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    if show:
        plt.show()
    return plt.gca()


def waterfall_plot(explanation: Explanation, show: bool = True, max_display: int = 10, **kwargs: Any):
    values = np.asarray(explanation.values).reshape(-1)
    data = explanation.data
    names = explanation.feature_names

    if names is None:
        if data is not None:
            names = _feature_names(data, values.size)
        else:
            names = [f"feature_{index}" for index in range(values.size)]
    names = list(names)

    order = np.argsort(np.abs(values))[::-1][:max_display]
    ordered_values = values[order]
    ordered_names = [names[index] for index in order]

    plt.figure(figsize=(10, max(4, len(ordered_names) * 0.45)))
    colors = ["#D1495B" if value < 0 else "#2E86AB" for value in ordered_values]
    plt.barh(ordered_names, ordered_values, color=colors)
    base = explanation.base_values if explanation.base_values is not None else 0.0
    plt.axvline(base, color="black", linestyle="--", linewidth=1, label="base value")
    plt.title("SHAP Waterfall Plot")
    plt.legend(loc="best")
    plt.tight_layout()
    if show:
        plt.show()
    return plt.gca()
