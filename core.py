"""Shared evaluation and CSV validation; importing this module never trains a model."""
import csv
import io
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "spam"
WORDS = "make address all 3d our over remove internet order mail receive will people report addresses free business email you credit your font 000 money hp hpl george 650 lab labs telnet 857 data 415 85 technology 1999 parts pm direct cs meeting original project re edu table conference".split()
FEATURES = ([f"word_freq_{w}" for w in WORDS]
            + [f"char_freq_{c}" for c in [";", "(", "[", "!", "$", "#"]]
            + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"])
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def validate_frame(frame):
    """Return an ordered numeric frame or explain why evaluation is impossible."""
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(required))
    if missing or extra:
        raise ValueError(f"CSV columns do not match. Missing: {missing}. Unexpected: {extra}.")
    if not 1 <= len(frame) <= 5000:
        raise ValueError("Upload between 1 and 5,000 test rows.")
    try:
        numeric = frame[required].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Every feature and the spam label must be numeric.") from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing, infinite, or NaN values are not allowed.")
    if not numeric[TARGET].isin([0, 1]).all():
        raise ValueError("The spam target must contain only 0 (non-spam) and 1 (spam).")
    if (numeric[FEATURES] < 0).any().any():
        raise ValueError("Spambase features cannot be negative.")
    if (numeric[FEATURES[:54]] > 100).any().any():
        raise ValueError("Word and character percentages must be between 0 and 100.")
    numeric[TARGET] = numeric[TARGET].astype(int)
    return numeric


def read_test_csv(content):
    if len(content) > 5 * 1024 * 1024:
        raise ValueError("Upload a test CSV smaller than 5 MB.")
    try:
        text = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)), [])
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names are not allowed.")
        return validate_frame(pd.read_csv(io.StringIO(text)))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with a header row.") from exc


def evaluate(model, frame):
    """Spam (1) is positive. AUC uses continuous scores, never hard predictions."""
    X, y = frame[FEATURES], frame[TARGET]
    predictions = model.predict(X)
    if hasattr(model, "predict_proba"):
        positive = list(model.classes_).index(1)
        scores = model.predict_proba(X)[:, positive]
    else:
        scores = model.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, scores) if y.nunique() == 2 else np.nan,
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return metrics, predictions, scores
