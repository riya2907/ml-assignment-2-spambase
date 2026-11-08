"""Interactive held-out evaluation. There is deliberately no model fitting here."""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve
from core import FEATURES, TARGET, METRICS, evaluate, read_test_csv

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Inbox Lens | Spambase", page_icon="✉", layout="wide")


@st.cache_resource
def load_models():
    manifest = json.loads((ROOT / "model" / "manifest.json").read_text(encoding="utf-8"))
    models = {name: joblib.load(ROOT / "model" / file) for name, file in manifest["models"].items()}
    return manifest, models


st.title("Inbox Lens")
st.caption("Five classifiers. One test set. Explore the cost of missed spam and blocked legitimate mail.")
try:
    manifest, models = load_models()
except (FileNotFoundError, ImportError, ValueError) as exc:
    st.error(f"Saved model artifacts are unavailable or incompatible. Run train.py with requirements.txt. Details: {exc}")
    st.stop()

with st.sidebar:
    st.header("Evaluation desk")
    st.markdown("**Labels:** 0 = non-spam · 1 = spam")
    selected = st.selectbox("Inspect a classifier", list(models))
    uploaded = st.file_uploader("Upload labeled test CSV", type=["csv"], max_upload_size=5)
    st.download_button("Download held-out test CSV", (ROOT / "test_data.csv").read_bytes(),
                       "test_data.csv", "text/csv")
    st.caption(f"CSV needs all {len(FEATURES)} features and spam. Maximum 5,000 rows / 5 MB.")
    st.caption("Saved models only. Uploaded data is used for evaluation without retraining.")

try:
    frame = read_test_csv(uploaded.getvalue() if uploaded is not None else (ROOT / "test_data.csv").read_bytes())
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.info("Evaluating your uploaded CSV." if uploaded is not None else "Evaluating the bundled held-out test set. Upload a labeled CSV to compare all models on new test data.")
if frame[TARGET].nunique() < 2:
    st.warning("This CSV contains one class. ROC AUC is undefined and shown as N/A; both classes are needed for an ROC curve.")
if frame.duplicated(subset=FEATURES).any():
    st.warning("This CSV contains duplicate feature rows. Metrics include every uploaded row, so duplicates affect their weight.")
summary = st.columns(3)
summary[0].metric("Test emails", len(frame))
summary[1].metric("Spam prevalence", f"{frame[TARGET].mean():.1%}")
summary[2].metric("Input features", len(FEATURES))

evaluated = {name: evaluate(model, frame) for name, model in models.items()}
table = pd.DataFrame([{"ML Model Name": name, **values[0]} for name, values in evaluated.items()])
st.subheader("Compare all five models")
st.dataframe(table.style.format({m: "{:.4f}" for m in METRICS}, na_rep="N/A"), hide_index=True, width="stretch")
st.caption("Precision, recall and F1 treat spam as positive. AUC uses continuous probability or SVM decision scores. MCC ranges from -1 to 1.")
st.download_button("Download current comparison", table.to_csv(index=False).encode(), "comparison.csv", "text/csv")

st.subheader(selected)
metrics, predictions, scores = evaluated[selected]
for col, metric in zip(st.columns(6), METRICS):
    value = metrics[metric]
    col.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

diagnostics, data_tab = st.tabs(["Errors and ranking", "Test rows and predictions"])
with diagnostics:
    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(frame[TARGET], predictions, labels=[0, 1],
        display_labels=["Non-spam", "Spam"], colorbar=False, cmap="Blues", ax=ax)
    ax.set_title("Where does the model make mistakes?")
    fig.tight_layout()
    left.pyplot(fig)
    plt.close(fig)
    if frame[TARGET].nunique() == 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, (_, _, score) in evaluated.items():
            fpr, tpr, _ = roc_curve(frame[TARGET], score)
            ax.plot(fpr, tpr, label=name, linewidth=2 if name == selected else 1, alpha=1 if name == selected else .6)
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC: compare ranking quality")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        right.pyplot(fig)
        plt.close(fig)
    report = classification_report(frame[TARGET], predictions, labels=[0, 1],
        target_names=["Non-spam", "Spam"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, width="stretch")
    st.caption("False positives block legitimate mail. False negatives let spam through. Consider both before choosing a classifier.")
with data_tab:
    prediction_frame = frame.copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["spam_score"] = scores
    st.caption("The SVM score is a decision margin, not a probability.")
    st.dataframe(prediction_frame.head(100), hide_index=True, width="stretch")
    st.download_button("Download predictions", prediction_frame.to_csv(index=False).encode(), "predictions.csv", "text/csv")
with st.expander("Experiment design and limitations"):
    st.write(f"Training rows: {manifest['train_rows']}. Held-out rows: {manifest['test_rows']}. Seed: {manifest['seed']}. Identical feature rows were removed before the split; preprocessing was fitted on training data only.")
    st.write("All five models use fixed settings. A single holdout does not establish a statistically reliable winner. This historical email collection includes personal and organization-specific vocabulary and is not representative of modern email.")
    st.link_button("Dataset source: UCI Spambase", manifest["dataset_url"])
