"""Create a report from actual saved results; never invent scores or links."""
import json
from pathlib import Path
import pandas as pd


def write_readme(root):
    manifest = json.loads((root / "model" / "manifest.json").read_text())
    links_file = root / "project_links.json"
    links = json.loads(links_file.read_text(encoding="utf-8")) if links_file.exists() else {}
    repository = links.get("github_repository")
    repository_text = (f"[GitHub repository]({repository})" if repository else
        "**Not configured:** create your repository after reviewing and adapting this learning example, and replace this paragraph with the real clickable URL. Do not invent a link or commit history.")
    metrics = pd.read_csv(root / "results" / "metrics.csv")
    rows = metrics.set_index("ML Model Name")
    winner = metrics.sort_values(["F1", "MCC"], ascending=False).iloc[0]
    def table(frame):
        header = "| " + " | ".join(frame.columns) + " |\n"
        header += "| " + " | ".join(["---"] * len(frame.columns)) + " |\n"
        return header + "\n".join("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    explanations = {
        "Logistic Regression": "A standardized linear baseline. A linear boundary can miss interactions among frequency features.",
        "Decision Tree": "A depth-limited tree captures nonlinear rules. A single tree can be sensitive to the split; depth 10 and leaf size 3 limit complexity.",
        "KNN": "Uses 11 distance-weighted neighbors after standardization. Local similarity can help, but 57-dimensional distances can be less informative.",
        "Naive Bayes": "Gaussian class-conditional densities provide a fast baseline. Correlated, zero-heavy frequency features may violate its assumptions.",
        "Support Vector Machine": "A standardized RBF kernel captures nonlinear relationships. C=2 and gamma=scale are fixed settings; margins support ROC AUC without probability calibration.",
    }
    observations = []
    for name, row in rows.iterrows():
        observations.append({"ML Model Name": name, "Observation about model performance":
            f"F1 {row['F1']:.4f}, precision {row['Precision']:.4f}, recall {row['Recall']:.4f}, AUC {row['AUC']:.4f}. " + explanations[name]})
    observations.append({"ML Model Name": "Overall, winner for this test set?", "Observation about model performance":
        f"{winner['ML Model Name']} has the highest observed F1 ({winner['F1']:.4f}); MCC breaks ties. This describes one holdout, not a proven population-level winner. F1 was chosen to balance spam precision and recall."})
    content = f"""# Inbox Lens: a Spambase classification learning project

This is an AI-assisted worked example for study and adaptation. Understand the code, make your own justified choices, and execute your experiment in BITS Virtual Lab before submission. Local results are not evidence of BITS execution.

## a. Problem statement

Classify email as spam (1) or non-spam (0) using numeric summaries of word frequency, character frequency, and capitalization. Compare five classifiers on exactly the same held-out examples, and explore their errors in Streamlit. False positives block legitimate mail; false negatives miss spam.

## b. Dataset description

- Source: [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase), [DOI](https://doi.org/10.24432/C53G6X).
- Original dimensions: {manifest['source_rows']:,} emails, {manifest['feature_count']} input features, one binary target, `spam`.
- Original labels: {manifest['source_class_counts']['0']:,} non-spam and {manifest['source_class_counts']['1']:,} spam.
- Inputs: 48 word percentages, 6 character percentages, and 3 capital-run measures. No missing values were found in the downloaded source.
- Cleaning: remove all {manifest['conflicting_label_rows_removed']} rows whose identical feature vector has conflicting labels; then remove {manifest['duplicate_rows_removed']} redundant duplicate rows. Retain {manifest['clean_rows']:,} unique, unambiguous examples.
- Split: stratified 80/20, `random_state={manifest['seed']}`; {manifest['train_rows']:,} training rows and {manifest['test_rows']:,} test rows. No identical feature vectors cross the split. The CSV includes all 57 features and the target.
- Limitations: this historical dataset contains organization- and person-specific tokens. Deduplication changes the example distribution. Results on this holdout do not measure modern spam filtering performance.

## c. Github Repository Link

{repository_text}

## d. Models used

All five classifiers train on the same rows. StandardScaler is fitted on training data only and saved inside the Logistic Regression, KNN, and SVM pipelines. The tree and Gaussian Naive Bayes use raw numeric inputs. Settings are fixed in `train.py`; no cross-validation, tuning, or test-based parameter selection was performed.

{table(metrics)}

Positive label is spam (1). Accuracy is the fraction correct. Precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is their harmonic mean. AUC measures ranking over thresholds, using probability scores except for the SVM's decision margins. MCC uses all four confusion-matrix cells and ranges from -1 to 1. Undefined precision/recall/F1 return zero; AUC is shown as N/A for a one-class upload. Values above are proportions, rounded to four decimals.

### Observations about model performance

{table(pd.DataFrame(observations))}

The explanations above describe plausible model behavior, not causal conclusions established by this experiment. A single split has sampling variability. Use cross-validation within the training data if you extend the study; reserve a fresh final holdout when making further model-selection decisions.

## Run and reproduce

Use Python {manifest['python']} (choose Python 3.12 on Community Cloud). From this project directory:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux / BITS Lab: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python verify.py
python -m streamlit run app.py
```

The `data/` folder contains the official UCI source and the generated training split. `test_data.csv` is the exact held-out CSV used for the reported results. `model/` contains all five saved estimators and preprocessing pipelines, plus `manifest.json` with split details, source/test hashes and library versions. `results/` contains full precision metrics and row-level predictions. `train.py` refreshes the README and dependency pins using the environment actually used to train.

## App features

- Upload a labeled test CSV or evaluate the bundled held-out CSV.
- Select any saved classifier; view all six metrics for all five models on the same uploaded data.
- Inspect the selected confusion matrix and classification report; compare ROC curves when both classes are present.
- Download the test CSV, current comparison, or selected predictions.
- Validate column names, numeric values, target labels, missing values, input size and duplicate headers. Reordered columns are accepted and restored to training order.
- No fitting occurs inside the app. Only repository-owned joblib files are loaded.

## Deployment and required PDF

Live app: **not deployed**. No BITS Virtual Lab screenshot has been supplied.

See `STUDY_GUIDE.md` for the learning walkthrough, actual BITS execution steps, Community Cloud deployment, and the required PDF order. The project is not a completed submission until you supply genuine links, run in the required lab, and include its screenshot.

References: [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency management](https://docs.streamlit.io/deploy/concepts/dependencies).
"""
    (root / "README.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    write_readme(Path(__file__).resolve().parent)
