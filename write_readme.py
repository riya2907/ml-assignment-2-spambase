"""Render the project README from measured results and verified publication links."""
import json
from pathlib import Path

import pandas as pd


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |\n"
    header += "| " + " | ".join(["---"] * len(frame.columns)) + " |\n"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(values) + " |")
    return header + "\n".join(rows)


def write_readme(root):
    manifest = json.loads((root / "model" / "manifest.json").read_text(encoding="utf-8"))
    links_file = root / "project_links.json"
    links = json.loads(links_file.read_text(encoding="utf-8")) if links_file.exists() else {}
    repository = links.get("github_repository")
    repository_text = f"[Open GitHub repository]({repository})" if repository else "Not configured."
    owner = repository.rstrip("/").split("/")[-2] if repository else "Not configured"
    live_app = links.get("streamlit_app")
    live_text = f"[Open deployed Streamlit app]({live_app})" if live_app else "**Not deployed yet.** A verified Streamlit Community Cloud URL will be added after deployment."
    clone_command = f"git clone {repository}.git\ncd {repository.rstrip('/').split('/')[-1]}" if repository else "# Clone your repository, then open its project folder."
    metrics = pd.read_csv(root / "results" / "metrics.csv")
    rows = metrics.set_index("ML Model Name")
    winner = metrics.sort_values(["F1", "MCC"], ascending=False).iloc[0]
    observations = []
    for name, row in rows.iterrows():
        if name == "Logistic Regression":
            note = (f"Accuracy {row['Accuracy']:.2%}, F1 {row['F1']:.4f}, and AUC {row['AUC']:.4f}. "
                    "A useful linear baseline with fairly similar spam precision and recall. A linear boundary may miss interactions between features; this experiment does not isolate that effect.")
        elif name == "Decision Tree":
            note = (f"Accuracy {row['Accuracy']:.2%}, F1 {row['F1']:.4f}, and AUC {row['AUC']:.4f}. "
                    "Its feature-threshold rules are interpretable. The depth and leaf-size limits control complexity, but this single tree does not lead the measured comparison.")
        elif name == "KNN":
            note = (f"Accuracy {row['Accuracy']:.2%}, F1 {row['F1']:.4f}, and AUC {row['AUC']:.4f}. "
                    "Distance-weighted neighbors provide a competitive baseline after standardization. Similarity depends on the scaling and geometry of the 57-dimensional inputs.")
        elif name == "Naive Bayes":
            highest_recall = "the highest" if row["Recall"] == metrics["Recall"].max() else "a"
            note = (f"Spam recall is {row['Recall']:.2%}, {highest_recall} observed recall, while precision is {row['Precision']:.2%} "
                    f"and F1 is {row['F1']:.4f}. It catches many spam emails at the cost of more false alarms. Gaussian likelihoods and conditional independence are strong assumptions for sparse, correlated frequency features.")
        else:
            leading = "the highest observed F1" if row["F1"] == metrics["F1"].max() else "an observed F1"
            note = (f"Accuracy {row['Accuracy']:.2%}, {leading} of {row['F1']:.4f}, and AUC {row['AUC']:.4f}. "
                    "The standardized RBF model gives a strong measured balance of precision and recall. Its decision margins are used for AUC and are not calibrated probabilities.")
        observations.append({"ML Model Name": name, "Observation about model performance": note})
    observations.append({"ML Model Name": "Overall, winner for this test set?", "Observation about model performance":
        f"**{winner['ML Model Name']}**, using F1 as the primary comparison measure (F1 {winner['F1']:.4f}, MCC {winner['MCC']:.4f}). F1 balances spam precision and recall; MCC breaks ties. This describes one holdout, not proven superiority on future email."})
    feature_groups = "\n\n".join("```text\n" + "\n".join(manifest["features"][a:b]) + "\n```"
                                    for a, b in [(0, 48), (48, 54), (54, 57)])
    groups = pd.DataFrame([
        {"Feature group": "Word frequencies", "Count": 48, "Description": "Percentage of words matching each recorded token."},
        {"Feature group": "Character frequencies", "Count": 6, "Description": "Percentage of characters matching each recorded symbol."},
        {"Feature group": "Capital-letter runs", "Count": 3, "Description": "Average, longest and total capital-letter run measures."},
    ])
    settings = pd.DataFrame([
        {"Classifier": "Logistic Regression", "Preprocessing": "StandardScaler", "Fixed settings": "max_iter=3000; random_state=2026"},
        {"Classifier": "Decision Tree", "Preprocessing": "Raw numeric inputs", "Fixed settings": "max_depth=10; min_samples_leaf=3; random_state=2026"},
        {"Classifier": "KNN", "Preprocessing": "StandardScaler", "Fixed settings": "n_neighbors=11; weights=distance"},
        {"Classifier": "Gaussian Naive Bayes", "Preprocessing": "Raw numeric inputs", "Fixed settings": "Default GaussianNB settings"},
        {"Classifier": "Support Vector Machine", "Preprocessing": "StandardScaler", "Fixed settings": "kernel=rbf; C=2.0; gamma=scale"},
    ])
    content = f"""# Inbox Lens
## Email Spam Classification and Machine Learning Benchmarks

**Name:** Riya Ganguli  
**BITS ID:** 2026AJ05124  
**Repository owner:** [{owner}](https://github.com/{owner})  
**Course:** Machine Learning - M.Tech (AIML), Assignment 2  
**Dataset:** UCI Spambase | **Task:** Binary classification | **Interface:** Streamlit

## a. Problem Statement

Build and compare five supervised classifiers that identify an email as **spam (1)** or **non-spam (0)** from numeric summaries of its vocabulary, punctuation and capitalization. Evaluate each classifier on the same held-out emails using Accuracy, ROC AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC).

The Streamlit application loads the saved models, accepts labeled test CSV uploads, and presents model comparisons and error diagnostics. Blocking legitimate email is a false positive; allowing spam through is a false negative. Both matter when interpreting performance.

## b. Dataset Description

| Item | Description |
| --- | --- |
| Dataset | [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) |
| Citation | [Spambase dataset DOI](https://doi.org/10.24432/C53G6X) |
| Original instances | {manifest['source_rows']:,} email records; exceeds the minimum of 500 |
| Input features | **{manifest['feature_count']} numeric predictors**; exceeds the minimum of 12 |
| Target | `spam`: `0` = non-spam, `1` = spam |
| Original class counts | {manifest['source_class_counts']['0']:,} non-spam; {manifest['source_class_counts']['1']:,} spam |
| Missing values | None found in the downloaded source |
| Cleaned instances | {manifest['clean_rows']:,} unique feature vectors with unambiguous labels |
| Training / test rows | {manifest['train_rows']:,} / {manifest['test_rows']:,} |
| Split | Stratified 80/20 holdout; random_state={manifest['seed']} |

The target is **additional to the 57 input features** and is never passed into a model as a predictor.

### Feature Groups

{markdown_table(groups)}

<details>
<summary>View all 57 input column names, in dataset order</summary>

The three blocks correspond to word frequencies, character frequencies and capital-letter runs.

{feature_groups}

</details>

### Data Preparation and Leakage Prevention

1. Check the source dimensions, numeric values and absence of missing or infinite inputs.
2. Remove all {manifest['conflicting_label_rows_removed']} rows whose identical feature vector appears with conflicting labels.
3. Remove {manifest['duplicate_rows_removed']} remaining redundant duplicate rows, leaving {manifest['clean_rows']:,} examples. Clean **before** splitting so the same feature vector cannot appear on both sides.
4. Use the same stratified training/test split for every classifier. The test CSV contains {manifest['test_class_counts']['0']} non-spam and {manifest['test_class_counts']['1']} spam emails.
5. Fit scaling only on training data, save it inside the relevant pipelines, and reuse it unchanged for test data.

Deduplication changes the evaluation distribution. This historical collection also contains organization- and person-specific vocabulary, so its holdout scores should not be treated as modern email-filtering performance.

## c. GitHub Repository Link

{repository_text}

## d. Models Used and Comparison Table

All classifiers use the same training examples and test examples. Hyperparameters are fixed in `train.py`; no cross-validation, parameter search or test-based tuning was performed.

### Classifiers and Settings

{markdown_table(settings)}

### Evaluation Results

These are measured results from the saved {manifest['test_rows']:,}-row test CSV. Scores are proportions rounded to four decimals; full precision values are in [`results/metrics.csv`](results/metrics.csv).

{markdown_table(metrics)}

### Observations on Model Performance

{markdown_table(pd.DataFrame(observations))}

Interpretations of model mechanisms above are plausible explanations, not causal findings established by this experiment. A single split has sampling variability. Use training-only cross-validation for an extended comparison and protect an appropriate final holdout.

### Metric Conventions

- **Accuracy:** fraction of all emails classified correctly.
- **Precision, Recall and F1:** binary scores with spam (`1`) as the positive class; undefined divisions return zero.
- **AUC:** calculated from continuous spam probabilities, or the SVM's decision margins, rather than hard predicted labels. A one-class upload has undefined AUC, displayed as `N/A`.
- **MCC:** uses all four confusion-matrix cells and ranges from -1 to 1.

## e. How to Run Locally

Use **Python 3.12**. This experiment was executed with Python {manifest['python']}; the model manifest records its exact library versions.

### 1. Clone the Repository

```bash
{clone_command}
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

**Linux / macOS / BITS Virtual Lab**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the App with the Included Saved Models

```bash
python -m streamlit run app.py
```

The repository already includes all five trained models and the test CSV. Training is unnecessary for viewing the app. Open the local URL printed by Streamlit, upload `test_data.csv` or use the bundled test set, and select a classifier.

### 5. Reproduce Training and Verify Results

```bash
python train.py
python verify.py
```

Run these commands in BITS Virtual Lab for the required lab execution. Training regenerates model artifacts, the train/test CSVs, metric files, dependency pins and this README. The repository link is preserved through `project_links.json`; reapply other personal README edits after regeneration.

## f. Streamlit Application Features

| Feature | Behavior |
| --- | --- |
| CSV upload | Accepts a labeled test CSV with all 57 inputs and `spam` |
| Model selector | Switches between all five saved classifiers |
| Comparison table | Recalculates all six metrics for every model on the same uploaded rows |
| Selected-model metrics | Shows six metric cards for the chosen classifier |
| Error analysis | Confusion matrix and classification report for the selected model |
| ROC comparison | Displays all five ROC curves when both target classes are present |
| Downloads | Test CSV, current comparison and selected-model predictions |
| Validation | Checks columns, numeric values, finite inputs, labels, duplicate headers and size limits |
| Evaluation only | Uploaded data never triggers model retraining |

Upload limits are **5 MB** and **5,000 rows**. Column order can differ, but all required names must be present and unexpected columns are rejected. Duplicate input rows produce a warning. SVM scores are margins rather than probabilities.

## g. Project Structure

```text
ml-assignment-2-spambase/
|-- app.py                   # Streamlit evaluation interface
|-- core.py                  # CSV validation and shared metric calculation
|-- train.py                 # Cleaning, splitting, fitting and artifact export
|-- verify.py                # Model, metric, input and app checks
|-- write_readme.py          # README generated from real results
|-- requirements.txt         # Exact direct dependency versions
|-- README.md
|-- STUDY_GUIDE.md
|-- project_links.json       # Verified repository/deployment links
|-- test_data.csv            # Labeled held-out test rows
|-- .streamlit/config.toml
|-- data/                    # UCI source files and generated training split
|-- model/                   # Five saved models/pipelines and manifest
`-- results/                 # Metric comparison and per-model predictions
```

`model/manifest.json` records the feature order, split counts, seed, source/test SHA-256 hashes and package versions. Credentials, temporary files and local dependency folders are excluded from the repository.

## h. Streamlit Community Cloud Deployment

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Choose **Create app**, select this repository and branch `main`, and set the entrypoint to `app.py`.
3. Select **Python 3.12** in Advanced settings and deploy.
4. Open the assigned URL, upload the test CSV, switch classifiers, and check the comparison and confusion matrix.
5. Record the verified app URL under `streamlit_app` in `project_links.json` and regenerate the README with `python write_readme.py`.

Official instructions: [Deploy an app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Streamlit Deployed App Link

{live_text}

## Submission Status

- GitHub repository: {"available at the link above" if repository else "not configured"}.
- Streamlit Community Cloud: {"app link recorded above" if live_app else "deployment pending"}.
- BITS Virtual Lab evidence: a user-provided screenshot of the app running inside the lab is included in the submission PDF.
- Final PDF: assemble the real GitHub link, live app link, lab screenshot, and full README content in that order after completing the required work.

See [`STUDY_GUIDE.md`](STUDY_GUIDE.md) for the learning walkthrough and submission steps. Keep an accurate record of your own work and assistance used; this repository is an AI-assisted learning example.

## References

- [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) and [dataset DOI](https://doi.org/10.24432/C53G6X).
- [scikit-learn documentation](https://scikit-learn.org/stable/) and [citation guidance](https://scikit-learn.org/stable/about.html#citing-scikit-learn).
- [Streamlit documentation](https://docs.streamlit.io/) and [dependency management](https://docs.streamlit.io/deploy/concepts/dependencies).
- [User-supplied VinoClass AI reference](https://github.com/arrjvvmukesh/BITS_ML_Assignment_2_Wine_ML_Classifier_VinoClass_AI_2026AJ05810): reference for README organization. This project's dataset, settings, results and wording are specific to Inbox Lens.
"""
    (root / "README.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    write_readme(Path(__file__).resolve().parent)
