# Inbox Lens: a Spambase classification learning project

This is an AI-assisted worked example for study and adaptation. Understand the code, make your own justified choices, and execute your experiment in BITS Virtual Lab before submission. Local results are not evidence of BITS execution.

## a. Problem statement

Classify email as spam (1) or non-spam (0) using numeric summaries of word frequency, character frequency, and capitalization. Compare five classifiers on exactly the same held-out examples, and explore their errors in Streamlit. False positives block legitimate mail; false negatives miss spam.

## b. Dataset description

- Source: [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase), [DOI](https://doi.org/10.24432/C53G6X).
- Original dimensions: 4,601 emails, 57 input features, one binary target, `spam`.
- Original labels: 2,788 non-spam and 1,813 spam.
- Inputs: 48 word percentages, 6 character percentages, and 3 capital-run measures. No missing values were found in the downloaded source.
- Cleaning: remove all 6 rows whose identical feature vector has conflicting labels; then remove 391 redundant duplicate rows. Retain 4,204 unique, unambiguous examples.
- Split: stratified 80/20, `random_state=2026`; 3,363 training rows and 841 test rows. No identical feature vectors cross the split. The CSV includes all 57 features and the target.
- Limitations: this historical dataset contains organization- and person-specific tokens. Deduplication changes the example distribution. Results on this holdout do not measure modern spam filtering performance.

## c. Github Repository Link

[GitHub repository](https://github.com/riya2907/ml-assignment-2-spambase)

## d. Models used

All five classifiers train on the same rows. StandardScaler is fitted on training data only and saved inside the Logistic Regression, KNN, and SVM pipelines. The tree and Gaussian Naive Bayes use raw numeric inputs. Settings are fixed in `train.py`; no cross-validation, tuning, or test-based parameter selection was performed.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9263 | 0.9713 | 0.9149 | 0.8985 | 0.9066 | 0.8458 |
| Decision Tree | 0.9180 | 0.9322 | 0.9130 | 0.8776 | 0.8950 | 0.8281 |
| KNN | 0.9215 | 0.9715 | 0.9063 | 0.8955 | 0.9009 | 0.8360 |
| Naive Bayes | 0.8288 | 0.9474 | 0.7099 | 0.9642 | 0.8177 | 0.6909 |
| Support Vector Machine | 0.9405 | 0.9777 | 0.9279 | 0.9224 | 0.9251 | 0.8759 |

Positive label is spam (1). Accuracy is the fraction correct. Precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is their harmonic mean. AUC measures ranking over thresholds, using probability scores except for the SVM's decision margins. MCC uses all four confusion-matrix cells and ranges from -1 to 1. Undefined precision/recall/F1 return zero; AUC is shown as N/A for a one-class upload. Values above are proportions, rounded to four decimals.

### Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | F1 0.9066, precision 0.9149, recall 0.8985, AUC 0.9713. A standardized linear baseline. A linear boundary can miss interactions among frequency features. |
| Decision Tree | F1 0.8950, precision 0.9130, recall 0.8776, AUC 0.9322. A depth-limited tree captures nonlinear rules. A single tree can be sensitive to the split; depth 10 and leaf size 3 limit complexity. |
| KNN | F1 0.9009, precision 0.9063, recall 0.8955, AUC 0.9715. Uses 11 distance-weighted neighbors after standardization. Local similarity can help, but 57-dimensional distances can be less informative. |
| Naive Bayes | F1 0.8177, precision 0.7099, recall 0.9642, AUC 0.9474. Gaussian class-conditional densities provide a fast baseline. Correlated, zero-heavy frequency features may violate its assumptions. |
| Support Vector Machine | F1 0.9251, precision 0.9279, recall 0.9224, AUC 0.9777. A standardized RBF kernel captures nonlinear relationships. C=2 and gamma=scale are fixed settings; margins support ROC AUC without probability calibration. |
| Overall, winner for this test set? | Support Vector Machine has the highest observed F1 (0.9251); MCC breaks ties. This describes one holdout, not a proven population-level winner. F1 was chosen to balance spam precision and recall. |

The explanations above describe plausible model behavior, not causal conclusions established by this experiment. A single split has sampling variability. Use cross-validation within the training data if you extend the study; reserve a fresh final holdout when making further model-selection decisions.

## Run and reproduce

Use Python 3.12.14 (choose Python 3.12 on Community Cloud). From this project directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
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
