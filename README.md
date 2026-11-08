# Inbox Lens
## Email Spam Classification and Machine Learning Benchmarks

**Name:** Riya Ganguli  
**BITS ID:** 2026AJ05124  
**Repository owner:** [riya2907](https://github.com/riya2907)  
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
| Original instances | 4,601 email records; exceeds the minimum of 500 |
| Input features | **57 numeric predictors**; exceeds the minimum of 12 |
| Target | `spam`: `0` = non-spam, `1` = spam |
| Original class counts | 2,788 non-spam; 1,813 spam |
| Missing values | None found in the downloaded source |
| Cleaned instances | 4,204 unique feature vectors with unambiguous labels |
| Training / test rows | 3,363 / 841 |
| Split | Stratified 80/20 holdout; random_state=2026 |

The target is **additional to the 57 input features** and is never passed into a model as a predictor.

### Feature Groups

| Feature group | Count | Description |
| --- | --- | --- |
| Word frequencies | 48 | Percentage of words matching each recorded token. |
| Character frequencies | 6 | Percentage of characters matching each recorded symbol. |
| Capital-letter runs | 3 | Average, longest and total capital-letter run measures. |

<details>
<summary>View all 57 input column names, in dataset order</summary>

The three blocks correspond to word frequencies, character frequencies and capital-letter runs.

```text
word_freq_make
word_freq_address
word_freq_all
word_freq_3d
word_freq_our
word_freq_over
word_freq_remove
word_freq_internet
word_freq_order
word_freq_mail
word_freq_receive
word_freq_will
word_freq_people
word_freq_report
word_freq_addresses
word_freq_free
word_freq_business
word_freq_email
word_freq_you
word_freq_credit
word_freq_your
word_freq_font
word_freq_000
word_freq_money
word_freq_hp
word_freq_hpl
word_freq_george
word_freq_650
word_freq_lab
word_freq_labs
word_freq_telnet
word_freq_857
word_freq_data
word_freq_415
word_freq_85
word_freq_technology
word_freq_1999
word_freq_parts
word_freq_pm
word_freq_direct
word_freq_cs
word_freq_meeting
word_freq_original
word_freq_project
word_freq_re
word_freq_edu
word_freq_table
word_freq_conference
```

```text
char_freq_;
char_freq_(
char_freq_[
char_freq_!
char_freq_$
char_freq_#
```

```text
capital_run_length_average
capital_run_length_longest
capital_run_length_total
```

</details>

### Data Preparation and Leakage Prevention

1. Check the source dimensions, numeric values and absence of missing or infinite inputs.
2. Remove all 6 rows whose identical feature vector appears with conflicting labels.
3. Remove 391 remaining redundant duplicate rows, leaving 4,204 examples. Clean **before** splitting so the same feature vector cannot appear on both sides.
4. Use the same stratified training/test split for every classifier. The test CSV contains 506 non-spam and 335 spam emails.
5. Fit scaling only on training data, save it inside the relevant pipelines, and reuse it unchanged for test data.

Deduplication changes the evaluation distribution. This historical collection also contains organization- and person-specific vocabulary, so its holdout scores should not be treated as modern email-filtering performance.

## c. GitHub Repository Link

[Open GitHub repository](https://github.com/riya2907/ml-assignment-2-spambase)

## d. Models Used and Comparison Table

All classifiers use the same training examples and test examples. Hyperparameters are fixed in `train.py`; no cross-validation, parameter search or test-based tuning was performed.

### Classifiers and Settings

| Classifier | Preprocessing | Fixed settings |
| --- | --- | --- |
| Logistic Regression | StandardScaler | max_iter=3000; random_state=2026 |
| Decision Tree | Raw numeric inputs | max_depth=10; min_samples_leaf=3; random_state=2026 |
| KNN | StandardScaler | n_neighbors=11; weights=distance |
| Gaussian Naive Bayes | Raw numeric inputs | Default GaussianNB settings |
| Support Vector Machine | StandardScaler | kernel=rbf; C=2.0; gamma=scale |

### Evaluation Results

These are measured results from the saved 841-row test CSV. Scores are proportions rounded to four decimals; full precision values are in [`results/metrics.csv`](results/metrics.csv).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9263 | 0.9713 | 0.9149 | 0.8985 | 0.9066 | 0.8458 |
| Decision Tree | 0.9180 | 0.9322 | 0.9130 | 0.8776 | 0.8950 | 0.8281 |
| KNN | 0.9215 | 0.9715 | 0.9063 | 0.8955 | 0.9009 | 0.8360 |
| Naive Bayes | 0.8288 | 0.9474 | 0.7099 | 0.9642 | 0.8177 | 0.6909 |
| Support Vector Machine | 0.9405 | 0.9777 | 0.9279 | 0.9224 | 0.9251 | 0.8759 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 92.63%, F1 0.9066, and AUC 0.9713. A useful linear baseline with fairly similar spam precision and recall. A linear boundary may miss interactions between features; this experiment does not isolate that effect. |
| Decision Tree | Accuracy 91.80%, F1 0.8950, and AUC 0.9322. Its feature-threshold rules are interpretable. The depth and leaf-size limits control complexity, but this single tree does not lead the measured comparison. |
| KNN | Accuracy 92.15%, F1 0.9009, and AUC 0.9715. Distance-weighted neighbors provide a competitive baseline after standardization. Similarity depends on the scaling and geometry of the 57-dimensional inputs. |
| Naive Bayes | Spam recall is 96.42%, the highest observed recall, while precision is 70.99% and F1 is 0.8177. It catches many spam emails at the cost of more false alarms. Gaussian likelihoods and conditional independence are strong assumptions for sparse, correlated frequency features. |
| Support Vector Machine | Accuracy 94.05%, the highest observed F1 of 0.9251, and AUC 0.9777. The standardized RBF model gives a strong measured balance of precision and recall. Its decision margins are used for AUC and are not calibrated probabilities. |
| Overall, winner for this test set? | **Support Vector Machine**, using F1 as the primary comparison measure (F1 0.9251, MCC 0.8759). F1 balances spam precision and recall; MCC breaks ties. This describes one holdout, not proven superiority on future email. |

Interpretations of model mechanisms above are plausible explanations, not causal findings established by this experiment. A single split has sampling variability. Use training-only cross-validation for an extended comparison and protect an appropriate final holdout.

### Metric Conventions

- **Accuracy:** fraction of all emails classified correctly.
- **Precision, Recall and F1:** binary scores with spam (`1`) as the positive class; undefined divisions return zero.
- **AUC:** calculated from continuous spam probabilities, or the SVM's decision margins, rather than hard predicted labels. A one-class upload has undefined AUC, displayed as `N/A`.
- **MCC:** uses all four confusion-matrix cells and ranges from -1 to 1.

## e. How to Run Locally

Use **Python 3.12**. This experiment was executed with Python 3.12.14; the model manifest records its exact library versions.

### 1. Clone the Repository

```bash
git clone https://github.com/riya2907/ml-assignment-2-spambase.git
cd ml-assignment-2-spambase
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
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

[Open deployed Streamlit app](https://inbox-lens-riya2907.streamlit.app/)

## Submission Status

- GitHub repository: available at the link above.
- Streamlit Community Cloud: app link recorded above.
- BITS Virtual Lab evidence: a user-provided screenshot of the app running inside the lab is included in the submission PDF.
- Final PDF: assemble the real GitHub link, live app link, lab screenshot, and full README content in that order after completing the required work.

See [`STUDY_GUIDE.md`](STUDY_GUIDE.md) for the learning walkthrough and submission steps. Keep an accurate record of your own work and assistance used; this repository is an AI-assisted learning example.

## References

- [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) and [dataset DOI](https://doi.org/10.24432/C53G6X).
- [scikit-learn documentation](https://scikit-learn.org/stable/) and [citation guidance](https://scikit-learn.org/stable/about.html#citing-scikit-learn).
- [Streamlit documentation](https://docs.streamlit.io/) and [dependency management](https://docs.streamlit.io/deploy/concepts/dependencies).
- [User-supplied VinoClass AI reference](https://github.com/arrjvvmukesh/BITS_ML_Assignment_2_Wine_ML_Classifier_VinoClass_AI_2026AJ05810): reference for README organization. This project's dataset, settings, results and wording are specific to Inbox Lens.
