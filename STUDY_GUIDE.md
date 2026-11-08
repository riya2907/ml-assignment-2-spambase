# Start here: Assignment 2

This project is a worked learning example, not a finished submission. The assignment permits AI learning support and disallows direct copy-paste submissions. Read, explain, and adapt the experiment, then run your version in BITS Virtual Lab. The brief's deadline is 8 November 2026, 23:59; use the course portal's official clock.

## What you need to understand

Start with `core.py`, then `train.py`, then `app.py`.

1. **Features and labels.** Each row describes one email. `X` contains 57 measured inputs. `y` is `spam`, where 1 means spam. The label must never be included among model inputs.
2. **Cleaning and splitting.** The raw dataset has duplicates. Random splitting before removing them can put identical examples on both sides and overstate generalization. This example first removes feature vectors with contradictory labels, then keeps one row per remaining feature vector, and uses a stratified split to preserve approximate class proportions. This policy is a choice you should justify.
3. **Preprocessing.** Capital-run counts can be much larger than percentages. KNN and RBF SVM depend on distances; Logistic Regression also benefits from scaling. StandardScaler learns means and standard deviations only on training rows. Saving it in a Pipeline makes the app apply the same transformation.
4. **Training and testing.** `fit` learns from training rows. `predict` and `decision_function`/`predict_proba` evaluate unseen inputs. The app never calls `fit`, so an upload cannot change the saved models.
5. **Fixed model settings.** These are transparent baselines, not claimed optimal parameters. For an extension, use stratified cross-validation inside the training set. Do not repeatedly tune on the reported test set.
6. **Reproducibility.** The random seed, source hash, test CSV hash and package versions are recorded in `model/manifest.json`. Metrics are computed from the saved test CSV so the app can reproduce the README.

## How the five models work

| Model | Intuition | Main question to consider |
| --- | --- | --- |
| Logistic Regression | Learns a weighted sum of standardized features and converts it to a probability. | Is one linear boundary sufficient? |
| Decision Tree | Repeatedly splits on feature thresholds. | Does a deep tree memorize rare examples? |
| KNN | Votes using nearby standardized training emails. | Are distances informative with 57 features? |
| Gaussian Naive Bayes | Combines per-feature Gaussian likelihoods under conditional independence. | Are sparse word percentages Gaussian or independent? |
| RBF SVM | Finds a nonlinear margin using similarity to support vectors. | How do C and gamma change boundary complexity? |

## Interpret the six metrics

Imagine 100 test emails: TP=30, FP=5, TN=55, FN=10. Positive means spam.

| Metric | Calculation | Result |
| --- | --- | --- |
| Accuracy | (TP+TN)/N | 0.8500 |
| Precision | TP/(TP+FP) | 0.8571 |
| Recall | TP/(TP+FN) | 0.7500 |
| F1 | 2TP/(2TP+FP+FN) | 0.8000 |
| MCC | (TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | about 0.6847 |
| AUC | Probability that a random positive ranks above a random negative (ties get half credit). | Cannot be calculated from this one confusion matrix alone. |

False positives are legitimate mail classified as spam. False negatives are spam that slips through. Precision and recall describe different costs. AUC uses continuous scores across thresholds; feeding hard predicted labels to ROC AUC discards ranking information. SVM decision margins are valid for AUC, but are not calibrated probabilities.

## Run in BITS Virtual Lab

Use the lab interface provided by the course; no lab execution has happened in this local workspace.

1. Copy your adapted project to BITS Virtual Lab. Keep `data/`, `model/`, and `test_data.csv` with the source. Do not upload `.packages`, `.venv`, or `tmp`.
2. Open a terminal in the project folder. Use Python 3.12 and follow the environment commands in README.
3. Run `python train.py`. Review the printed metrics and regenerated `results/metrics.csv` and README.
4. Run `python verify.py`, then `python -m streamlit run app.py`. Use the lab's preview or forwarded port to open the app.
5. Upload `test_data.csv`. Switch among all five models and confirm the table agrees with `results/metrics.csv` to four decimal places.
6. Capture ONE genuine screenshot showing the BITS Virtual Lab interface and your execution results or running app. Keep enough of the lab interface visible to establish its origin; a local app screenshot is not BITS evidence.

If lab access fails, the assignment gives support contacts and an email subject on page 1. Follow those instructions yourself or explicitly ask for help drafting the message.

## Publish your repository and deploy

After you have understood and adapted the project, use your own GitHub repository for the source, requirements, README, test CSV, saved models, manifest, and results. Use genuine commits as you develop. The repository link, when configured, is shown in README.

In [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**, choose your repository and branch, set the entrypoint to `app.py`, select Python 3.12 in Advanced settings, and deploy. If you upload this folder beneath another repository directory, use that path, for example `assignment2/app.py`. Keep requirements alongside the entrypoint. The application needs no dataset download or training during startup.

Open the resulting real `.streamlit.app` URL, upload the test CSV, exercise the model selector, and inspect metrics and confusion matrix. Add the real app link to README after your final training run. The repository link is preserved through `project_links.json` when configured, but repeat other personal README edits after rerunning `train.py`, which regenerates that document.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Build the single submission PDF

Use the following exact order from the assignment. Do not submit placeholders.

1. **GitHub repository link:** your real, clickable URL.
2. **Live Streamlit app link:** the deployed, tested, interactive app URL.
3. **BITS Virtual Lab screenshot:** the genuine screenshot captured above.
4. **README content:** problem statement, dataset description, repository link, all-five-model metric table, per-model observations and the overall winner discussion. Use the latest lab-generated results.

Before submitting, open the final PDF, click both links, check that tables are readable, and ensure all README content is present. A starter, template or draft is not the final submission. The official portal submission is a separate step.

## Explain it in your own words

Before you submit, be able to answer:

- Why does the cleaned instance count differ from 4,601?
- Why is `spam` excluded from input features?
- Why is the scaler fitted only on training rows?
- Why does AUC need scores rather than hard labels?
- Which model had the highest test F1, and did it also have the highest AUC?
- What happens to AUC if an uploaded CSV contains only non-spam emails?
- Why does an uploaded training CSV give an unreliable estimate of generalization?
- Why might a model that works on historical Spambase perform poorly on modern email?

Your own justified changes, interpretation, and actual lab execution are the important next steps.
