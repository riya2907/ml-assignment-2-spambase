"""Meaningful checks: input contract, metric definitions, saved artifacts and app flows."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from core import FEATURES, TARGET, METRICS, evaluate, read_test_csv

ROOT = Path(__file__).resolve().parent


class EvaluationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (ROOT / "test_data.csv").read_bytes()
        cls.frame = read_test_csv(cls.raw)
        cls.manifest = json.loads((ROOT / "model" / "manifest.json").read_text())

    def test_metric_definitions_with_known_predictions(self):
        class KnownClassifier:
            classes_ = np.array([0, 1])
            def predict(self, X):
                return np.array([0, 1, 1, 0])
            def predict_proba(self, X):
                scores = np.array([.1, .8, .6, .4])
                return np.column_stack([1 - scores, scores])
        frame = self.frame.iloc[:4].copy()
        frame[TARGET] = [0, 0, 1, 1]
        result, _, _ = evaluate(KnownClassifier(), frame)
        for key in ["Accuracy", "AUC", "Precision", "Recall", "F1"]:
            self.assertAlmostEqual(result[key], .5)
        self.assertAlmostEqual(result["MCC"], 0)

    def test_models_reproduce_report_and_have_no_split_overlap(self):
        report = pd.read_csv(ROOT / "results" / "metrics.csv").set_index("ML Model Name")
        for name, file in self.manifest["models"].items():
            metrics, _, _ = evaluate(joblib.load(ROOT / "model" / file), self.frame)
            np.testing.assert_allclose([metrics[m] for m in METRICS], report.loc[name, METRICS].to_numpy(float), atol=1e-12)
        train = pd.read_csv(ROOT / "data" / "train_data.csv")
        train_rows = set(map(tuple, train[FEATURES].to_numpy()))
        test_rows = set(map(tuple, self.frame[FEATURES].to_numpy()))
        self.assertFalse(train_rows & test_rows)

    def test_reordered_columns_and_one_class(self):
        reversed_csv = self.frame[self.frame.columns[::-1]].to_csv(index=False).encode()
        pd.testing.assert_frame_equal(read_test_csv(reversed_csv), self.frame)
        model = joblib.load(ROOT / "model" / next(iter(self.manifest["models"].values())))
        metrics, _, _ = evaluate(model, self.frame[self.frame[TARGET] == 0])
        self.assertTrue(np.isnan(metrics["AUC"]))

    def test_invalid_csvs_rejected(self):
        invalid = [b"", b"a,a\n1,2\n", b"\xff\xfe", self.frame.drop(columns=[FEATURES[0]]).to_csv(index=False).encode()]
        for column, value in [(TARGET, 2), (FEATURES[0], np.nan), (FEATURES[0], -1), (FEATURES[0], "bad")]:
            bad = self.frame.iloc[:4].copy().astype(object)
            bad.loc[bad.index[0], column] = value
            invalid.append(bad.to_csv(index=False).encode())
        for data in invalid:
            with self.subTest(data=data[:50]), self.assertRaises(ValueError):
                read_test_csv(data)

    def test_streamlit_default_selection_and_upload_flows(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0, str(at.exception))
        self.assertEqual(len(at.dataframe[0].value), 5)
        for name in self.manifest["models"]:
            at.selectbox[0].select(name).run()
            self.assertEqual(len(at.exception), 0, str(at.exception))
        # AppTest has no native upload action; patch only the file input and execute
        # the real app with valid, single-class and invalid CSVs.
        for data, expected_error in [(self.raw, False),
                (self.frame[self.frame[TARGET] == 0].to_csv(index=False).encode(), False),
                (b"wrong,spam\n1,0\n", True)]:
            with patch("streamlit.file_uploader", return_value=io.BytesIO(data)):
                uploaded_app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertEqual(len(uploaded_app.exception), 0, str(uploaded_app.exception))
            self.assertEqual(bool(len(uploaded_app.error)), expected_error)
            if not expected_error:
                self.assertEqual(len(uploaded_app.dataframe[0].value), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
