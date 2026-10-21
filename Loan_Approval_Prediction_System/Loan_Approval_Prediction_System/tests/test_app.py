"""Automated test suite for the Loan Approval Prediction System.

Tests dataset integrity, Scikit-Learn model predictions, input validation,
user authentication, session security, database operations, and user isolation.
"""

import unittest
import json
from pathlib import Path
import pandas as pd
import joblib

from config import Config
from app import app
from database.db import init_db


class TestLoanApprovalSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        init_db()

    # 1. Dataset Integrity Tests
    def test_dataset_exists_and_valid(self):
        csv_path = Config.DATA_DIR / "loan_data.csv"
        self.assertTrue(csv_path.exists(), "Dataset file data/loan_data.csv must exist.")
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 4000, "Dataset should have >4000 records.")
        
        # Check target column
        columns = [c.strip().lower() for c in df.columns]
        self.assertIn("loan_status", columns, "Dataset must contain loan_status column.")

    # 2. Machine Learning Model Pipelines Tests
    def test_all_three_models_exist_and_predict(self):
        sample_applicant = pd.DataFrame({
            "no_of_dependents": [2],
            "education": ["Graduate"],
            "self_employed": ["No"],
            "income_annum": [9500000.0],
            "loan_amount": [25000000.0],
            "loan_term": [10],
            "cibil_score": [820],
            "residential_assets_value": [18000000.0],
            "commercial_assets_value": [12000000.0],
            "luxury_assets_value": [24000000.0],
            "bank_asset_value": [8500000.0]
        })

        for model_name in ["logistic_regression", "decision_tree", "random_forest"]:
            model_path = Config.MODELS_DIR / f"{model_name}.joblib"
            self.assertTrue(model_path.exists(), f"Model {model_name}.joblib must exist.")
            pipeline = joblib.load(model_path)
            
            # Predict outcome
            pred = pipeline.predict(sample_applicant)[0]
            self.assertIn(str(pred).strip(), ["Approved", "Rejected"])
            
            # Predict probabilities
            proba = pipeline.predict_proba(sample_applicant)[0]
            self.assertEqual(len(proba), 2)
            self.assertAlmostEqual(sum(proba), 1.0, places=2)

    # 3. Prediction API & Input Validation Tests
    def test_prediction_api_success(self):
        payload = {
            "no_of_dependents": 1,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 9500000,
            "loan_amount": 25000000,
            "loan_term": 10,
            "cibil_score": 820,
            "residential_assets_value": 18000000,
            "commercial_assets_value": 12000000,
            "luxury_assets_value": 24000000,
            "bank_asset_value": 8500000
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("consensus", data)
        self.assertIn("predictions", data)
        self.assertIn("random_forest", data["predictions"])
        self.assertIn("decision_tree", data["predictions"])
        self.assertIn("logistic_regression", data["predictions"])

    def test_prediction_api_validation_errors(self):
        # Test missing field
        res = self.client.post("/api/predict", json={"income_annum": 5000000})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Missing required fields", json.loads(res.data)["message"])

        # Test invalid CIBIL score (< 300)
        invalid_payload = {
            "no_of_dependents": 1,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000,
            "loan_amount": 10000000,
            "loan_term": 10,
            "cibil_score": 250,  # Invalid
            "residential_assets_value": 5000000,
            "commercial_assets_value": 2000000,
            "luxury_assets_value": 5000000,
            "bank_asset_value": 2000000
        }
        res = self.client.post("/api/predict", json=invalid_payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("CIBIL", json.loads(res.data)["message"])

    # 4. Authentication Flow Tests
    def test_user_registration_login_logout(self):
        email = f"test_{pd.Timestamp.now().value}@bank.com"
        reg_payload = {
            "full_name": "Test Applicant",
            "email": email,
            "password": "securePassword123",
            "confirm_password": "securePassword123"
        }
        # Register
        res = self.client.post("/api/register", json=reg_payload)
        self.assertEqual(res.status_code, 201)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")

        # Check /api/me while authenticated
        res_me = self.client.get("/api/me")
        data_me = json.loads(res_me.data)
        self.assertTrue(data_me["authenticated"])
        self.assertEqual(data_me["user"]["email"], email)

        # Logout
        res_logout = self.client.post("/api/logout")
        self.assertEqual(res_logout.status_code, 200)

        # Check /api/me after logout
        res_me_after = self.client.get("/api/me")
        self.assertFalse(json.loads(res_me_after.data)["authenticated"])

        # Login again
        login_payload = {"email": email, "password": "securePassword123"}
        res_login = self.client.post("/api/login", json=login_payload)
        self.assertEqual(res_login.status_code, 200)
        self.assertEqual(json.loads(res_login.data)["status"], "success")

    # 5. User Data Isolation Tests
    def test_user_data_isolation(self):
        # Register User A
        email_a = f"user_a_{pd.Timestamp.now().value}@bank.com"
        self.client.post("/api/register", json={
            "full_name": "User Alpha", "email": email_a,
            "password": "PasswordA123", "confirm_password": "PasswordA123"
        })

        # Submit an application as User A
        payload = {
            "no_of_dependents": 2, "education": "Graduate", "self_employed": "Yes",
            "income_annum": 6000000, "loan_amount": 15000000, "loan_term": 10,
            "cibil_score": 750, "residential_assets_value": 8000000,
            "commercial_assets_value": 4000000, "luxury_assets_value": 10000000,
            "bank_asset_value": 5000000
        }
        res_pred = self.client.post("/api/predict", json=payload)
        app_id_a = json.loads(res_pred.data)["application_id"]
        self.assertIsNotNone(app_id_a)

        # Check User A can view this application
        res_detail_a = self.client.get(f"/api/applications/{app_id_a}")
        self.assertEqual(res_detail_a.status_code, 200)

        # Logout and register User B
        self.client.post("/api/logout")
        email_b = f"user_b_{pd.Timestamp.now().value}@bank.com"
        self.client.post("/api/register", json={
            "full_name": "User Beta", "email": email_b,
            "password": "PasswordB123", "confirm_password": "PasswordB123"
        })

        # Attempt to access User A's application as User B
        res_detail_b = self.client.get(f"/api/applications/{app_id_a}")
        self.assertEqual(res_detail_b.status_code, 404, "User B must not access User A's record.")

    # 6. Evaluation Metrics API Test
    def test_metrics_api_endpoint(self):
        res = self.client.get("/api/models/metrics")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["positive_class"], "Approved")
        self.assertEqual(len(data["metrics"]), 3)
        for m in data["metrics"]:
            self.assertIn("accuracy", m)
            self.assertIn("precision_approved", m)
            self.assertIn("recall_approved", m)
            self.assertIn("f1_approved", m)
            self.assertIn("roc_auc", m)


if __name__ == "__main__":
    unittest.main()
