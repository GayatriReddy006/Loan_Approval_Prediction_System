# 🏦 CreditPulse AI: Loan Approval Prediction System
### *Supervised Machine Learning for Automated Credit Risk Classification*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://palletsprojects.com/p/flask/)
[![MySQL](https://img.shields.io/badge/Database-MySQL%20%7C%20SQLite-blue.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/Academic-Project-purple.svg)]()

An enterprise-grade Data Science and Machine Learning college project built with **Python, Scikit-Learn, Flask, MySQL, and modern Vanilla Web technologies**. The application evaluates borrower creditworthiness across three distinct machine learning algorithms simultaneously, featuring secure user authentication, application history tracking, explainable AI drivers, and an administrative telemetry dashboard.

---

## 📑 Table of Contents
1. [Project Overview & Key Features](#-project-overview--key-features)
2. [Technology Architecture](#-technology-architecture)
3. [Which Machine Learning Algorithms Are Used Here?](#-which-machine-learning-algorithms-are-used-here)
4. [Empirical Model Comparison & Benchmarks](#-empirical-model-comparison--benchmarks)
5. [Feature Importance & Risk Drivers](#-feature-importance--risk-drivers)
6. [Exact Project Directory Structure](#-exact-project-directory-structure)
7. [Step-by-Step Installation & Setup (Windows & VS Code)](#-step-by-step-installation--setup-windows--vs-code)
8. [MySQL Database Setup](#-mysql-database-setup)
9. [Model Retraining & Testing](#-model-retraining--testing)
10. [REST API Documentation](#-rest-api-documentation)
11. [Troubleshooting Common Issues](#-troubleshooting-common-issues)
12. [Ethical AI & UN Sustainable Development Goals](#-ethical-ai--un-sustainable-development-goals)

---

## 🎯 Project Overview & Key Features

* **Multi-Model Decisioning:** Evaluates each loan application across **Logistic Regression**, **Decision Tree (CART)**, and **Random Forest** to calculate consensus and probability distribution.
* **User Authentication & Session Management:** Secure user registration, login, session cookies, and PBKDF2:SHA-256 password hashing (Werkzeug).
* **Personalized User Dashboard:** Track previous applications, review consensus decisions, and inspect individual model predictions.
* **User Isolation & Security:** Parameterized SQL queries ensure applicants can only view their own records.
* **Interactive Live Screener:** Features dynamic CIBIL credit score meter, live collateral-to-debt ratio gauge, and one-click applicant persona presets.
* **Explainable AI (XAI):** Gini importance breakdown highlighting why an applicant was approved or rejected (CIBIL score bracket, collateral coverage).
* **Administrator Portal:** Global analytics showing total users, application volume, model outcome distribution, and keyword search across applicant records.
* **Dual Deployment Architecture:** Operates both as a full **Flask + MySQL REST API service** and as a **standalone client-side offline dashboard**.

---

## 🏗️ Technology Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Frontend Layer (Web UI)                         │
│   HTML5 · Vanilla CSS Design System · Responsive Cards · Google Fonts │
│       Static Inference Fallback (predictor.js) · Dynamic Modals        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST API
┌───────────────────────────────────▼────────────────────────────────────┐
│                       Backend Layer (Python Flask)                     │
│    Session Auth · Input Validation · Scikit-Learn Pipeline Runner      │
└─────────────────┬──────────────────────────────────┬───────────────────┘
                  │                                  │
┌─────────────────▼──────────────┐   ┌───────────────▼───────────────────┐
│     Machine Learning Engine    │   │          Database Layer           │
│   Random Forest (200 Trees)    │   │        MySQL Database Service     │
│   Decision Tree (CART Gini)    │   │    (Automatic SQLite Fallback)    │
│   Logistic Regression (Sigmoid)│   │  Tables: users, loan_applications │
│   Pipelines: Joblib Binaries   │   │          prediction_history       │
└────────────────────────────────┘   └───────────────────────────────────┘
```

---

## 🧠 Which Machine Learning Algorithms Are Used Here?

The system evaluates loan eligibility across **three supervised classification algorithms**, comparing linear decision boundaries, single rule-based trees, and ensemble forests on the exact same stratified test set ($N_{\text{test}} = 854$ applicants):

### 1. 🌲 Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`) — *Champion Model*
* **Algorithmic Category:** Supervised Ensemble Learning via **Bootstrap Aggregation (Bagging)**.
* **How it works:**
  * Builds an ensemble of $B = 200$ independent decision trees trained on bootstrap samples of the training set.
  * At each node split, a random subset of features ($\sqrt{p}$) is evaluated to maximize tree diversity and de-correlate errors.
  * The final classification probability is obtained by averaging the posterior distributions across all 200 trees:
    $$\hat{P}(Y = \text{Approved} \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} P_b(Y = \text{Approved} \mid \mathbf{x})$$
* **Why it performs best:**
  * Cancels out individual tree variance by $1/B$ without increasing bias, making it robust against outliers and noise.
  * Delivers **98.13% Accuracy**, **99.25% Approval Recall**, and **0.997 ROC-AUC** on the held-out test set.

### 2. 🌳 Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`) — *Rule-Based Model*
* **Algorithmic Category:** Non-parametric recursive binary partitioning (**CART** algorithm).
* **How it works:**
  * Recursively chooses feature splits that maximize information gain by minimizing **Gini Impurity**:
    $$I_G(t) = 1 - \sum_{i=1}^{C} p(i \mid t)^2$$
  * Produces axis-parallel orthogonal decision boundaries (e.g., `if cibil_score > 549.5 and loan_term <= 10`).
* **Role in banking:** Provides transparent white-box logic, allowing credit underwriters to trace explicit reasoning behind an assessment. Delivers **97.31% Accuracy** and **97.84% F1-score**.

### 3. 📈 Logistic Regression (`sklearn.linear_model.LogisticRegression`) — *Linear Baseline*
* **Algorithmic Category:** Generalized Linear Model (GLM) with a logistic (sigmoid) link function.
* **How it works:**
  * Calculates a weighted linear sum of normalized inputs and passes it through the sigmoid function:
    $$z = \beta_0 + \sum_{j=1}^{p} \beta_j x_j, \quad P(Y = \text{Approved} \mid \mathbf{x}) = \frac{1}{1 + e^{-z}}$$
  * Standardizes all 9 continuous inputs using `StandardScaler` ($\mu=0, \sigma=1$) and one-hot encodes categorical variables.
  * Fits weights by minimizing L2-regularized Cross-Entropy (Log-Loss) using the `lbfgs` solver (`max_iter=2000`).
* **Role in banking:** Regulatory benchmark for credit scorecards; provides direct log-odds interpretations. Delivers **92.27% Accuracy** and **97.50% ROC-AUC**.

---

## 📊 Empirical Model Comparison & Benchmarks

All metrics were computed dynamically on the exact same stratified 20% test split ($N = 854$ records, positive class: `Approved`):

| Classifier Algorithm | Test Accuracy | Precision (Approved) | Recall (Approved) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Ensemble)** | **98.13%** | **97.77%** | **99.25%** | **98.50%** | **0.997** |
| **Decision Tree (CART)** | 97.31% | 97.39% | 98.31% | 97.84% | 0.970 |
| **Logistic Regression** | 92.27% | 92.66% | 95.10% | 93.87% | 0.975 |

### Confusion Matrix Breakdown ($N = 854$)
* **Random Forest:** True Positives = 527, False Negatives = 4, False Positives = 12, True Negatives = 311.
* **Approval Recall:** $\frac{527}{527 + 4} = 99.25\%$ (identifies eligible borrowers with minimal Type II error).

---

## 🔍 Feature Importance & Risk Drivers

Extracted from the fitted Random Forest model via Gini importance:

```text
CIBIL Credit Score         ████████████████████████████████████ 78.2%
Loan Term Duration         ███ 6.1%
Requested Loan Amount      █ 3.2%
Annual Income              █ 2.2%
Luxury Assets Value        █ 2.2%
Commercial Assets Value    █ 2.2%
Residential Assets Value   █ 2.1%
Bank Liquid Asset Value    █ 1.9%
Number of Dependents       ▌ 1.0%
Self Employed Status       ▏ 0.3%
Education Level            ▏ 0.2%
```

1. **CIBIL Score (78.2%):** The primary driver. Scores below 550 result in rejection across all three models.
2. **Loan Term (6.1%):** Shorter terms (2–6 years) present lower default exposure than extended 15–20 year terms.
3. **Collateral Coverage (11.5% combined assets):** Cumulative declared asset valuations serve as secondary security for borderline applicants.

---

## 📁 Exact Project Directory Structure

```text
Loan_Approval_Prediction_System/
├── app.py                      # Flask application: routes, APIs, auth, model inference
├── config.py                   # Configuration module reading from .env
├── train_model.py              # ML pipeline: data audit, training, evaluation, export
├── requirements.txt            # Python dependencies list
├── .env.example                # Template for environment configuration
├── .env                        # Local environment variables (DB credentials, secret key)
├── README.md                   # Comprehensive technical documentation
├── index.html                  # Responsive modern banking web dashboard
│
├── database/
│   ├── schema.sql              # MySQL database schema (users, apps, predictions)
│   ├── db.py                   # Resilient database manager (MySQL with SQLite fallback)
│   └── loan_approval.db        # Auto-generated SQLite database (fallback)
│
├── static/
│   ├── styles.css              # Custom fintech design system & responsive styling
│   ├── script.js               # Reactive frontend logic, APIs, auth, charts
│   ├── predictor.js            # Offline client-side JavaScript prediction engine
│   └── model_data.js           # Exported scikit-learn weights and tree structures
│
├── data/
│   └── loan_data.csv           # Benchmark dataset (4,269 applicant records)
│
├── models/
│   ├── decision_tree.joblib    # Serialized scikit-learn Decision Tree pipeline
│   ├── logistic_regression.joblib # Serialized scikit-learn Logistic Regression pipeline
│   ├── random_forest.joblib    # Serialized scikit-learn Random Forest pipeline
│   └── metadata.json           # Feature schema, categorical levels, and defaults
│
├── results/
│   ├── confusion_matrices.png  # Confusion matrix heatmaps for all 3 models
│   ├── evaluation_results.json # JSON benchmark metrics and confusion matrices
│   ├── feature_importance.png  # Gini feature importance bar chart
│   ├── model_comparison.png   # Performance comparison bar chart (Accuracy, F1, AUC)
│   └── model_metrics.csv       # CSV metrics summary
│
└── tests/
    └── test_app.py             # Automated test suite (7 comprehensive test cases)
```

---

## 🚀 Step-by-Step Installation & Setup (Windows & VS Code)

### Prerequisites
* **Python 3.10 or newer** (tested on Python 3.13)
* **MySQL Server** (optional; application auto-falls back to SQLite if MySQL is offline)
* Modern web browser (Chrome, Edge, Firefox, Brave)

### 1. Open Terminal in VS Code
Open the project directory in VS Code or PowerShell:
```powershell
cd c:\Users\gayat\Downloads\Loan_Approval_Prediction_System\Loan_Approval_Prediction_System
```

### 2. Set Up Virtual Environment & Dependencies
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Open `.env` and set your MySQL password if using MySQL:
```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=loan_approval_db
SECRET_KEY=my-super-secret-key-2026
```

---

## 🗄️ MySQL Database Setup

1. Start your local MySQL server (via MySQL Workbench, XAMPP, or Windows Service).
2. Create the database and tables using `database/schema.sql`:
```powershell
mysql -u root -p < database\schema.sql
```
*(If MySQL is not installed or configured, the system seamlessly uses the included SQLite driver `database/loan_approval.db` without crashing).*

3. Optional: Create an Administrator account via CLI:
```powershell
python app.py --create-admin admin@creditpulse.bank SecretPassword123 "Lead Risk Officer"
```

---

## 🧪 Model Retraining & Testing

### Retrain Models from Scratch
To clean data, fit all 3 algorithms, evaluate test splits, and regenerate all figures:
```powershell
python train_model.py
```

### Run Automated Test Suite
To verify dataset integrity, pipeline predictions, authentication, and user isolation:
```powershell
python -m unittest tests/test_app.py
```

---

## 🌐 Running the Application

### Start the Flask Server
```powershell
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### Standalone Offline Mode
You can also directly open `index.html` in any browser. `static/predictor.js` computes full multi-model predictions locally in JavaScript without requiring a server.

---

## 📡 REST API Documentation

| Method | Endpoint | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/register` | Register a new user account | Public |
| `POST` | `/api/login` | Authenticate user & start session | Public |
| `POST` | `/api/logout` | Terminate session | Authenticated |
| `GET` | `/api/me` | Check session authentication status | Public |
| `POST` | `/api/predict` | Evaluate applicant across 3 ML models | Public / User |
| `GET` | `/api/applications` | Get logged-in user's application history | Authenticated |
| `GET` | `/api/applications/<id>` | Get application features & 3 model outputs | Owner / Admin |
| `GET` | `/api/models/metrics` | Retrieve evaluation scores and feature importances | Public |
| `GET` | `/api/admin/stats` | System telemetry (users, apps, predictions) | Admin Only |
| `GET` | `/api/admin/applications` | Global application search and inspection | Admin Only |

---

## ❓ Troubleshooting Common Issues

* **Port 5000 Already in Use:**  
  Run `python app.py` on a different port by editing the last line in `app.py`: `app.run(port=5001)`.
* **MySQL Access Denied (Error 1045):**  
  Update `DB_PASSWORD` in `.env` to match your local MySQL root password, or let the app run on its automatic SQLite fallback.
* **PowerShell Execution Policy Error on Activate.ps1:**  
  Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell, then re-run `.venv\Scripts\Activate.ps1`.

---

## 🌍 Ethical AI & UN Sustainable Development Goals

* **SDG 1 (No Poverty):** Transparent algorithmic scoring democratizes access to financial capital for qualified applicants.
* **SDG 8 (Decent Work & Economic Growth):** Objective risk analysis reduces subjective bias in small enterprise credit decisions.

> **⚠️ Academic & Regulatory Disclaimer:**  
> This project is designed for educational and benchmarking demonstration. Models trained on historical data may inherit institutional biases. In production retail banking, automated credit decisioning must comply with fair lending regulations (e.g. ECOA, FCRA), include adverse action notices, and undergo regular fair-lending algorithmic audits.
