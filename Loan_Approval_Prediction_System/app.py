"""Root launcher for Loan Approval Prediction System."""

import os
import sys
from pathlib import Path

# Set working directory to the inner project folder so all paths resolve correctly
inner_dir = Path(__file__).resolve().parent / "Loan_Approval_Prediction_System"
os.chdir(inner_dir)
sys.path.insert(0, str(inner_dir))

# Import and execute the Flask server
from app import app, get_db_engine

if __name__ == "__main__":
    print("=" * 65)
    print(">> CreditPulse AI - Loan Approval Prediction System Web Server")
    print(">> URL: http://127.0.0.1:5000")
    print(f">> Database Engine: {get_db_engine().upper()}")
    print(">> Models Active: Random Forest (98.1%), Decision Tree (97.3%), Logistic Regression (92.3%)")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=True)
