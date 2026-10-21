"""Utility script to view all users, loan applications, and model predictions from the active database."""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from database.db import get_connection, get_db_engine, execute_sql


def print_table(title: str, headers: list, rows: list):
    print("\n" + "=" * 90)
    print(f"  {title.upper()}")
    print("=" * 90)
    if not rows:
        print("  (No records found)")
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val if val is not None else "")))

    # Format header
    header_str = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    print(header_str)
    print("-+-".join("-" * widths[i] for i in range(len(headers))))

    # Format rows
    for row in rows:
        print(" | ".join(f"{str(val if val is not None else ''):<{widths[i]}}" for i, val in enumerate(row)))


def main():
    engine = get_db_engine()
    print("=" * 90)
    print(f">> ACTIVE DATABASE ENGINE: {engine.upper()}")
    if engine == "sqlite":
        print(f">> FILE PATH: {BASE_DIR / 'database' / 'loan_approval.db'}")
    else:
        print(">> MYSQL DATABASE: loan_approval_db")
    print("=" * 90)

    # 1. Users
    users_sql = "SELECT id, full_name, email, is_admin, created_at FROM users ORDER BY id ASC"
    user_rows = execute_sql(users_sql, fetchall=True)
    user_data = []
    for u in user_rows:
        user_data.append([
            u["id"],
            u["full_name"],
            u["email"],
            "Yes (Admin)" if u["is_admin"] else "No (User)",
            u["created_at"]
        ])
    print_table("1. Registered Users Table (`users`)", ["ID", "Full Name", "Email Address", "Role", "Registered At"], user_data)

    # 2. Loan Applications
    apps_sql = """
        SELECT a.id, a.user_id, u.full_name, a.no_of_dependents, a.education, a.self_employed,
               a.income_annum, a.loan_amount, a.loan_term, a.cibil_score, a.created_at
        FROM loan_applications a
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.id ASC
    """
    app_rows = execute_sql(apps_sql, fetchall=True)
    app_data = []
    for a in app_rows:
        app_data.append([
            f"#{a['id']}",
            a['full_name'] or f"User {a['user_id']}",
            f"{a['no_of_dependents']} dep",
            a['education'],
            a['self_employed'],
            f"Rs. {a['income_annum']:,.0f}",
            f"Rs. {a['loan_amount']:,.0f}",
            f"{a['loan_term']} yrs",
            a['cibil_score'],
            a['created_at']
        ])
    print_table("2. Loan Applications Table (`loan_applications`)", 
                ["App ID", "Applicant", "Dependents", "Education", "Self-Emp", "Income", "Loan Req", "Term", "CIBIL", "Submitted At"], 
                app_data)

    # 3. Model Predictions History
    preds_sql = """
        SELECT p.application_id, a.user_id, p.model_name, p.predicted_class, p.approved_probability, p.created_at
        FROM prediction_history p
        JOIN loan_applications a ON p.application_id = a.id
        ORDER BY p.application_id ASC, p.id ASC
    """
    pred_rows = execute_sql(preds_sql, fetchall=True)
    pred_data = []
    for p in pred_rows:
        prob_str = f"{p['approved_probability'] * 100:.1f}%" if p['approved_probability'] is not None else "--"
        pred_data.append([
            f"#{p['application_id']}",
            p['model_name'],
            p['predicted_class'],
            prob_str,
            p['created_at']
        ])
    print_table("3. Multi-Model Predictions Table (`prediction_history`)",
                ["App ID", "Algorithm Name", "Predicted Outcome", "Approval Probability", "Evaluated At"],
                pred_data)
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
