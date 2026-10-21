"""Utility script to export all new loan applications from the database into a CSV dataset file."""

import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from database.db import execute_sql

def main():
    out_file = BASE_DIR / "data" / "new_entries_dataset.csv"
    sql = """
        SELECT a.id as application_id, a.user_id, u.full_name, u.email,
               a.no_of_dependents, a.education, a.self_employed,
               a.income_annum, a.loan_amount, a.loan_term, a.cibil_score,
               a.residential_assets_value, a.commercial_assets_value,
               a.luxury_assets_value, a.bank_asset_value, a.created_at
        FROM loan_applications a
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.id ASC
    """
    rows = execute_sql(sql, fetchall=True)
    if not rows:
        print("No loan application entries found in the database.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("=" * 70)
    print(f">> EXPORT SUCCESSFUL: {len(rows)} entries exported.")
    print(f">> FILE PATH: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
