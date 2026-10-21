"""Root launcher to view database records."""
import os, sys
from pathlib import Path
inner_dir = Path(__file__).resolve().parent / "Loan_Approval_Prediction_System"
os.chdir(inner_dir)
sys.path.insert(0, str(inner_dir))
from view_database import main
if __name__ == "__main__":
    main()
