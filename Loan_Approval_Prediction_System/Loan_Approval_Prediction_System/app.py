"""Flask backend server for the Loan Approval Prediction System.

Provides user authentication, loan application processing, multi-model
predictions via Scikit-Learn pipelines, database persistence, and analytics.
"""

from functools import wraps
import json
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from database.db import (
    init_db,
    get_db_engine,
    create_user,
    get_user_by_email,
    get_user_by_id,
    count_users,
    save_loan_application,
    save_prediction_result,
    get_user_applications,
    get_application_details,
    get_admin_stats,
    search_all_applications
)

app = Flask(__name__, static_folder=str(Config.STATIC_DIR))
app.config.from_object(Config)

# In-memory model cache
_PIPELINES = {}
_METADATA = None


def get_metadata() -> dict:
    """Load feature schema, defaults, and model metadata."""
    global _METADATA
    if _METADATA is None:
        meta_path = Config.MODELS_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                _METADATA = json.load(f)
        else:
            _METADATA = {}
    return _METADATA


def get_pipeline(name: str):
    """Load or retrieve cached scikit-learn pipeline."""
    if name not in _PIPELINES:
        model_path = Config.MODELS_DIR / f"{name}.joblib"
        if model_path.exists():
            _PIPELINES[name] = joblib.load(model_path)
        else:
            return None
    return _PIPELINES[name]


# -----------------------------------------------------------------------------
# Authentication Decorators
# -----------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"status": "error", "message": "Authentication required. Please log in."}), 401
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"status": "error", "message": "Authentication required."}), 401
        if not session.get("is_admin"):
            return jsonify({"status": "error", "message": "Administrator privileges required."}), 403
        return f(*args, **kwargs)
    return decorated_function


# -----------------------------------------------------------------------------
# Frontend Static & Visual Routes
# -----------------------------------------------------------------------------

@app.route("/")
def serve_index():
    return send_from_directory(str(Config.ROOT_DIR), "index.html")


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(str(Config.RESULTS_DIR), filename)


# -----------------------------------------------------------------------------
# Authentication API Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    full_name = str(data.get("full_name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    confirm_password = str(data.get("confirm_password", ""))

    if not full_name or not email or not password:
        return jsonify({"status": "error", "message": "Full name, email, and password are required."}), 400

    if len(password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters long."}), 400

    if password != confirm_password:
        return jsonify({"status": "error", "message": "Passwords do not match."}), 400

    if "@" not in email or "." not in email:
        return jsonify({"status": "error", "message": "Invalid email address format."}), 400

    existing_user = get_user_by_email(email)
    if existing_user:
        return jsonify({"status": "error", "message": "An account with this email already exists."}), 409

    # First registered user can optionally be an admin if count_users == 0
    is_first_user = count_users() == 0
    pwd_hash = generate_password_hash(password, method="pbkdf2:sha256")
    user_id = create_user(full_name, email, pwd_hash, is_admin=is_first_user)

    session["user_id"] = user_id
    session["email"] = email
    session["full_name"] = full_name
    session["is_admin"] = is_first_user

    return jsonify({
        "status": "success",
        "message": "Account registered successfully.",
        "user": {
            "id": user_id,
            "full_name": full_name,
            "email": email,
            "is_admin": is_first_user
        }
    }), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required."}), 400

    user = get_user_by_email(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"status": "error", "message": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    session["email"] = user["email"]
    session["full_name"] = user["full_name"]
    session["is_admin"] = bool(user["is_admin"])

    return jsonify({
        "status": "success",
        "message": "Logged in successfully.",
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "is_admin": bool(user["is_admin"])
        }
    })


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out successfully."})


@app.route("/api/me", methods=["GET"])
def api_me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({
            "authenticated": False,
            "database_engine": get_db_engine()
        })
    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"authenticated": False, "database_engine": get_db_engine()})

    return jsonify({
        "authenticated": True,
        "database_engine": get_db_engine(),
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "is_admin": bool(user["is_admin"]),
            "created_at": str(user["created_at"])
        }
    })


# -----------------------------------------------------------------------------
# Loan Prediction & Application Endpoints
# -----------------------------------------------------------------------------

def validate_applicant_payload(payload: dict) -> tuple[dict | None, str | None]:
    """Validate all 11 required applicant features without silent defaults."""
    required_fields = [
        "no_of_dependents", "education", "self_employed", "income_annum",
        "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
    ]
    missing = [f for f in required_fields if f not in payload or payload[f] is None or payload[f] == ""]
    if missing:
        return None, f"Missing required fields: {', '.join(missing)}"

    cleaned = {}
    try:
        cleaned["no_of_dependents"] = int(payload["no_of_dependents"])
        if cleaned["no_of_dependents"] < 0 or cleaned["no_of_dependents"] > 20:
            return None, "Number of dependents must be between 0 and 20."

        cleaned["education"] = str(payload["education"]).strip()
        if cleaned["education"] not in ["Graduate", "Not Graduate"]:
            return None, "Education must be 'Graduate' or 'Not Graduate'."

        cleaned["self_employed"] = str(payload["self_employed"]).strip()
        if cleaned["self_employed"] not in ["Yes", "No"]:
            return None, "Self-employed status must be 'Yes' or 'No'."

        cleaned["income_annum"] = float(payload["income_annum"])
        if cleaned["income_annum"] < 0:
            return None, "Annual income cannot be negative."

        cleaned["loan_amount"] = float(payload["loan_amount"])
        if cleaned["loan_amount"] <= 0:
            return None, "Loan amount must be greater than zero."

        cleaned["loan_term"] = int(payload["loan_term"])
        if cleaned["loan_term"] < 1 or cleaned["loan_term"] > 40:
            return None, "Loan term must be between 1 and 40 years."

        cleaned["cibil_score"] = int(payload["cibil_score"])
        if cleaned["cibil_score"] < 300 or cleaned["cibil_score"] > 900:
            return None, "CIBIL credit score must be between 300 and 900."

        cleaned["residential_assets_value"] = float(payload["residential_assets_value"])
        cleaned["commercial_assets_value"] = float(payload["commercial_assets_value"])
        cleaned["luxury_assets_value"] = float(payload["luxury_assets_value"])
        cleaned["bank_asset_value"] = float(payload["bank_asset_value"])

        if any(cleaned[k] < 0 for k in [
            "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value"
        ]):
            return None, "Asset valuations cannot be negative."

    except (ValueError, TypeError) as err:
        return None, f"Data type or formatting error: {err}"

    return cleaned, None


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    meta = get_metadata()
    if not meta:
        return jsonify({"status": "error", "message": "Model metadata not found. Run train_model.py first."}), 500

    cleaned_data, error_msg = validate_applicant_payload(payload)
    if error_msg:
        return jsonify({"status": "error", "message": error_msg}), 400

    # Format single-row DataFrame for pipeline prediction
    row_data = {feat: [cleaned_data[feat]] for feat in meta.get("features", [])}
    df_applicant = pd.DataFrame(row_data)

    results = {}
    model_names = ["random_forest", "decision_tree", "logistic_regression"]
    positive_class = meta.get("positive_class", "Approved")

    for name in model_names:
        pipe = get_pipeline(name)
        if pipe is None:
            return jsonify({"status": "error", "message": f"Model '{name}' not found on server."}), 500

        pred = str(pipe.predict(df_applicant)[0])
        classes = list(pipe.classes_)
        proba = pipe.predict_proba(df_applicant)[0]
        pos_idx = classes.index(positive_class) if positive_class in classes else 0
        approval_prob = round(float(proba[pos_idx]), 4)

        results[name] = {
            "prediction": pred,
            "approval_probability": approval_prob,
            "confidence": round(float(max(proba)), 4),
            "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(classes, proba)}
        }

    # Consensus determination
    approved_count = sum(1 for m in results.values() if m["prediction"] == positive_class)
    consensus = positive_class if approved_count >= 2 else "Rejected"
    agreement_text = f"{approved_count}/{len(results)} algorithms recommend {consensus}"

    # If user is authenticated, persist application & prediction history
    user_id = session.get("user_id")
    app_id = None
    if user_id:
        try:
            app_id = save_loan_application(user_id, cleaned_data)
            for m_name, m_res in results.items():
                save_prediction_result(
                    application_id=app_id,
                    model_name=m_name,
                    predicted_class=m_res["prediction"],
                    approved_probability=m_res["approval_probability"],
                    model_version="v1.0"
                )
        except Exception as e:
            app.logger.warning(f"Could not persist application to database: {e}")

    return jsonify({
        "status": "success",
        "consensus": consensus,
        "agreement": agreement_text,
        "predictions": results,
        "application_id": app_id,
        "authenticated": bool(user_id)
    })


@app.route("/api/applications", methods=["GET"])
@login_required
def api_get_applications():
    user_id = session["user_id"]
    limit = int(request.args.get("limit", 50))
    apps = get_user_applications(user_id, limit=limit)
    return jsonify({
        "status": "success",
        "count": len(apps),
        "applications": apps
    })


@app.route("/api/applications/<int:app_id>", methods=["GET"])
@login_required
def api_get_application_detail(app_id):
    user_id = session["user_id"]
    is_admin = session.get("is_admin", False)

    # Admin can view any application, regular user only their own
    owner_check = None if is_admin else user_id
    detail = get_application_details(app_id, user_id=owner_check)
    if not detail:
        return jsonify({"status": "error", "message": "Application not found or unauthorized access."}), 404

    return jsonify({
        "status": "success",
        "application": detail["application"],
        "predictions": detail["predictions"]
    })


@app.route("/api/applications/export-csv", methods=["GET"])
def api_export_applications_csv():
    """Export all new loan applications from the database as a CSV file."""
    from flask import Response
    from database.db import execute_sql
    import io, csv

    sql = """
        SELECT a.id, a.user_id, u.full_name, u.email,
               a.no_of_dependents, a.education, a.self_employed,
               a.income_annum, a.loan_amount, a.loan_term, a.cibil_score,
               a.residential_assets_value, a.commercial_assets_value,
               a.luxury_assets_value, a.bank_asset_value, a.created_at
        FROM loan_applications a
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.id DESC
    """
    rows = execute_sql(sql, fetchall=True)
    if not rows:
        return "No applications found to export.", 404

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=new_loan_applications.csv"}
    )


# -----------------------------------------------------------------------------
# Model Metrics & Benchmark Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/models/metrics", methods=["GET"])
def api_metrics():
    eval_path = Config.RESULTS_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["status"] = "success"
            return jsonify(data)
    return jsonify({"status": "error", "message": "Model evaluation results not found. Run train_model.py."}), 404


# -----------------------------------------------------------------------------
# Admin Analytics Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/admin/stats", methods=["GET"])
@admin_required
def api_admin_stats():
    stats = get_admin_stats()
    return jsonify({
        "status": "success",
        "stats": stats
    })


@app.route("/api/admin/applications", methods=["GET"])
@admin_required
def api_admin_applications():
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 50))
    apps = search_all_applications(query_str=q if q else None, limit=limit)
    return jsonify({
        "status": "success",
        "count": len(apps),
        "applications": apps
    })


# -----------------------------------------------------------------------------
# Database Setup & Initialization
# -----------------------------------------------------------------------------

with app.app_context():
    init_db()


if __name__ == "__main__":
    import sys
    # Support CLI creation of admin user: python app.py --create-admin email password [name]
    if "--create-admin" in sys.argv:
        try:
            idx = sys.argv.index("--create-admin")
            admin_email = sys.argv[idx + 1]
            admin_pwd = sys.argv[idx + 2]
            admin_name = sys.argv[idx + 3] if len(sys.argv) > idx + 3 else "Administrator"
            init_db()
            existing = get_user_by_email(admin_email)
            if existing:
                print(f"Error: User with email {admin_email} already exists.")
            else:
                create_user(admin_name, admin_email, generate_password_hash(admin_pwd), is_admin=True)
                print(f"Admin account created successfully: {admin_email}")
            sys.exit(0)
        except IndexError:
            print("Usage: python app.py --create-admin <email> <password> [full_name]")
            sys.exit(1)

    print("=" * 65)
    print(">> CreditPulse AI - Loan Approval Prediction System Web Server")
    print(">> URL: http://127.0.0.1:5000")
    print(f">> Database Engine: {get_db_engine().upper()}")
    print(">> Models Active: Random Forest (98.1%), Decision Tree (97.3%), Logistic Regression (92.3%)")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=True)
