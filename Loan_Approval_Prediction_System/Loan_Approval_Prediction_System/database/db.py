"""Database management module supporting MySQL with graceful SQLite fallback."""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pymysql
from pymysql.cursors import DictCursor
from config import Config

DB_FILE = Config.ROOT_DIR / "database" / "loan_approval.db"
_DB_ENGINE = None  # 'mysql' or 'sqlite'


def get_connection():
    """Attempt to establish a MySQL connection. If unsuccessful, fallback to SQLite."""
    global _DB_ENGINE
    # 1. Try MySQL if configured
    try:
        conn = pymysql.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            cursorclass=DictCursor,
            connect_timeout=2,
            autocommit=True
        )
        _DB_ENGINE = "mysql"
        return conn, "mysql"
    except Exception:
        # 2. Transparent fallback to SQLite
        _DB_ENGINE = "sqlite"
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        return conn, "sqlite"


def get_db_engine():
    """Return the currently active database engine ('mysql' or 'sqlite')."""
    global _DB_ENGINE
    if _DB_ENGINE is None:
        conn, engine = get_connection()
        conn.close()
        return engine
    return _DB_ENGINE


def init_db():
    """Initialize database tables for either MySQL or SQLite."""
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        full_name VARCHAR(150) NOT NULL,
                        email VARCHAR(191) NOT NULL UNIQUE,
                        password_hash VARCHAR(255) NOT NULL,
                        is_admin TINYINT(1) NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_user_email (email)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS loan_applications (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        no_of_dependents INT NOT NULL,
                        education VARCHAR(50) NOT NULL,
                        self_employed VARCHAR(50) NOT NULL,
                        income_annum DECIMAL(15, 2) NOT NULL,
                        loan_amount DECIMAL(15, 2) NOT NULL,
                        loan_term INT NOT NULL,
                        cibil_score INT NOT NULL,
                        residential_assets_value DECIMAL(15, 2) NOT NULL,
                        commercial_assets_value DECIMAL(15, 2) NOT NULL,
                        luxury_assets_value DECIMAL(15, 2) NOT NULL,
                        bank_asset_value DECIMAL(15, 2) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_app_user (user_id),
                        INDEX idx_app_created (created_at),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        application_id INT NOT NULL,
                        model_name VARCHAR(100) NOT NULL,
                        predicted_class VARCHAR(50) NOT NULL,
                        approved_probability DECIMAL(6, 4) NULL,
                        model_version VARCHAR(50) DEFAULT 'v1.0',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_pred_app (application_id),
                        INDEX idx_pred_model (model_name),
                        FOREIGN KEY (application_id) REFERENCES loan_applications(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)
        else:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS loan_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    no_of_dependents INTEGER NOT NULL,
                    education TEXT NOT NULL,
                    self_employed TEXT NOT NULL,
                    income_annum REAL NOT NULL,
                    loan_amount REAL NOT NULL,
                    loan_term INTEGER NOT NULL,
                    cibil_score INTEGER NOT NULL,
                    residential_assets_value REAL NOT NULL,
                    commercial_assets_value REAL NOT NULL,
                    luxury_assets_value REAL NOT NULL,
                    bank_asset_value REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    application_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    approved_probability REAL NULL,
                    model_version TEXT DEFAULT 'v1.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (application_id) REFERENCES loan_applications(id) ON DELETE CASCADE
                );
            """)
            conn.commit()
    finally:
        conn.close()


def execute_sql(sql, params=None, fetchone=False, fetchall=False, commit=False):
    """Unified query executor adapting placeholders (%s vs ?) to MySQL or SQLite."""
    conn, engine = get_connection()
    params = params or ()
    
    # Adjust placeholders if SQLite
    query = sql
    if engine == "sqlite":
        query = sql.replace("%s", "?")
    
    try:
        if engine == "mysql":
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if commit:
                    conn.commit()
                    return cursor.lastrowid
                if fetchone:
                    return cursor.fetchone()
                if fetchall:
                    return cursor.fetchall()
                return cursor.rowcount
        else:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
                last_id = cursor.lastrowid
                return last_id
            if fetchone:
                row = cursor.fetchone()
                return dict(row) if row else None
            if fetchall:
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
            return cursor.rowcount
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# User Account Operations
# -----------------------------------------------------------------------------

def create_user(full_name, email, password_hash, is_admin=False):
    sql = """
        INSERT INTO users (full_name, email, password_hash, is_admin, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return execute_sql(sql, (full_name.strip(), email.strip().lower(), password_hash, 1 if is_admin else 0, now), commit=True)


def get_user_by_email(email):
    sql = "SELECT * FROM users WHERE email = %s LIMIT 1"
    return execute_sql(sql, (email.strip().lower(),), fetchone=True)


def get_user_by_id(user_id):
    sql = "SELECT id, full_name, email, is_admin, created_at FROM users WHERE id = %s LIMIT 1"
    return execute_sql(sql, (user_id,), fetchone=True)


def count_users():
    sql = "SELECT COUNT(*) as count FROM users"
    res = execute_sql(sql, fetchone=True)
    return res["count"] if res else 0


# -----------------------------------------------------------------------------
# Loan Applications & Prediction Operations
# -----------------------------------------------------------------------------

def save_loan_application(user_id, app_data):
    sql = """
        INSERT INTO loan_applications (
            user_id, no_of_dependents, education, self_employed,
            income_annum, loan_amount, loan_term, cibil_score,
            residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    params = (
        user_id,
        int(app_data["no_of_dependents"]),
        str(app_data["education"]).strip(),
        str(app_data["self_employed"]).strip(),
        float(app_data["income_annum"]),
        float(app_data["loan_amount"]),
        int(app_data["loan_term"]),
        int(app_data["cibil_score"]),
        float(app_data["residential_assets_value"]),
        float(app_data["commercial_assets_value"]),
        float(app_data["luxury_assets_value"]),
        float(app_data["bank_asset_value"]),
        now
    )
    return execute_sql(sql, params, commit=True)


def save_prediction_result(application_id, model_name, predicted_class, approved_probability=None, model_version="v1.0"):
    sql = """
        INSERT INTO prediction_history (
            application_id, model_name, predicted_class, approved_probability, model_version, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    prob_val = float(approved_probability) if approved_probability is not None else None
    return execute_sql(sql, (application_id, model_name, predicted_class, prob_val, model_version, now), commit=True)


def get_user_applications(user_id, limit=50):
    """Get applications for a specific user, including individual model outputs."""
    sql = """
        SELECT a.*,
               GROUP_CONCAT(CONCAT(p.model_name, ':', p.predicted_class, ':', IFNULL(p.approved_probability, '0')) SEPARATOR ';') as predictions_summary
        FROM loan_applications a
        LEFT JOIN prediction_history p ON a.id = p.application_id
        WHERE a.user_id = %s
        GROUP BY a.id
        ORDER BY a.created_at DESC
        LIMIT %s
    """
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cursor:
                cursor.execute(sql, (user_id, limit))
                return cursor.fetchall()
        else:
            # SQLite GROUP_CONCAT syntax
            sql_sqlite = """
                SELECT a.*,
                       GROUP_CONCAT(p.model_name || ':' || p.predicted_class || ':' || IFNULL(p.approved_probability, '0'), ';') as predictions_summary
                FROM loan_applications a
                LEFT JOIN prediction_history p ON a.id = p.application_id
                WHERE a.user_id = ?
                GROUP BY a.id
                ORDER BY a.created_at DESC
                LIMIT ?
            """
            cursor = conn.cursor()
            cursor.execute(sql_sqlite, (user_id, limit))
            return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_application_details(application_id, user_id=None):
    """Retrieve an application with its predictions. Enforce user ownership if user_id is provided."""
    app_sql = "SELECT * FROM loan_applications WHERE id = %s"
    app_params = (application_id,)
    if user_id is not None:
        app_sql += " AND user_id = %s"
        app_params = (application_id, user_id)
        
    app = execute_sql(app_sql, app_params, fetchone=True)
    if not app:
        return None
        
    pred_sql = "SELECT * FROM prediction_history WHERE application_id = %s ORDER BY id ASC"
    predictions = execute_sql(pred_sql, (application_id,), fetchall=True)
    
    return {
        "application": app,
        "predictions": predictions or []
    }


# -----------------------------------------------------------------------------
# Admin Analytics Operations
# -----------------------------------------------------------------------------

def get_admin_stats():
    """Retrieve statistical aggregations across users and model predictions."""
    total_users = count_users()
    
    app_count_res = execute_sql("SELECT COUNT(*) as count FROM loan_applications", fetchone=True)
    total_apps = app_count_res["count"] if app_count_res else 0
    
    pred_counts = execute_sql("""
        SELECT model_name, predicted_class, COUNT(*) as count
        FROM prediction_history
        GROUP BY model_name, predicted_class
    """, fetchall=True)
    
    recent_apps = execute_sql("""
        SELECT a.id, a.user_id, u.full_name, u.email, a.income_annum, a.loan_amount, a.cibil_score, a.created_at
        FROM loan_applications a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.created_at DESC
        LIMIT 10
    """, fetchall=True)
    
    return {
        "total_users": total_users,
        "total_applications": total_apps,
        "prediction_counts": pred_counts or [],
        "recent_applications": recent_apps or [],
        "database_engine": get_db_engine()
    }


def search_all_applications(query_str=None, limit=50):
    """Search applications across all users for admin review."""
    sql = """
        SELECT a.id, a.user_id, u.full_name, u.email, a.income_annum, a.loan_amount, a.loan_term, a.cibil_score, a.created_at
        FROM loan_applications a
        JOIN users u ON a.user_id = u.id
    """
    params = []
    if query_str:
        sql += " WHERE u.full_name LIKE %s OR u.email LIKE %s OR a.id = %s"
        wildcard = f"%{query_str}%"
        try:
            int_id = int(query_str)
        except ValueError:
            int_id = -1
        params = [wildcard, wildcard, int_id]
        
    sql += " ORDER BY a.created_at DESC LIMIT %s"
    params.append(limit)
    return execute_sql(sql, tuple(params), fetchall=True)
