"""Application configuration settings for the Loan Approval Prediction System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "loan-approval-super-secret-key-2026-secure")
    
    # Database configuration
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "loan_approval_db")
    
    # Path mappings
    ROOT_DIR = BASE_DIR
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    DATA_DIR = BASE_DIR / "data"
    STATIC_DIR = BASE_DIR / "static"
    
    # Session cookie settings
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
