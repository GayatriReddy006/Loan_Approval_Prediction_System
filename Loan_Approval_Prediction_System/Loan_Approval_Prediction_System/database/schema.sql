-- ============================================================================
-- Loan Approval Prediction System - Database Schema (MySQL)
-- ============================================================================

CREATE DATABASE IF NOT EXISTS loan_approval_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE loan_approval_db;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(191) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_admin TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Loan Applications Table
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

-- 3. Prediction History Table
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
