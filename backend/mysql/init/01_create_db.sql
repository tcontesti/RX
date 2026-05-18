CREATE DATABASE IF NOT EXISTS cxr_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cxr_detection;

CREATE TABLE cxr_studies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    study_uid VARCHAR(100) UNIQUE NOT NULL,
    patient_id VARCHAR(50),
    patient_name VARCHAR(200),
    status ENUM('queued', 'processing', 'completed', 'error') DEFAULT 'queued',
    image_format VARCHAR(10) DEFAULT 'png',
    num_detections INT DEFAULT 0,
    inference_time_ms FLOAT,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    INDEX idx_status (status),
    INDEX idx_patient (patient_id),
    INDEX idx_created (created_at DESC)
) ENGINE=InnoDB;

CREATE TABLE cxr_detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    study_id INT NOT NULL,
    x1 FLOAT NOT NULL,
    y1 FLOAT NOT NULL,
    x2 FLOAT NOT NULL,
    y2 FLOAT NOT NULL,
    score FLOAT NOT NULL,
    label VARCHAR(50) DEFAULT 'nodule',
    model_source VARCHAR(30),
    FOREIGN KEY (study_id) REFERENCES cxr_studies(id) ON DELETE CASCADE,
    INDEX idx_study (study_id),
    INDEX idx_score (score DESC)
) ENGINE=InnoDB;

CREATE TABLE cxr_annotated_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    study_id INT NOT NULL UNIQUE,
    image_data LONGBLOB NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (study_id) REFERENCES cxr_studies(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE cxr_original_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    study_id INT NOT NULL UNIQUE,
    image_data LONGBLOB NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (study_id) REFERENCES cxr_studies(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE cxr_validations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    study_id INT NOT NULL UNIQUE,
    validated_by VARCHAR(100),
    validation_result ENUM('correct', 'incorrect', 'partial') NOT NULL,
    notes TEXT,
    validated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (study_id) REFERENCES cxr_studies(id) ON DELETE CASCADE,
    INDEX idx_result (validation_result),
    INDEX idx_validator (validated_by)
) ENGINE=InnoDB;

CREATE TABLE cxr_manual_annotations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    validation_id INT NOT NULL,
    x1 FLOAT NOT NULL,
    y1 FLOAT NOT NULL,
    x2 FLOAT NOT NULL,
    y2 FLOAT NOT NULL,
    label VARCHAR(50) DEFAULT 'nodule',
    annotation_type ENUM('missed', 'false_positive', 'corrected') DEFAULT 'missed',
    notes TEXT,
    FOREIGN KEY (validation_id) REFERENCES cxr_validations(id) ON DELETE CASCADE,
    INDEX idx_validation (validation_id)
) ENGINE=InnoDB;
