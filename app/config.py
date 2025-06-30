# app/config.py
from pydantic import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Настройки приложения
    APP_NAME: str = "Dog Muzzle Detection System"
    APP_VERSION: str = "1.0.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_RELOAD: bool = True
    
    # Настройки базы данных
    DATABASE_URL: str = "sqlite:///./detections.db"
    
    # Настройки файлов
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: str = ".jpg,.jpeg,.png,.mp4,.mov,.avi"
    UPLOAD_FOLDER: str = "./static/uploads"
    REPORT_FOLDER: str = "./static/reports"
    VIDEO_FRAME_RATE: int = 10
    
    # Настройки моделей ML
    DETECTION_MODEL_PATH: str = "./models/yolov8n.pt"
    MUZZLE_CLASSIFIER_PATH: str = "./models/yolov8n_muzzle_cls.pt"
    CONFIDENCE_THRESHOLD: float = 0.6
    MUZZLE_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Безопасность
    SECRET_KEY: str = "your-secret-key-here"
    CORS_ORIGINS: str = "*"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

settings = Settings()