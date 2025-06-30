from .main import app
from .models import DetectionResult
from .database import Base, engine, SessionLocal
from .detection import detect_dogs, load_models, process_video, check_for_muzzle
from .report_generator import generate_pdf_report

__all__ = [
    'app',
    'DetectionResult',
    'Base',
    'engine',
    'SessionLocal',
    'detect_dogs',
    'load_models',
    'process_video',
    'check_for_muzzle',
    'generate_pdf_report',
    'generate_full_pdf_report'
]
