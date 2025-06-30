from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from .database import Base

class DetectionResult(Base):
    """Модель для хранения результатов детекции в базе данных"""
    __tablename__ = "detection_results"
    
    # Уникальный идентификатор записи
    id = Column(Integer, primary_key=True, index=True)
    
    # Имя исходного файла изображения
    filename = Column(String, nullable=False)
    
    # Имя сгенерированного PDF отчета
    report_filename = Column(String, nullable=True)
    
    # Данные детекции в формате JSON (строкой)
    detection_data = Column(String, nullable=False)
    
    # Флаг наличия собак без намордников
    has_no_muzzle = Column(Boolean, nullable=False)
    
    # Дата и время создания записи (автоматически)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Флаг видеофайла
    is_video: bool = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<DetectionResult(id={self.id}, filename='{self.filename}', has_no_muzzle={self.has_no_muzzle})>"