# app/main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import os
import uuid
import logging
import json

# Импортируем настройки из config.py через относительный импорт
from .config import settings
from .models import DetectionResult
from .database import SessionLocal, engine, Base
from .detection import detect_dogs, load_models, process_video
from .report_generator import generate_pdf_report, generate_full_pdf_report

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title=settings.APP_NAME,
    description="Система для обнаружения собак без намордников в общественных местах",
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Создание необходимых директорий, если они не существуют
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(settings.REPORT_FOLDER, exist_ok=True)

# Настройка статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_db():
    """
    Генератор сессий для работы с базой данных.
    Обеспечивает корректное закрытие сессии после использования.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    """
    Инициализация при запуске приложения:
    - Создание таблиц в базе данных
    - Загрузка ML моделей
    - Проверка директорий
    """
    Base.metadata.create_all(bind=engine)
    load_models()  # Загружаем модели при старте приложения
    logger.info("Application started. Database tables created and models loaded.")

@app.get("/", include_in_schema=False)
async def home(request: Request):
    """
    Главная страница с веб-интерфейсом.
    Не включается в OpenAPI схему.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload/", response_model=dict)
async def upload_image(
    file: UploadFile = File(..., description="Изображение для анализа (JPG/PNG, до 10MB)"),
    db: Session = Depends(get_db)
):
    """Загрузка и обработка изображения с собаками"""
    try:
        # Валидация файла
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS.split(','):
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимый формат файла. Разрешены: {settings.ALLOWED_EXTENSIONS}"
            )

        # Проверка размера файла
        file.file.seek(0, 2)
        file_size = file.file.tell()
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой. Максимальный размер: {settings.MAX_FILE_SIZE_MB}MB"
            )
        file.file.seek(0)

        # Генерация уникального имени файла
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_location = os.path.join(settings.UPLOAD_FOLDER, unique_filename)

        # Сохранение файла
        try:
            os.makedirs(os.path.dirname(file_location), exist_ok=True)
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())
            logger.info(f"Файл сохранен: {file_location}")
        except Exception as e:
            logger.error(f"Ошибка сохранения файла: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Ошибка при сохранении файла на сервер"
            )

        # Обработка изображения
        try:
            detection_results = detect_dogs(file_location)
            logger.info(f"Детекция завершена для файла {unique_filename}")
        except Exception as e:
            logger.error(f"Ошибка детекции: {str(e)}")
            if os.path.exists(file_location):
                os.remove(file_location)
            raise HTTPException(
                status_code=500,
                detail="Ошибка при обработке изображения"
            )

        # Сохранение результатов в БД
        db_result = DetectionResult(
            filename=unique_filename,
            detection_data=json.dumps(detection_results, ensure_ascii=False),
            has_no_muzzle=len(detection_results["dogs_without_muzzle"]) > 0,
            is_video=False
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        logger.info(f"Результаты сохранены в БД, ID: {db_result.id}")

        # Генерация PDF отчета
        report_filename = None
        try:
            report_filename = f"report_{db_result.id}.pdf"
            report_path = os.path.join(settings.REPORT_FOLDER, report_filename)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            generate_pdf_report(
                file_location,
                detection_results,
                report_path,
                is_video=False
            )
            logger.info(f"PDF отчет сгенерирован: {report_path}")
            
            # Обновляем запись с именем отчета
            db_result.report_filename = report_filename
            db.commit()
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {str(e)}")
            if report_filename and os.path.exists(report_path):
                os.remove(report_path)

        return {
            "id": db_result.id,
            "filename": unique_filename,
            "results": detection_results,
            "is_video": False,
            "report_url": f"/api/reports/{report_filename}" if report_filename else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера"
        )

@app.get("/api/history/", response_model=dict)
async def get_detection_history(
    skip: int = 0,
    limit: int = 10,
    has_violations: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    try:
        query = db.query(DetectionResult)
        
        if has_violations is not None:
            query = query.filter(DetectionResult.has_no_muzzle == has_violations)
        
        total = query.count()
        history = query.order_by(DetectionResult.id.desc()).offset(skip).limit(limit).all()
        
        return {
            "items": [
                {
                    "id": item.id,
                    "filename": item.filename,
                    "created_at": item.created_at,
                    "has_no_muzzle": item.has_no_muzzle,
                    "is_video": item.is_video,  # Убедитесь, что это поле есть
                    "report_url": f"/api/reports/{item.report_filename}" if item.report_filename else None
                }
                for item in history
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Ошибка получения истории: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Ошибка при получении истории проверок"
        )

@app.get("/api/history/{detection_id}", response_model=dict)
async def get_detection_details(
    detection_id: int,
    db: Session = Depends(get_db)
):
    """
    Получение детальной информации о конкретной проверке.
    
    Параметры:
    - detection_id: ID записи в БД
    
    Возвращает:
    - Полную информацию о проверке включая данные детекции
    """
    try:
        item = db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()
        if not item:
            raise HTTPException(
                status_code=404,
                detail="Запись не найдена"
            )
        
        return {
            "id": item.id,
            "filename": item.filename,
            "detection_data": json.loads(item.detection_data),  # Безопасное преобразование
            "created_at": item.created_at,
            "has_no_muzzle": item.has_no_muzzle,
            "image_url": f"/api/images/{item.filename}",
            "report_url": f"/api/reports/{item.report_filename}" if item.report_filename else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения деталей проверки: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Ошибка при получении деталей проверки"
        )
        
@app.post("/api/upload_video/", response_model=dict)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    logger.info(f"Начало обработки видеофайла: {file.filename}")
    
    try:
        # Проверка формата и размера
        file_ext = Path(file.filename).suffix.lower()
        allowed_video_ext = [".mp4", ".mov", ".avi"]
        if file_ext not in allowed_video_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимый видеоформат. Разрешены: {', '.join(allowed_video_ext)}"
            )

        file.file.seek(0, 2)
        file_size = file.file.tell()
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой. Максимальный размер: {settings.MAX_FILE_SIZE_MB}MB"
            )
        file.file.seek(0)

        # Сохранение видео
        video_filename = f"{uuid.uuid4()}{file_ext}"
        video_path = os.path.join(settings.UPLOAD_FOLDER, video_filename)
        
        try:
            with open(video_path, "wb+") as file_object:
                file_object.write(file.file.read())
        except Exception as e:
            logger.error(f"Ошибка сохранения видео: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Ошибка при сохранении видео"
            )

        # Обработка видео
        try:
            video_results = process_video(video_path, settings.VIDEO_FRAME_RATE)
            
            # Проверка нарушений
            has_violations = video_results["violations_count"] > 0
            
            # Генерация PDF отчета
            report_filename = None
            if has_violations:
                try:
                    report_filename = f"report_{uuid.uuid4()}.pdf"
                    report_path = os.path.join(settings.REPORT_FOLDER, report_filename)
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    
                    generate_pdf_report(video_path, video_results, report_path, is_video=True)
                    logger.info(f"PDF отчет сгенерирован: {report_path}")
                except Exception as e:
                    logger.error(f"Ошибка генерации PDF отчета: {str(e)}")
                    report_filename = None

            # Сохранение в БД
            db_result = DetectionResult(
                filename=video_filename,
                detection_data=json.dumps(video_results, ensure_ascii=False),
                is_video=True,
                has_no_muzzle=has_violations,
                report_filename=report_filename
            )
            
            db.add(db_result)
            db.commit()
            
            return {
                "id": db_result.id,
                "filename": video_filename,
                "results": video_results,
                "is_video": True,
                "report_url": f"/api/reports/{report_filename}" if report_filename else None,
                "has_violations": has_violations
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки видео: {str(e)}", exc_info=True)
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception as cleanup_error:
                logger.error(f"Ошибка при удалении временного файла: {str(cleanup_error)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера"
        )
        
@app.delete("/api/history/{detection_id}")
async def delete_detection(
    detection_id: int,
    db: Session = Depends(get_db)
):
    """
    Удаление записи о проверке и связанных файлов.
    
    Параметры:
    - detection_id: ID записи в БД
    
    Возвращает:
    - Сообщение об успешном удалении
    """
    try:
        item = db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()
        if not item:
            raise HTTPException(
                status_code=404,
                detail="Запись не найдена"
            )
        
        # Удаляем связанные файлы
        try:
            if item.filename and os.path.exists(os.path.join(settings.UPLOAD_FOLDER, item.filename)):
                os.remove(os.path.join(settings.UPLOAD_FOLDER, item.filename))
            if item.report_filename and os.path.exists(os.path.join(settings.REPORT_FOLDER, item.report_filename)):
                os.remove(os.path.join(settings.REPORT_FOLDER, item.report_filename))
        except Exception as e:
            logger.error(f"Ошибка удаления файлов: {str(e)}")

        db.delete(item)
        db.commit()
        logger.info(f"Запись удалена, ID: {detection_id}")
        
        return {"message": "Запись успешно удалена"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления записи: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Ошибка при удалении записи"
        )

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """
    Получение изображения по имени файла.
    
    Параметры:
    - filename: имя файла изображения
    
    Возвращает:
    - Файл изображения
    """
    image_path = os.path.join(settings.UPLOAD_FOLDER, filename)
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Изображение не найдено"
        )
    return FileResponse(image_path)

@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    """
    Получение PDF отчета по имени файла.
    
    Параметры:
    - filename: имя файла отчета
    
    Возвращает:
    - PDF файл отчета
    """
    report_path = os.path.join(settings.REPORT_FOLDER, filename)
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404,
            detail="Отчет не найден"
        )
    return FileResponse(
        report_path,
        media_type='application/pdf',
        filename=filename
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений для единообразного формата ошибок"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )
    
@app.get("/api/generate_full_report/")
async def generate_full_report(db: Session = Depends(get_db)):
    """Генерация PDF отчета по всей базе данных"""
    try:
        # Получаем все записи из БД
        all_items = db.query(DetectionResult).order_by(DetectionResult.created_at.desc()).all()
        
        if not all_items:
            raise HTTPException(
                status_code=404,
                detail="Нет данных для генерации отчета"
            )

        # Генерируем уникальное имя файла
        report_filename = f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(settings.REPORT_FOLDER, report_filename)
        
        # Создаем PDF отчет
        if not generate_full_pdf_report(all_items, report_path):
            raise HTTPException(
                status_code=500,
                detail="Не удалось сгенерировать отчет"
            )
        
        return {
            "report_url": f"/api/reports/{report_filename}",
            "message": "Полный отчет успешно сгенерирован"
        }
        
    except HTTPException as he:
        logger.error(f"HTTP ошибка при генерации отчета: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Ошибка генерации полного отчета: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при генерации полного отчета: {str(e)}"
        )    

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Обработчик всех неожиданных исключений"""
    logger.error(f"Необработанное исключение: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Внутренняя ошибка сервера",
            "status_code": 500
        }
    )