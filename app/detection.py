import logging
import tempfile
from ultralytics import YOLO
import cv2
import os
import numpy as np
from .config import settings

logger = logging.getLogger(__name__)

# Глобальные переменные для моделей
detection_model = None
muzzle_classifier = None

def load_models():
    """Загрузка моделей при старте приложения"""
    global detection_model, muzzle_classifier
    try:
        # Основная модель для детекции собак
        detection_model = YOLO(settings.DETECTION_MODEL_PATH)
        detection_model.fuse()  # Оптимизация модели
        
        # Модель для классификации намордников
        muzzle_classifier = YOLO(settings.MUZZLE_CLASSIFIER_PATH)
        muzzle_classifier.fuse()
        
        # Проверка доступности CUDA
        device = 'cuda' if muzzle_classifier.device.type != 'cpu' else 'cpu'
        print(f"Модели загружены. Устройство: {device}")
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки моделей: {e}") from e

def detect_dogs(image_path):
    """Основная функция детекции собак и проверки намордников"""
    global detection_model, muzzle_classifier
    
    if None in (detection_model, muzzle_classifier):
        load_models()
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    detections = detection_model(image, verbose=False)
    
    detection_data = {
        "dogs": [],
        "dogs_without_muzzle": [],
        "dogs_with_muzzle": [],
        "error": None
    }

    try:
        for result in detections:
            for box in result.boxes:
                if box.conf.item() < settings.CONFIDENCE_THRESHOLD:
                    continue
                    
                if detection_model.names[int(box.cls)] == 'dog':
                    bbox = [round(x) for x in box.xyxy[0].tolist()]
                    detection_data["dogs"].append(bbox)
                    
                    status = check_for_muzzle(image, bbox)
                    if status == "without":
                        detection_data["dogs_without_muzzle"].append(bbox)
                    elif status == "with":
                        detection_data["dogs_with_muzzle"].append(bbox)
    except Exception as e:
        detection_data["error"] = str(e)
    
    return detection_data

def check_for_muzzle(image, bbox):
    """Улучшенная проверка наличия намордника с проверкой confidence"""
    x1, y1, x2, y2 = bbox
    
    padding = 15
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    dog_roi = image[y1:y2, x1:x2]
    if dog_roi.size == 0:
        return "unknown"
    
    try:
        resized = cv2.resize(dog_roi, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        results = muzzle_classifier(rgb, verbose=False)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            if box.conf.item() < settings.MUZZLE_CONFIDENCE_THRESHOLD:
                return "unknown"
            return "without" if int(box.cls) == 0 else "with"
    except Exception as e:
        logger.error(f"Ошибка классификации намордника: {str(e)}")
    
    return "unknown"
    
def process_video(video_path, frame_rate=10):
    """Обработка видеофайла с извлечением и анализом кадров"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видеофайл: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps / frame_rate))

    results = {
        "frame_results": [],
        "total_frames": total_frames,
        "duration_sec": total_frames / fps,
        "analyzed_frames": 0,
        "violations_count": 0
    }

    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                try:
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    nparr = np.frombuffer(img_encoded, np.uint8)
                    frame_copy = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    frame_result = detect_dogs_from_frame(frame_copy)
                    
                    frame_result["dogs"] = [
                        bbox for bbox in frame_result["dogs"] 
                        if bbox["confidence"] >= settings.CONFIDENCE_THRESHOLD
                    ]
                    
                    frame_result["dogs_with_muzzle"] = [
                        bbox for bbox in frame_result["dogs_with_muzzle"]
                        if bbox["confidence"] >= settings.MUZZLE_CONFIDENCE_THRESHOLD
                    ]
                    
                    frame_result["dogs_without_muzzle"] = [
                        bbox for bbox in frame_result["dogs_without_muzzle"]
                        if bbox["confidence"] >= settings.MUZZLE_CONFIDENCE_THRESHOLD
                    ]
                    
                    if frame_result["dogs_without_muzzle"]:
                        if results["frame_results"] and \
                           (frame_count - results["frame_results"][-1]["frame_index"]) <= 2:
                            results["violations_count"] += 1
                    
                    results["frame_results"].append({
                        "frame_index": frame_count,
                        "frame_time": frame_count / fps,
                        "results": frame_result,
                        "is_confirmed_violation": bool(frame_result["dogs_without_muzzle"])
                    })
                    
                    results["analyzed_frames"] += 1
                    
                except Exception as e:
                    logger.error(f"Ошибка обработки кадра {frame_count}: {str(e)}")
                    continue

            frame_count += 1
            
            if frame_count > 3600 * fps:
                logger.warning("Видео слишком длинное, обработано только первые 3600 секунд")
                break

    finally:
        cap.release()
    
    return results

def detect_dogs_from_frame(frame):
    """Обнаружение собак и намордников в одном кадре видео"""
    global detection_model, muzzle_classifier
    
    if None in (detection_model, muzzle_classifier):
        load_models()
    
    detection_data = {
        "dogs": [],
        "dogs_with_muzzle": [],
        "dogs_without_muzzle": [],
        "error": None
    }

    try:
        detections = detection_model(frame, verbose=False)
        
        for result in detections:
            for box in result.boxes:
                if box.conf.item() < settings.CONFIDENCE_THRESHOLD:
                    continue
                    
                if detection_model.names[int(box.cls)] == 'dog':
                    bbox = [round(x) for x in box.xyxy[0].tolist()]
                    bbox_data = {
                        "bbox": bbox,
                        "confidence": box.conf.item()
                    }
                    detection_data["dogs"].append(bbox_data)
                    
                    status = check_for_muzzle(frame, bbox)
                    if status == "without":
                        detection_data["dogs_without_muzzle"].append(bbox_data)
                    elif status == "with":
                        detection_data["dogs_with_muzzle"].append(bbox_data)
    except Exception as e:
        detection_data["error"] = str(e)
    
    return detection_data