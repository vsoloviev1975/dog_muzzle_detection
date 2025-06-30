from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, Table, PageBreak, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from datetime import datetime
import cv2
import numpy as np
import os
from tempfile import mkdtemp
import logging
from .config import settings
from io import BytesIO


logger = logging.getLogger(__name__)

# Регистрация русского шрифта
def register_russian_font():
    """Регистрирует русский шрифт для корректного отображения кириллицы в PDF"""
    try:
        # Попробуем использовать стандартный шрифт Arial
        try:
            pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            return 'Arial'
        except:
            # Если Arial не найден, используем встроенный шрифт
            from reportlab.rl_config import TTFSearchPath
            TTFSearchPath.append(os.path.dirname(__file__))
            try:
                pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
                return 'DejaVuSans'
            except:
                logger.warning("Русский шрифт не найден, используется Helvetica")
                return 'Helvetica'
    except Exception as e:
        logger.error(f"Ошибка регистрации шрифта: {str(e)}")
        return 'Helvetica'

# Получение стилей с русским шрифтом
def get_russian_styles():
    """Возвращает стили для PDF с поддержкой русского языка"""
    styles = getSampleStyleSheet()
    font_name = register_russian_font()
    
    # Регистрируем bold версию шрифта
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'))
    except Exception as e:
        logger.warning(f"Не удалось загрузить bold шрифт: {str(e)}")
    
    # Список стандартных стилей, которые нужно настроить
    style_names = [
        'Normal', 'BodyText', 'Italic', 'Heading1', 'Heading2', 
        'Heading3', 'Heading4', 'Title', 'Bullet', 'Definition', 'Code'
    ]
    
    # Настраиваем только существующие стили
    for style_name in style_names:
        if style_name in styles:
            styles[style_name].fontName = font_name
            styles[style_name].encoding = 'UTF-8'
    
    # Дополнительные настройки
    styles['Heading1'].fontSize = 16
    styles['Heading1'].textColor = colors.HexColor("#2c3e50")
    styles['Heading2'].fontSize = 14
    styles['BodyText'].spaceAfter = 12
    
    return styles

def generate_pdf_report(media_path, detection_data, output_path, is_video=False):
    """
    Генерирует PDF отчет с результатами детекции
    Для видео включает только первый кадр с нарушением
    """
    try:
        # Проверяем и создаем директорию для отчета
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Инициализация PDF документа
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = get_russian_styles()
        story = []
        
        # Формируем заголовок отчета
        title = "ОТЧЕТ ПО ПРОВЕРКЕ СОБАК НА НАМОРДНИКИ"
        subtitle = ""
        
        if is_video:
            # Проверяем наличие нарушений в видео
            has_violations = any(
                frame["results"].get("dogs_without_muzzle", [])
                for frame in detection_data.get("frame_results", [])
            )
            
            title += " (ВИДЕОФАЙЛ)"
            subtitle = "Нарушения обнаружены" if has_violations else "Нарушений не обнаружено"
        else:
            # Для изображений
            title += " (ИЗОБРАЖЕНИЕ)"
            subtitle = "Нарушения обнаружены" if detection_data.get("dogs_without_muzzle") else "Нарушений не обнаружено"

        # Добавляем заголовок и подзаголовок
        title_style = styles["Heading1"]
        title_style.textColor = colors.darkblue
        story.append(Paragraph(title, title_style))
        
        subtitle_style = styles["Heading2"]
        subtitle_style.textColor = colors.red if "обнаружены" in subtitle else colors.green
        story.append(Paragraph(subtitle, subtitle_style))
        story.append(Spacer(1, 0.3 * inch))

        # Добавляем информацию о файле
        file_info = [
            f"Файл: {os.path.basename(media_path)}",
            f"Дата проверки: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            f"Тип: {'Видео' if is_video else 'Изображение'}"
        ]
        
        for info in file_info:
            story.append(Paragraph(info, styles["BodyText"]))
        story.append(Spacer(1, 0.3 * inch))

        # Обработка медиа-контента
        if is_video:
            # Для видео - только первый кадр с нарушением
            process_video_report(media_path, detection_data, story, styles)
        else:
            # Для изображений - полная обработка
            process_image_report(media_path, detection_data, story, styles)

        # Добавляем статистику
        stats = calculate_stats(detection_data, is_video)
        story.extend(add_statistics_section(stats, styles))
        
        # Добавляем заключение
        story.extend(add_conclusion_section(detection_data, styles))
        
        # Добавляем служебную информацию
        story.append(Spacer(1, 0.5 * inch))
        footer_text = "Отчет сгенерирован автоматически. Система детекции собак без намордников"
        
        # Добавляем версию только если settings доступен
        try:
            footer_text += f" v{settings.APP_VERSION}"
        except:
            pass
            
        footer = Paragraph(footer_text, styles["Italic"])
        story.append(footer)

        # Генерация PDF
        doc.build(story)
        logger.info(f"PDF отчет успешно сгенерирован: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка генерации PDF отчета: {str(e)}", exc_info=True)
        raise RuntimeError(f"Ошибка генерации отчета: {str(e)}")

def process_video_report(video_path, detection_data, story, styles):
    """Обрабатывает видео и добавляет кадры с обнаружениями в отчет с проверкой на ложные срабатывания"""
    try:
        # Собираем все кадры с нарушениями
        violation_frames = []
        prev_frame_idx = -10  # Для проверки последовательных кадров
        
        for frame_result in detection_data.get("frame_results", []):
            if frame_result["results"].get("dogs_without_muzzle"):
                current_idx = frame_result["frame_index"]
                
                # Проверяем, что это наружение на нескольких последовательных кадрах
                # (разница менее 2 кадров считается последовательной)
                if current_idx - prev_frame_idx <= 2:
                    violation_frames.append(frame_result)
                prev_frame_idx = current_idx
        
        # Если нарушений нет или они не подтверждены на нескольких кадрах
        if not violation_frames:
            story.append(Paragraph("Не обнаружено собак без намордников", styles["BodyText"]))
            return
        
        # Берем только первое подтвержденное нарушение (минимум 2 кадра подряд)
        frame_result = violation_frames[0]
        first_violation_time = frame_result["frame_time"]
        frame_idx = frame_result["frame_index"]
        
        # Открываем видео и получаем нужный кадр
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Рисуем bounding boxes прямо в памяти
            for box in frame_result["results"]["dogs_without_muzzle"]:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "No Muzzle", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            # Конвертируем кадр в временный файл в памяти
            _, img_buffer = cv2.imencode('.jpg', frame)
            img_buffer = BytesIO(img_buffer.tobytes())
            
            # Добавляем изображение в отчет
            img = Image(img_buffer, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
            
            # Добавляем информацию о времени и количестве подтверждений
            time_text = (f"Нарушение обнаружено на {first_violation_time:.1f} секунде. "
                        f"Подтверждено на {len(violation_frames)} кадрах.")
            story.append(Paragraph(time_text, styles["Heading4"]))
            story.append(Spacer(1, 0.2 * inch))
        
    except Exception as e:
        logger.error(f"Ошибка обработки видео: {str(e)}", exc_info=True)
        story.append(Paragraph("Ошибка обработки видео", styles["BodyText"]))

def process_image_report(image_path, detection_data, story, styles):
    """Обрабатывает изображение и добавляет его в отчет"""
    add_media_to_story(image_path, story, styles)
    
    # Добавляем информацию о количестве обнаружений
    if detection_data.get("dogs_without_muzzle"):
        count = len(detection_data["dogs_without_muzzle"])
        alert_text = f"<b>ВНИМАНИЕ:</b> Обнаружено {count} собак без намордников!"
        story.append(Paragraph(alert_text, styles["BodyText"]))
    else:
        story.append(Paragraph("Нарушений не обнаружено", styles["BodyText"]))


def save_detection_frames(video_path, detection_data, output_dir):
    """Сохраняет кадры с обнаружениями"""
    frames_to_save = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видео: {video_path}")
        return frames_to_save

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for frame_result in detection_data.get("frame_results", []):
        if frame_result["results"]["dogs_without_muzzle"]:
            frame_time = frame_result["frame_time"]
            frame_idx = frame_result["frame_index"]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Рисуем bounding boxes
                for box in frame_result["results"]["dogs_without_muzzle"]:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "No Muzzle", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                
                # Сохраняем кадр
                frame_path = os.path.join(output_dir, f"detection_{frame_time:.1f}_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)
                frames_to_save.append(frame_path)
    
    cap.release()
    return frames_to_save

def calculate_stats(detection_data, is_video):
    """Вычисляет статистику для отчета"""
    stats = {
        "total_dogs": 0,
        "with_muzzle": 0,
        "without_muzzle": 0,
        "detection_times": []
    }
    
    if is_video:
        for frame in detection_data.get("frame_results", []):
            stats["total_dogs"] += len(frame["results"]["dogs"])
            stats["with_muzzle"] += len(frame["results"]["dogs_with_muzzle"])
            stats["without_muzzle"] += len(frame["results"]["dogs_without_muzzle"])
            
            if frame["results"]["dogs_without_muzzle"]:
                stats["detection_times"].append(frame["frame_time"])
    else:
        stats["total_dogs"] = len(detection_data["dogs"])
        stats["with_muzzle"] = len(detection_data["dogs_with_muzzle"])
        stats["without_muzzle"] = len(detection_data["dogs_without_muzzle"])
    
    return stats

def add_statistics_section(stats, styles):
    """Добавляет секцию со статистикой"""
    section = []
    
    section.append(Paragraph("Статистика обнаружений", styles["Heading2"]))
    section.append(Spacer(1, 0.2 * inch))
    
    stats_text = (
        f"Всего собак обнаружено: {stats['total_dogs']}<br/>"
        f"С намордниками: {stats['with_muzzle']}<br/>"
        f"Без намордников: {stats['without_muzzle']}"
    )
    
    if stats['detection_times']:
        times = ", ".join(f"{t:.1f} сек" for t in stats['detection_times'])
        stats_text += f"<br/>Моменты обнаружения: {times}"
    
    section.append(Paragraph(stats_text, styles["BodyText"]))
    section.append(Spacer(1, 0.2 * inch))
    
    return section

def add_conclusion_section(detection_data, styles):
    """Добавляет заключительную секцию"""
    section = []
    
    if has_violations(detection_data):
        alert_style = styles["BodyText"]
        alert_style.textColor = colors.red
        section.append(Paragraph("ВНИМАНИЕ: Обнаружены собаки без намордников!", alert_style))
    else:
        success_style = styles["BodyText"]
        success_style.textColor = colors.green
        section.append(Paragraph("Нарушений не обнаружено. Все собаки в намордниках.", success_style))
    
    # Дата генерации
    section.append(Spacer(1, 0.5 * inch))
    section.append(Paragraph(f"Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Italic"]))
    
    return section

def has_violations(detection_data):
    """Проверяет наличие нарушений"""
    if "frame_results" in detection_data:
        return any(
            frame["results"].get("dogs_without_muzzle", []) 
            for frame in detection_data["frame_results"]
            if frame.get("results")
        )
    return bool(detection_data.get("dogs_without_muzzle", []))

def add_media_to_story(image_path, story, styles):
    """Добавляет медиа-элемент в отчет"""
    try:
        if os.path.exists(image_path):
            img = Image(image_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
        else:
            story.append(Paragraph("Изображение не найдено", styles["BodyText"]))
    except Exception as e:
        story.append(Paragraph(f"Ошибка загрузки медиа: {str(e)}", styles["BodyText"]))
        
def generate_full_pdf_report(items, output_path):
    """Генерирует PDF отчет по всем записям в БД"""
    try:
        # Создаем PDF документ
        doc = SimpleDocTemplate(output_path, pagesize=landscape(letter))
        styles = get_russian_styles()
        story = []
        
        # Заголовок отчета
        title = Paragraph(
            "ПОЛНЫЙ ОТЧЕТ ПО ПРОВЕРКАМ СОБАК НА НАМОРДНИКИ",
            styles["Heading1"]
        )
        story.append(title)
        story.append(Spacer(1, 0.3 * inch))
        
        # Информация о генерации
        gen_info = Paragraph(
            f"Отчет сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            styles["BodyText"]
        )
        story.append(gen_info)
        story.append(Spacer(1, 0.2 * inch))
        
        # Добавляем таблицу с данными
        data = [
            [
                Paragraph("Имя файла", styles["BodyText"]),
                Paragraph("Дата проверки", styles["BodyText"]),
                Paragraph("Тип", styles["BodyText"]),
                Paragraph("Результат", styles["BodyText"])
            ]
        ]
        
        for item in items:
            date_str = item.created_at.strftime('%d.%m.%Y %H:%M:%S')
            file_type = "Видео" if item.is_video else "Изображение"
            result = "Нарушение" if item.has_no_muzzle else "Норма"
            
            data.append([
                Paragraph(item.filename, styles["Normal"]),
                Paragraph(date_str, styles["Normal"]),
                Paragraph(file_type, styles["Normal"]),
                Paragraph(result, styles["Normal"])
            ])
        
        # Ширины столбцов
        col_widths = [3.5 * inch, 1.8 * inch, 1.5 * inch, 2.0 * inch]
        
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            # Изменено на светло-серый фон для заголовка
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
            # Черный текст для лучшей читаемости на светлом фоне
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), styles["BodyText"].fontName),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 1), (-1, -1), styles["Normal"].fontName),
            ('WORDWRAP', (0, 0), (-1, -1), True),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.5 * inch))
        
        # Статистика
        total = len(items)
        violations = sum(1 for item in items if item.has_no_muzzle)
        stats_text = f"Всего проверок: {total}, нарушений: {violations} ({(violations/total)*100:.1f}%)"
        story.append(Paragraph(stats_text, styles["BodyText"]))
        
        doc.build(story)
        return True
        
    except Exception as e:
        logger.error(f"Ошибка генерации отчета: {str(e)}", exc_info=True)
        raise RuntimeError(f"Ошибка генерации отчета: {str(e)}")