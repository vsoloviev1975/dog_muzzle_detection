dog_muzzle_detection/
├── app/                          # Основная папка приложения
│   ├── __init__.py
│   ├── main.py                   # Главный файл FastAPI
│   ├── models.py                 # Модели SQLAlchemy
│   ├── detection.py              # Логика детекции собак и намордников
│   ├── database.py               # Настройки подключения к БД
│   ├── report_generator.py       # Генерация PDF отчетов
│   └── utils/                    # Дополнительные утилиты
│       └── helpers.py            # Вспомогательные функции
├── datasets/                     # Папка для датасетов
│   ├── muzzle_classification/    # Датасет для классификации намордников
│   │   ├── train/
│   │   │   ├── with_muzzle/      # Примеры собак с намордниками
│   │   │   │   ├── dog1.jpg
│   │   │   │   └── ...
│   │   │   └── without_muzzle/   # Примеры собак без намордников
│   │   │       ├── dog1.jpg
│   │   │       └── ...
│   │   └── val/                  # Валидационный набор
│   │       ├── with_muzzle/
│   │       └── without_muzzle/
│   └── raw_images/               # Сырые изображения для тестирования
├── static/
│   ├── images/                   # Загруженные пользователями изображения
│   ├── reports/                  # Сгенерированные PDF отчеты
│   └── styles.css                # Стили для фронтенда
├── templates/
│   └── index.html                # HTML шаблон главной страницы
├── models/                       # Обученные модели
│   ├── yolov8n.pt                # Базовая модель YOLOv8
│   └── yolov8n_muzzle_cls.pt     # Дообученная модель классификации намордников
├── notebooks/                    # Jupyter ноутбуки для анализа/обучения
│   └── muzzle_classification.ipynb
├── requirements.txt              # Зависимости Python
├── README.md                     # Документация проекта
└── .env                          # Переменные окружения (опционально)dog_detection/

Запуск:
uvicorn app.main:app --reload