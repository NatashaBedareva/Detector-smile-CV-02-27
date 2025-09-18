
# Face and Smile Detection CV-01-08

## О программе

Программа для обнаружения улыбок на изображениях с использованием каскадов Хаара (OpenCV). Включает предобработку изображения, обнаружение лиц (анфас и профиль), фильтрацию по наличию глаз с использованием решения задачи  CV-1-11 detect_faces.py

## Использование

Виртуальное окружение

```bash
   # Для Windows
   python -m venv venv
   venv\Scripts\activate

   # Для Linux/MacOS
   python -m venv venv
   source venv/bin/activate
```

Установка библиотек

```bash
pip install -r requirements.txt
```
Запуск програмы

```bash
python detect_smile.py <путь к изображению>
```

Пример


```bash
python detect_smile.py smile.png
```

## Параметры командной строки
```bash
python detect_smile.py image.jpg \
    --scale 1.1 \          # Фактор масштабирования для детектора лиц
    --neighbors 5 \        # Минимальное количество соседей для детектора лиц
    --minsize 60 \         # Минимальный размер лица
    --smile_neighbors 15 \ # Параметр для детектора улыбок
    --save result.jpg      # Путь для сохранения результата
```
