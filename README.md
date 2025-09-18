
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