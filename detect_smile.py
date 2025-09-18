"""
Скрипт для обнаружения улыбок на изображениях с использованием каскадов Хаара.
Использует detect_face для обнаружения лиц, затем ищет улыбки в области рта.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_faces import load_image_from_path, improve_gray_contrast, det_faces


def detect_smiles(image_path, scale=1.1, neighbors=5, minsize=60, 
                 with_profiles=True, check_eyes=True, smile_neighbors=15):
    """
    Обнаруживает лица на изображении и находит улыбки в области рта.
    
    Args:
        image_path (str): Путь к изображению
        scale (float): Параметр масштабирования для детектора лиц
        neighbors (int): Минимальное количество соседей для детектора лиц
        minsize (int): Минимальный размер лица
        with_profiles (bool): Использовать детекцию профильных лиц
        check_eyes (bool): Проверять наличие глаз
        smile_neighbors (int): Параметр для детектора улыбок
        
    Returns:
        tuple: Изображение с выделенными лицами и улыбками, количество найденных улыбок
    """
    
    # Загружаем изображение
    rgb, src_name = load_image_from_path(image_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Предобработка
    gray = improve_gray_contrast(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Обнаруживаем лица
    faces = det_faces(image_path, scale, neighbors, minsize, with_profiles, check_eyes)
    
    # Загружаем каскад для обнаружения улыбок
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    smile_count = 0
    
    # Для каждого обнаруженного лица ищем улыбку
    for (x, y, w, h) in faces:
        # Определяем область рта (нижняя часть лица)
        roi_gray = gray[y + int(h * 0.6):y + h, x:x + w]
        roi_color = rgb[y + int(h * 0.6):y + h, x:x + w]
        
        # Обнаруживаем улыбки
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=smile_neighbors,
            minSize=(int(w * 0.2), int(h * 0.1))
        )
        
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Рисуем прямоугольники вокруг улыбок
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(rgb, (x + sx, y + int(h * 0.6) + sy), 
                         (x + sx + sw, y + int(h * 0.6) + sy + sh), 
                         (255, 0, 0), 2)
            smile_count += 1
            
            # Добавляем текст "Smile"
            cv2.putText(rgb, 'Smile', (x + sx, y + int(h * 0.6) + sy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return rgb, smile_count


def main():
    """Основная функция для запуска детекции улыбок"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect smiles in images')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--save', default='result_smile.jpg', 
                       help='Path to save the result image (default: result_smile.jpg)')
    parser.add_argument('--scale', type=float, default=1.1,
                       help='Scale factor for face detection (default: 1.1)')
    parser.add_argument('--neighbors', type=int, default=5,
                       help='Min neighbors for face detection (default: 5)')
    parser.add_argument('--minsize', type=int, default=60,
                       help='Minimum face size (default: 60)')
    parser.add_argument('--smile_neighbors', type=int, default=30,
                       help='Min neighbors for smile detection (default: 15)')
    
    args = parser.parse_args()
    
    # Обнаруживаем улыбки
    result_image, smile_count = detect_smiles(
        args.image_path, 
        scale=args.scale,
        neighbors=args.neighbors,
        minsize=args.minsize,
        smile_neighbors=args.smile_neighbors
    )
    
    # Отображаем результат
    print(f"Found {smile_count} smile(s).")
    plt.imshow(result_image)
    plt.title(f"Found {smile_count} smile(s)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Сохраняем результат
    bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.save, bgr)
    print(f"Saved result to {args.save}")


if __name__ == "__main__":
    main()