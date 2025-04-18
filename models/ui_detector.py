"""
Модуль для обучения и использования моделей YOLO для детекции UI элементов
"""

import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np

class UIYOLODetector:
    """
    Класс для обучения и использования модели YOLO для детекции UI элементов
    """
    
    def __init__(self, model_path=None):
        """
        Инициализация детектора
        
        Args:
            model_path (str, optional): Путь к модели. Если None, будет использована предварительно обученная модель YOLOv8n.
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print("Модель не найдена, будет использована предобученная YOLOv8n")
            self.model = YOLO("yolov8n.pt")
        
        self.class_names = [
            "button", "text_field", "checkbox", "radio", "dropdown", 
            "menu", "icon", "tab", "scrollbar", "window", "dialog"
        ]
    
    def train(self, data_yaml, epochs=100, img_size=640, batch=16, name="ui_detector"):
        """
        Обучение модели YOLO на пользовательском датасете
        
        Args:
            data_yaml (str): Путь к YAML-файлу с конфигурацией датасета
            epochs (int): Количество эпох обучения
            img_size (int): Размер изображения для обучения
            batch (int): Размер батча
            name (str): Название эксперимента
            
        Returns:
            dict: Результаты обучения
        """
        print(f"Запуск обучения модели на {epochs} эпох, размер изображения {img_size}...")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            name=name
        )
        
        print("Обучение завершено!")
        return results
    
    def export_model(self, format="onnx", output_path="models/detector.onnx"):
        """
        Экспорт модели в различные форматы
        
        Args:
            format (str): Формат экспорта (onnx, torchscript, tflite и т.д.)
            output_path (str): Путь для сохранения экспортированной модели
            
        Returns:
            str: Путь к экспортированной модели
        """
        print(f"Экспортирование модели в формат {format}...")
        
        # Обеспечиваем существование директории для сохранения
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Экспортируем модель
        exported_model = self.model.export(format=format)
        
        # Копируем в указанную директорию
        if os.path.exists(exported_model) and output_path != exported_model:
            import shutil
            shutil.copy(exported_model, output_path)
            print(f"Модель экспортирована и сохранена в {output_path}")
        else:
            print(f"Модель экспортирована в {exported_model}")
        
        return output_path
    
    def detect(self, image, conf=0.25):
        """
        Детекция объектов на изображении
        
        Args:
            image (numpy.ndarray или str): Изображение или путь к файлу изображения
            conf (float): Порог уверенности для детекций
            
        Returns:
            list: Список обнаруженных объектов с классами и координатами
        """
        # Проверяем тип входных данных
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                raise FileNotFoundError(f"Файл изображения не найден: {image}")
        
        # Запускаем детекцию
        results = self.model.predict(image, conf=conf, verbose=False)
        
        # Обрабатываем результаты
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                # Получаем имя класса, если доступно
                if cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "center_x": int((x1 + x2) / 2),
                    "center_y": int((y1 + y2) / 2)
                })
        
        return detections
    
    def visualize_detections(self, image, detections, output_path=None, show=True):
        """
        Визуализация детекций на изображении
        
        Args:
            image (numpy.ndarray): Входное изображение
            detections (list): Список детекций
            output_path (str, optional): Путь для сохранения визуализации
            show (bool): Показывать ли изображение
            
        Returns:
            numpy.ndarray: Изображение с визуализированными детекциями
        """
        # Копируем изображение для рисования
        vis_image = image.copy()
        
        # Определяем цвета для классов
        colors = {
            "button": (0, 0, 255),      # Красный
            "text_field": (0, 255, 0),  # Зеленый
            "checkbox": (255, 0, 0),    # Синий
            "radio": (255, 255, 0),     # Голубой
            "dropdown": (255, 0, 255),  # Пурпурный
            "menu": (0, 255, 255),      # Желтый
            "icon": (128, 0, 128),      # Фиолетовый
            "tab": (128, 128, 0),       # Оливковый
            "scrollbar": (0, 128, 128), # Темно-голубой
            "window": (128, 0, 0),      # Бордовый
            "dialog": (0, 128, 0)       # Темно-зеленый
        }
        
        # Рисуем каждую детекцию
        for det in detections:
            # Получаем параметры детекции
            x1, y1, x2, y2 = det["box"]
            class_name = det["class_name"]
            conf = det["confidence"]
            
            # Определяем цвет для класса
            color = colors.get(class_name, (255, 255, 255))  # По умолчанию белый
            
            # Рисуем прямоугольник
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Рисуем текст
            label = f"{class_name} {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Сохраняем изображение, если указан путь
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Визуализация сохранена в {output_path}")
        
        # Показываем изображение, если нужно
        if show:
            cv2.imshow("Detection Results", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return vis_image

# Пример использования
if __name__ == "__main__":
    # Создаем детектор
    detector = UIYOLODetector()
    
    # Путь к тестовому изображению
    test_image_path = "test_screenshot.png"
    
    # Проверяем наличие тестового изображения
    if os.path.exists(test_image_path):
        # Загружаем изображение
        image = cv2.imread(test_image_path)
        
        # Детектируем объекты
        detections = detector.detect(image)
        
        # Выводим результаты
        print(f"Обнаружено элементов: {len(detections)}")
        for det in detections:
            print(f"Класс: {det['class_name']}, Уверенность: {det['confidence']:.2f}, Координаты: {det['box']}")
        
        # Визуализируем результаты
        detector.visualize_detections(image, detections)
    else:
        print(f"Тестовое изображение не найдено: {test_image_path}")
        
        # Пример обучения модели (закомментирован, чтобы не запускать случайно)
        """
        # Путь к конфигурации датасета
        data_yaml = "dataset/data.yaml"
        
        # Обучаем модель
        detector.train(data_yaml, epochs=100)
        
        # Экспортируем модель в ONNX
        detector.export_model(format="onnx")
        """ 