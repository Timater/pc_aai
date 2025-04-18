"""
Модуль для обучения и использования моделей YOLO для детекции UI элементов
"""

import os
import cv2
import logging
import numpy as np
import shutil
from typing import List, Dict, Tuple, Union, Optional, Any

# Проверяем наличие необходимых библиотек
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("Библиотека torch не установлена. Обучение моделей YOLO будет недоступно.")

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logging.warning("Библиотека ultralytics не установлена. Обучение моделей YOLO будет недоступно.")

logger = logging.getLogger(__name__)

class UIYOLODetector:
    """
    Класс для обучения и использования моделей YOLO для детекции UI элементов
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 class_names: Optional[List[str]] = None):
        """
        Инициализация детектора UI элементов на основе YOLO

        Args:
            model_path (str, optional): Путь к предобученной модели YOLO.
                Если None, будет использоваться yolov8n.pt.
            class_names (List[str], optional): Список названий классов UI элементов.
                Если None, будет использован стандартный список.
        """
        # Проверяем наличие необходимых библиотек
        if not HAS_TORCH or not HAS_ULTRALYTICS:
            raise ImportError("Для работы UIYOLODetector требуются библиотеки torch и ultralytics. "
                             "Установите их командами: "
                             "pip install torch torchvision"
                             "pip install ultralytics")
        
        # Стандартные классы UI элементов
        self.default_class_names = [
            "button", "text_field", "checkbox", "radio_button", "dropdown", 
            "toggle", "slider", "icon", "tab", "window", "dialog", "menu"
        ]
        
        # Используем переданный список классов или стандартный
        self.class_names = class_names if class_names is not None else self.default_class_names
        
        # Инициализация модели
        self.model_path = model_path
        self.model = None
        
        # Если указан путь к модели, загружаем её
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """
        Загрузка модели YOLO

        Args:
            model_path (str): Путь к модели YOLO
        """
        logger.info(f"Загрузка модели YOLO из {model_path}")
        
        try:
            # Проверяем существование файла
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            
            # Загружаем модель
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            logger.info(f"Модель YOLO успешно загружена: {model_path}")
            
            # Проверяем соответствие классов
            model_class_names = self.model.names
            model_class_count = len(model_class_names)
            
            logger.info(f"Модель имеет {model_class_count} классов: {model_class_names}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели YOLO: {e}")
            raise
    
    def train(self, 
              data_yaml: str,
              epochs: int = 100,
              img_size: int = 640,
              batch: int = 16,
              name: str = "ui_detector") -> str:
        """
        Обучение модели YOLO на данных UI элементов

        Args:
            data_yaml (str): Путь к YAML-файлу с конфигурацией данных
            epochs (int): Количество эпох обучения
            img_size (int): Размер изображений для обучения
            batch (int): Размер батча
            name (str): Название проекта обучения

        Returns:
            str: Путь к обученной модели
        """
        logger.info(f"Подготовка к обучению модели YOLO")
        
        # Проверяем существование файла конфигурации данных
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Файл конфигурации данных не найден: {data_yaml}")
        
        # Проверяем валидность YAML-файла
        try:
            import yaml
            with open(data_yaml, 'r') as file:
                data_config = yaml.safe_load(file)
            
            # Проверяем наличие необходимых ключей в конфигурации
            required_keys = ['train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data_config:
                    raise ValueError(f"В конфигурации данных отсутствует обязательный ключ: {key}")
            
            # Проверяем пути к данным
            train_path = data_config['train']
            val_path = data_config['val']
            
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Не найден путь к тренировочным данным: {train_path}")
            
            if not os.path.exists(val_path):
                raise FileNotFoundError(f"Не найден путь к валидационным данным: {val_path}")
            
            # Проверяем соответствие классов
            if len(data_config['names']) != data_config['nc']:
                raise ValueError(f"Количество классов ({data_config['nc']}) не соответствует "
                               f"количеству названий классов ({len(data_config['names'])})")
            
            logger.info(f"Конфигурация данных валидна: {data_config['nc']} классов, "
                      f"{len(os.listdir(train_path))} файлов для обучения, "
                      f"{len(os.listdir(val_path))} файлов для валидации")
            
        except Exception as e:
            logger.error(f"Ошибка при проверке конфигурации данных: {e}")
            raise
        
        # Инициализируем новую модель YOLO для обучения
        try:
            # Используем YOLOv8n в качестве базовой модели
            self.model = YOLO('yolov8n.pt')
            
            # Начинаем обучение
            logger.info(f"Начало обучения модели YOLO на {epochs} эпохах")
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch,
                name=name,
                verbose=True
            )
            
            # Получаем путь к обученной модели
            trained_model_path = str(results.model.ckpt_path)
            logger.info(f"Обучение модели завершено. Модель сохранена: {trained_model_path}")
            
            # Копируем модель в директорию models
            os.makedirs('models', exist_ok=True)
            model_filename = os.path.basename(trained_model_path)
            output_path = os.path.join('models', model_filename)
            
            shutil.copy2(trained_model_path, output_path)
            logger.info(f"Модель скопирована в: {output_path}")
            
            # Обновляем текущую модель
            self.model_path = output_path
            
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise
    
    def export_model(self, 
                     format: str = 'onnx', 
                     output_path: Optional[str] = None,
                     optimize: bool = True) -> str:
        """
        Экспорт модели YOLO в другие форматы (ONNX, TFLite и др.)

        Args:
            format (str): Формат для экспорта ('onnx', 'tflite', 'torchscript' и др.)
            output_path (str, optional): Путь для сохранения экспортированной модели
            optimize (bool): Применять ли оптимизации при экспорте

        Returns:
            str: Путь к экспортированной модели
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала загрузите или обучите модель.")
        
        # Валидация формата
        supported_formats = ['onnx', 'torchscript', 'openvino', 'tflite', 'pb', 'coreml']
        if format not in supported_formats:
            raise ValueError(f"Неподдерживаемый формат: {format}. "
                           f"Поддерживаемые форматы: {', '.join(supported_formats)}")
        
        # Если путь не указан, используем стандартный
        if output_path is None:
            os.makedirs('models', exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.model_path))[0]
            output_path = f"models/{base_name}.{format}"
        
        logger.info(f"Экспорт модели в формат {format}: {output_path}")
        
        try:
            # Экспортируем модель
            export_path = self.model.export(format=format, optimize=optimize)
            
            # Если путь экспорта отличается от желаемого, копируем файл
            if str(export_path) != output_path:
                shutil.copy2(export_path, output_path)
                logger.info(f"Модель экспортирована и скопирована в: {output_path}")
            else:
                logger.info(f"Модель успешно экспортирована: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте модели: {e}")
            raise
    
    def detect(self, image: Union[str, np.ndarray], conf: float = 0.25, iou: float = 0.45) -> List[Dict]:
        """
        Детекция UI элементов на изображении

        Args:
            image (str или numpy.ndarray): Путь к изображению или загруженное изображение
            conf (float): Порог уверенности для детекции (0.0-1.0)
            iou (float): Порог IOU для NMS (0.0-1.0)

        Returns:
            list: Список обнаруженных UI элементов с координатами, классами и уверенностью
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала загрузите или обучите модель.")
        
        logger.info(f"Начало детекции UI элементов с использованием YOLO модели")
        
        # Проверяем, что изображение существует, если это путь
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Изображение не найдено: {image}")
            
            # Сохраняем путь для отладки
            image_path = image
        else:
            # Если это массив, сохраняем временное изображение для отладки
            image_path = "temp_detection_image.jpg"
            cv2.imwrite(image_path, image)
        
        try:
            # Выполняем инференс модели
            results = self.model(image, conf=conf, iou=iou, verbose=False)
            
            # Преобразуем результаты в список словарей
            detections = []
            
            # Если изображение - путь к файлу, загружаем его для получения размеров
            if isinstance(image, str):
                img = cv2.imread(image)
                height, width = img.shape[:2]
            else:
                height, width = image.shape[:2]
            
            # Обрабатываем результаты
            for result in results:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Получаем координаты, уверенность и класс
                    box = boxes[i].xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    conf = float(boxes[i].conf)
                    cls_id = int(boxes[i].cls)
                    
                    # Ограничиваем координаты размерами изображения
                    x1, y1, x2, y2 = box
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(width, int(x2))
                    y2 = min(height, int(y2))
                    
                    # Получаем название класса
                    if cls_id < len(self.class_names):
                        class_name = self.class_names[cls_id]
                    else:
                        # Если модель использует другие классы, берем их из модели
                        class_name = self.model.names[cls_id]
                    
                    # Формируем словарь с информацией о детекции
                    detection_info = {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "box": [x1, y1, x2, y2],
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "center_x": int((x1 + x2) / 2),
                        "center_y": int((y1 + y2) / 2),
                        "relative_x": float((x1 + x2) / (2 * width)),
                        "relative_y": float((y1 + y2) / (2 * height))
                    }
                    
                    detections.append(detection_info)
            
            # Сортируем по уверенности (убывающий порядок)
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Обнаружено UI элементов: {len(detections)}")
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении детекции: {e}")
            raise
    
    def find_element_by_name(self, 
                            detections: List[Dict], 
                            class_name: str, 
                            min_confidence: float = 0.0) -> Optional[Dict]:
        """
        Поиск элемента по названию класса

        Args:
            detections (list): Список детекций
            class_name (str): Название класса элемента
            min_confidence (float): Минимальная уверенность для фильтрации

        Returns:
            dict or None: Найденный элемент или None, если не найден
        """
        filtered = [d for d in detections if d["class_name"] == class_name and d["confidence"] >= min_confidence]
        if filtered:
            # Возвращаем элемент с наивысшей уверенностью
            return max(filtered, key=lambda x: x["confidence"])
        return None
    
    def find_element_by_coordinates(self, 
                                   detections: List[Dict], 
                                   x: int, 
                                   y: int) -> Optional[Dict]:
        """
        Поиск элемента по координатам точки

        Args:
            detections (list): Список детекций
            x (int): Координата X точки
            y (int): Координата Y точки

        Returns:
            dict or None: Найденный элемент или None, если не найден
        """
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return det
        return None
    
    def visualize_detections(self, 
                            image: Union[str, np.ndarray], 
                            detections: List[Dict], 
                            output_path: Optional[str] = None, 
                            show: bool = False) -> np.ndarray:
        """
        Визуализация результатов детекции

        Args:
            image (str или numpy.ndarray): Путь к изображению или загруженное изображение
            detections (list): Список детекций
            output_path (str, optional): Путь для сохранения результата визуализации
            show (bool): Показать результаты с помощью OpenCV

        Returns:
            numpy.ndarray: Изображение с отрисованными боксами
        """
        # Загружаем изображение, если это путь
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Изображение не найдено: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Не удалось прочитать изображение: {image}")
        else:
            # Если это уже массив numpy, делаем копию
            img = image.copy()
        
        # Создаем цветовую палитру для классов
        np.random.seed(42)  # для воспроизводимости
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        # Отрисовываем каждую детекцию
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_id = det["class_id"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Выбираем цвет для класса
            color = tuple(map(int, colors[class_id % len(colors)]))
            
            # Рисуем бокс
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Рисуем фон для текста
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Рисуем текст
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Сохраняем результат, если указан путь
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(output_path, img)
            logger.info(f"Визуализация сохранена в {output_path}")
        
        # Показываем результат
        if show:
            cv2.imshow("Detections", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Проверка наличия необходимых библиотек
    if not HAS_TORCH or not HAS_ULTRALYTICS:
        logger.error("Для работы необходимо установить torch и ultralytics")
        exit(1)
    
    # Пример использования детектора
    try:
        # Путь к предобученной модели YOLO
        model_path = "models/yolov8n.pt"
        
        if os.path.exists(model_path):
            # Инициализация детектора
            detector = UIYOLODetector(model_path=model_path)
            
            # Тестовое изображение
            test_image_path = "test_screenshot.png"
            
            if os.path.exists(test_image_path):
                # Детекция объектов
                detections = detector.detect(test_image_path)
                
                # Вывод результатов
                for i, det in enumerate(detections):
                    logger.info(f"{i+1}. {det['class_name']} (уверенность: {det['confidence']:.2f}), "
                             f"координаты: {det['box']}")
                
                # Визуализация результатов
                detector.visualize_detections(
                    test_image_path, 
                    detections, 
                    output_path="output_detection.jpg"
                )
            else:
                logger.warning(f"Тестовое изображение не найдено: {test_image_path}")
        else:
            logger.warning(f"Модель не найдена: {model_path}")
            logger.info("Для обучения своей модели используйте метод train()")
    
    except Exception as e:
        logger.error(f"Ошибка: {e}") 