"""
YOLOv8 детектор UI элементов
Использует модели YOLOv8 для обнаружения элементов пользовательского интерфейса
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from .ui_detector import UIDetector

logger = logging.getLogger(__name__)

class YoloDetector(UIDetector):
    """
    Детектор UI элементов на основе YOLOv8
    
    Поддерживает модели YOLOv8 для обнаружения элементов интерфейса.
    Требует установки библиотеки ultralytics.
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        confidence_threshold: float = 0.5,
        custom_model: bool = False,
        device: str = None,
        **kwargs
    ):
        """
        Инициализация детектора UI элементов YOLOv8
        
        Args:
            model_path (str): Путь к модели YOLOv8 или название предобученной модели
            class_names (List[str]): Список имен классов UI элементов
            confidence_threshold (float): Порог уверенности для фильтрации обнаружений
            custom_model (bool): Флаг, указывающий, что используется собственная модель
            device (str): Устройство для запуска: 'cpu', 'cuda', '0' и т.д.
            **kwargs: Дополнительные параметры
        """
        super().__init__(
            model_path=model_path,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
        self.custom_model = custom_model
        self.device = device
        self.model = None
        
        # Загружаем модель при инициализации
        self._load_model()
        
    def _load_model(self) -> bool:
        """
        Загрузка модели YOLOv8
        
        Returns:
            bool: True, если модель успешно загружена
        """
        if not self._ensure_model_loaded():
            return False
        
        try:
            # Импортируем библиотеку ultralytics в функции, чтобы она не требовалась
            # если не используется YOLOv8
            from ultralytics import YOLO
            
            # Загружаем модель
            self.model = YOLO(self.model_path)
            
            # Если были предоставлены имена классов, проверяем соответствие
            model_classes = len(self.model.names)
            if self.class_names and len(self.class_names) != model_classes:
                logger.warning(
                    f"Количество предоставленных имен классов ({len(self.class_names)}) "
                    f"не соответствует количеству классов модели ({model_classes}). "
                    f"Будут использованы имена классов из модели."
                )
                self.class_names = [self.model.names[i] for i in range(model_classes)]
            elif not self.class_names:
                # Если имена классов не предоставлены, используем из модели
                self.class_names = [self.model.names[i] for i in range(model_classes)]
            
            logger.info(f"Модель YOLOv8 успешно загружена из {self.model_path}")
            logger.info(f"Классы модели: {self.class_names}")
            logger.info(f"Устройство: {self.device or 'auto'}")
            
            return True
        except ImportError:
            logger.error("Не удалось импортировать библиотеку ultralytics. "
                         "Установите ее командой: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Ошибка загрузки модели YOLOv8: {e}")
            return False
    
    def _ensure_model_loaded(self) -> bool:
        """
        Проверяет, загружена ли модель, и загружает её, если необходимо
        
        Returns:
            bool: True, если модель успешно загружена или уже была загружена
        """
        if self.model is not None:
            return True
        
        # Проверка пути к модели
        if self.model_path is None:
            logger.error("Путь к модели YOLOv8 не задан")
            return False
        
        # Для предобученных моделей не требуется проверка существования файла
        if not self.custom_model:
            return True
        
        # Проверка существования файла модели
        if not os.path.exists(self.model_path):
            logger.error(f"Файл модели YOLOv8 не найден: {self.model_path}")
            return False
        
        return True  # Разрешаем загрузку
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Обнаружение UI элементов на изображении с помощью YOLOv8
        
        Args:
            image (np.ndarray): Исходное изображение (BGR)
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных элементов с атрибутами
        """
        if not self._ensure_model_loaded():
            logger.error("Модель YOLOv8 не загружена, обнаружение невозможно")
            return []
        
        try:
            # Получаем результаты предсказания
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                device=self.device
            )[0]  # Берем первый результат, т.к. мы передаем только одно изображение
            
            detections = []
            
            # Преобразуем результаты в наш формат
            for i, box in enumerate(results.boxes):
                # Получаем координаты бокса (xyxy формат)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Получаем id класса и уверенность
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Получаем имя класса
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Извлекаем текст, если есть
                text = self._extract_text(image, (x1, y1, x2, y2))
                
                # Формируем описание обнаружения
                detection = {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class': class_name,
                    'text': text
                }
                
                detections.append(detection)
            
            logger.info(f"Обнаружено {len(detections)} UI элементов")
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении обнаружения YOLOv8: {e}")
            return []


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Список моделей для тестирования
    models = [
        "yolov8n.pt",           # Нано модель, предобученная на COCO
        "models/ui_yolov8.pt"    # Собственная модель (если есть)
    ]
    
    # Пробуем загрузить и использовать модель
    model_loaded = False
    
    for model_path in models:
        custom_model = not model_path.startswith("yolov8")
        
        try:
            if custom_model and not os.path.exists(model_path):
                print(f"Пропуск модели {model_path} - файл не найден")
                continue
                
            # Создаем экземпляр детектора
            detector = YoloDetector(
                model_path=model_path,
                custom_model=custom_model
            )
            
            model_loaded = True
            print(f"Модель {model_path} успешно загружена")
            
            # Проверяем на тестовом изображении
            test_images = ["test/test_ui_screenshot.png", "test/desktop_screenshot.png"]
            
            for img_path in test_images:
                if os.path.exists(img_path):
                    print(f"Тестирование на изображении: {img_path}")
                    
                    # Загружаем изображение
                    image = cv2.imread(img_path)
                    
                    # Обнаруживаем элементы
                    detections = detector.detect(image)
                    print(f"Обнаружено элементов: {len(detections)}")
                    
                    if len(detections) > 0:
                        # Визуализируем результаты
                        vis_image = detector.visualize_detections(image, detections)
                        
                        # Сохраняем результат
                        os.makedirs("results", exist_ok=True)
                        result_path = f"results/yolo_detection_{os.path.basename(img_path)}"
                        cv2.imwrite(result_path, vis_image)
                        print(f"Результат сохранен в файл {result_path}")
            
            break  # Если модель успешно загружена, прекращаем перебор
            
        except ImportError:
            print("Не удалось импортировать ultralytics. Установите библиотеку: pip install ultralytics")
            break
            
        except Exception as e:
            print(f"Ошибка при работе с моделью {model_path}: {e}")
    
    if not model_loaded:
        print("Не удалось загрузить ни одну модель YOLOv8.")
        print("Убедитесь, что установлена библиотека ultralytics и доступны модели.") 