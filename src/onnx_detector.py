"""
Модуль для распознавания элементов пользовательского интерфейса с использованием ONNX-моделей
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import cv2
import onnxruntime
from .ui_detector import UIDetector

# Настройка логирования
logger = logging.getLogger(__name__)

class OnnxDetector(UIDetector):
    """
    Детектор UI элементов, использующий модели ONNX
    
    Поддерживает различные архитектуры нейронных сетей, 
    экспортированные в формат ONNX (YOLO, SSD, Faster R-CNN и т.д.)
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        confidence_threshold: float = 0.5,
        input_width: int = 640,
        input_height: int = 640,
        providers: List[str] = None,
        **kwargs
    ):
        """
        Инициализация ONNX-детектора UI элементов
        
        Args:
            model_path (str): Путь к файлу модели ONNX
            class_names (List[str]): Список имен классов UI элементов
            confidence_threshold (float): Порог уверенности для фильтрации обнаружений
            input_width (int): Ширина входного изображения для модели
            input_height (int): Высота входного изображения для модели
            providers (List[str]): Список провайдеров для ONNX Runtime (если None, 
                                  используются провайдеры по умолчанию)
            **kwargs: Дополнительные параметры
        """
        super().__init__(
            model_path=model_path,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
        
        self.input_width = input_width
        self.input_height = input_height
        
        # Провайдеры для ONNX Runtime
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Загружаем модель при инициализации
        self._load_model()
        
    def _load_model(self) -> bool:
        """
        Загрузка ONNX модели
        
        Returns:
            bool: True, если модель успешно загружена
        """
        if not self._ensure_model_loaded():
            return False
        
        try:
            # Создаем сессию ONNX Runtime
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=self.get_available_providers()
            )
            
            # Получаем информацию о входах и выходах модели
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Запоминаем размеры входа
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4:  # NCHW или NHWC формат
                self.input_height = input_shape[2] if input_shape[2] != -1 else self.input_height
                self.input_width = input_shape[3] if input_shape[3] != -1 else self.input_width
            
            logger.info(f"Модель ONNX успешно загружена из {self.model_path}")
            logger.info(f"Размер входа: {self.input_width}x{self.input_height}")
            logger.info(f"Используемые провайдеры: {self.session.get_providers()}")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели ONNX: {e}")
            return False
    
    def get_available_providers(self) -> List[str]:
        """
        Получает список доступных провайдеров ONNX Runtime
        
        Returns:
            List[str]: Список доступных провайдеров
        """
        available_providers = onnxruntime.get_available_providers()
        
        # Фильтруем запрошенные провайдеры, оставляя только доступные
        filtered_providers = [p for p in self.providers if p in available_providers]
        
        # Если нет доступных провайдеров из запрошенных, используем все доступные
        if not filtered_providers:
            logger.warning(f"Запрошенные провайдеры {self.providers} недоступны. "
                         f"Используем доступные: {available_providers}")
            return available_providers
        
        return filtered_providers
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения перед подачей в модель
        
        Args:
            image (np.ndarray): Исходное изображение (BGR)
            
        Returns:
            np.ndarray: Предобработанное изображение в формате, подходящем для модели
        """
        # Изменяем размер изображения до входного размера модели
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Конвертируем BGR в RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Нормализуем изображение
        normalized = rgb.astype(np.float32) / 255.0
        
        # Меняем размерность для соответствия формату NCHW (batch, channels, height, width)
        # Некоторые модели могут требовать NHWC (batch, height, width, channels)
        # Зависит от конкретной модели
        input_tensor = normalized.transpose(2, 0, 1)  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # CHW -> NCHW
        
        return input_tensor
    
    def _ensure_model_loaded(self) -> bool:
        """
        Проверяет, загружена ли модель, и загружает её, если необходимо
        
        Returns:
            bool: True, если модель успешно загружена или уже была загружена
        """
        if hasattr(self, 'session') and self.session is not None:
            return True
        
        # Проверка пути к модели
        if self.model_path is None:
            logger.error("Путь к модели ONNX не задан")
            return False
        
        # Проверка существования файла модели
        if not os.path.exists(self.model_path):
            logger.error(f"Файл модели ONNX не найден: {self.model_path}")
            return False
        
        return True  # Разрешаем загрузку
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Обнаружение UI элементов на изображении с помощью ONNX модели
        
        Args:
            image (np.ndarray): Исходное изображение (BGR)
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных элементов с атрибутами
        """
        if not self._ensure_model_loaded():
            logger.error("Модель ONNX не загружена, обнаружение невозможно")
            return []
        
        # Получаем размеры исходного изображения для масштабирования результатов
        original_height, original_width = image.shape[:2]
        
        # Предобработка изображения
        input_tensor = self.preprocess_image(image)
        
        try:
            # Выполняем предсказание
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            
            # Обрабатываем результаты в зависимости от формата выходных данных модели
            # Здесь предполагается формат YOLO (для других моделей может потребоваться другая логика)
            detections = self._process_yolo_output(
                outputs, 
                original_width, 
                original_height
            )
            
            # Отфильтровываем по порогу уверенности
            filtered_detections = [
                det for det in detections 
                if det.get('confidence', 0) >= self.confidence_threshold
            ]
            
            # Добавляем текст, если возможно
            for detection in filtered_detections:
                x1, y1, x2, y2 = (
                    detection['x1'], detection['y1'], 
                    detection['x2'], detection['y2']
                )
                detection['text'] = self._extract_text(image, (x1, y1, x2, y2))
            
            logger.info(f"Обнаружено {len(filtered_detections)} UI элементов")
            return filtered_detections
        
        except Exception as e:
            logger.error(f"Ошибка при выполнении обнаружения: {e}")
            return []
    
    def _process_yolo_output(
        self, 
        outputs: List[np.ndarray], 
        original_width: int, 
        original_height: int
    ) -> List[Dict[str, Any]]:
        """
        Обработка выходных данных модели YOLO
        
        Args:
            outputs (List[np.ndarray]): Выходные данные модели
            original_width (int): Ширина исходного изображения
            original_height (int): Высота исходного изображения
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных элементов
        """
        # В зависимости от версии YOLO, формат выходных данных может различаться
        # Здесь предполагается формат YOLOv5/v8
        
        # Обрабатываем выходные данные (пример для YOLOv8)
        # У YOLOv8 один выходной тензор формата [batch, num_boxes, 85]
        # 85 = 4 (box) + 1 (conf) + 80 (class scores для COCO)
        # Для UI-элементов количество классов может быть другим
        
        detections = []
        
        try:
            # Первый выходной тензор содержит предсказания
            predictions = outputs[0]
            
            # Определяем количество классов
            # Может быть разным в зависимости от обучения модели
            box_data_length = 4  # x, y, width, height
            num_classes = predictions.shape[2] - box_data_length - 1  # -1 для conf
            
            # Проходим по всем предсказаниям
            for i in range(predictions.shape[1]):
                # Получаем данные бокса
                box_data = predictions[0, i, :box_data_length]
                confidence = predictions[0, i, box_data_length]
                
                # Если уверенность ниже порога, пропускаем
                if confidence < self.confidence_threshold:
                    continue
                
                # Получаем вероятности классов
                class_scores = predictions[0, i, box_data_length+1:box_data_length+1+num_classes]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                
                # Если вероятность класса слишком мала, пропускаем
                if class_score < self.confidence_threshold:
                    continue
                
                # Комбинируем уверенность детекции и класса
                combined_confidence = confidence * class_score
                
                # Если комбинированная уверенность ниже порога, пропускаем
                if combined_confidence < self.confidence_threshold:
                    continue
                
                # Преобразуем данные бокса в формат (x1, y1, x2, y2)
                # YOLO выдает (centerX, centerY, width, height) относительно размера входа
                centerX, centerY, width, height = box_data
                
                # Масштабируем координаты к размеру исходного изображения
                x1 = int((centerX - width/2) * original_width)
                y1 = int((centerY - height/2) * original_height)
                x2 = int((centerX + width/2) * original_width)
                y2 = int((centerY + height/2) * original_height)
                
                # Ограничиваем координаты границами изображения
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                x2 = max(0, min(x2, original_width - 1))
                y2 = max(0, min(y2, original_height - 1))
                
                # Получаем имя класса
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Добавляем обнаружение в список
                detections.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': float(combined_confidence),
                    'class_id': int(class_id),
                    'class': class_name
                })
            
            # Применяем NMS для удаления дублирующихся обнаружений
            # Сначала подготовим данные для NMS
            boxes = [(d['x1'], d['y1'], d['x2'], d['y2']) for d in detections]
            scores = [d['confidence'] for d in detections]
            
            # Получаем индексы оставшихся после NMS боксов
            keep_indices = self._apply_nms(boxes, scores)
            
            # Оставляем только нужные детекции
            filtered_detections = [detections[i] for i in keep_indices]
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Ошибка при обработке выходных данных YOLO: {e}")
            return []


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Создаем экземпляр детектора
    model_path = "models/ui_detector.onnx"
    
    if os.path.exists(model_path):
        detector = OnnxDetector(model_path=model_path)
        
        # Загружаем тестовое изображение
        image_path = "test/test_ui_screenshot.png"
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            
            # Обнаруживаем элементы
            detections = detector.detect(image)
            print(f"Обнаружено элементов: {len(detections)}")
            
            # Визуализируем результаты
            vis_image = detector.visualize_detections(image, detections)
            
            # Сохраняем результат
            os.makedirs("results", exist_ok=True)
            cv2.imwrite("results/test_detections.png", vis_image)
            print("Результат сохранен в файл results/test_detections.png")
        else:
            print(f"Тестовое изображение не найдено: {image_path}")
    else:
        print(f"Файл модели не найден: {model_path}")
        print("Пожалуйста, скачайте или обучите модель перед использованием.") 