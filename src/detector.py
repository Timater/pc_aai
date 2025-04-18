"""
Модуль детектора интерфейсных элементов на основе ONNX модели
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)

class Detector:
    """
    Класс для детекции элементов интерфейса с использованием ONNX модели
    """
    
    def __init__(self, model_path, confidence_threshold=0.5, input_size=(640, 640)):
        """
        Инициализация детектора

        Args:
            model_path (str): Путь к ONNX модели
            confidence_threshold (float): Порог уверенности для детекций
            input_size (tuple): Размер входного изображения для модели (ширина, высота)
        """
        if not os.path.exists(model_path):
            logger.error(f"Модель не найдена по пути {model_path}")
            raise FileNotFoundError(f"Модель не найдена по пути {model_path}")
        
        logger.info(f"Инициализация детектора с моделью {model_path}")
        
        # Инициализация параметров
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Классы UI элементов
        self.class_names = [
            "button", "text_field", "checkbox", "radio", "dropdown", 
            "menu", "icon", "tab", "scrollbar", "window", "dialog"
        ]
        
        # Создание сессии ONNX
        try:
            self.session = ort.InferenceSession(
                model_path, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info("Сессия ONNX успешно инициализирована")
        except Exception as e:
            logger.error(f"Ошибка при инициализации сессии ONNX: {e}")
            raise RuntimeError(f"Ошибка при инициализации сессии ONNX: {e}")
    
    def preprocess_image(self, image):
        """
        Предобработка изображения для модели

        Args:
            image (np.ndarray): Исходное изображение (BGR)

        Returns:
            tuple: (предобработанное изображение, информация для постобработки)
        """
        # Сохраняем оригинальные размеры для постобработки
        orig_height, orig_width = image.shape[:2]
        
        # Изменяем размер изображения для входа модели
        resized_image = cv2.resize(image, self.input_size)
        
        # Нормализация [0, 255] -> [0, 1]
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Перестановка каналов BGR -> RGB и изменение формата HWC -> NCHW
        processed_image = np.transpose(normalized_image, (2, 0, 1))
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Информация для постобработки
        preprocess_info = {
            "orig_height": orig_height,
            "orig_width": orig_width,
            "input_height": self.input_size[1],
            "input_width": self.input_size[0]
        }
        
        return processed_image, preprocess_info
    
    def postprocess_results(self, outputs, preprocess_info):
        """
        Постобработка результатов модели

        Args:
            outputs (list): Выходные данные модели
            preprocess_info (dict): Информация о предобработке

        Returns:
            list: Список обнаруженных элементов с координатами и классами
        """
        # Распаковываем выходные данные YOLO
        # В зависимости от экспорта модели структура вывода может отличаться
        if isinstance(outputs, dict):
            # Если вывод - словарь (некоторые экспортированные модели)
            output = list(outputs.values())[0]
        else:
            # Если вывод - список (стандартный формат YOLO)
            output = outputs[0]
        
        # Размеры изображения
        orig_height = preprocess_info["orig_height"]
        orig_width = preprocess_info["orig_width"]
        input_height = preprocess_info["input_height"]
        input_width = preprocess_info["input_width"]
        
        # Коэффициенты масштабирования
        scale_x = orig_width / input_width
        scale_y = orig_height / input_height
        
        # Список детекций
        detections = []
        
        # Формат вывода YOLO:
        # [x1, y1, x2, y2, confidence, class_id1, class_id2, ...]
        num_boxes = output.shape[1]
        
        for i in range(num_boxes):
            box = output[0, i, :]
            
            # Проверяем, что у нас есть как минимум 6 значений
            # (x1, y1, x2, y2, confidence, class_id)
            if len(box) < 6:
                continue
            
            confidence = box[4]
            
            # Пропускаем детекции с низкой уверенностью
            if confidence < self.confidence_threshold:
                continue
            
            # Получаем координаты
            x1, y1, x2, y2 = box[0:4]
            
            # Масштабируем обратно к исходному размеру
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Получаем ID класса с максимальной вероятностью
            class_scores = box[5:]
            class_id = np.argmax(class_scores)
            
            # Получаем имя класса
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"class_{class_id}"
            
            # Формируем запись о детекции
            detection = {
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(confidence),
                "box": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1,
                "center_x": int((x1 + x2) / 2),
                "center_y": int((y1 + y2) / 2)
            }
            
            detections.append(detection)
        
        return detections
    
    def detect(self, image):
        """
        Детекция элементов интерфейса на изображении

        Args:
            image (np.ndarray или str): Входное изображение или путь к файлу

        Returns:
            list: Список обнаруженных элементов
        """
        # Проверка типа входных данных
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Файл изображения не найден: {image}")
            image = cv2.imread(image)
        
        if image is None or image.size == 0:
            raise ValueError("Неверный формат изображения")
        
        # Предобработка
        processed_image, preprocess_info = self.preprocess_image(image)
        
        # Запуск инференса
        try:
            outputs = self.session.run(None, {self.input_name: processed_image})
            
            # Постобработка
            detections = self.postprocess_results(outputs, preprocess_info)
            
            logger.info(f"Обнаружено {len(detections)} элементов интерфейса")
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении инференса: {e}")
            raise RuntimeError(f"Ошибка при выполнении инференса: {e}")
    
    def find_element_by_name(self, image, element_name, partial_match=True):
        """
        Поиск элемента по имени класса

        Args:
            image (np.ndarray или str): Входное изображение или путь к файлу
            element_name (str): Имя элемента для поиска
            partial_match (bool): Искать частичные совпадения

        Returns:
            dict: Информация о найденном элементе или None
        """
        # Получаем все детекции
        detections = self.detect(image)
        
        if not detections:
            logger.warning(f"Элементы не обнаружены при поиске '{element_name}'")
            return None
        
        # Поиск элемента по имени класса
        matching_elements = []
        
        for detection in detections:
            class_name = detection["class_name"]
            
            # Проверка на совпадение
            if partial_match:
                if element_name.lower() in class_name.lower():
                    matching_elements.append(detection)
            else:
                if element_name.lower() == class_name.lower():
                    matching_elements.append(detection)
        
        # Если найдены совпадения, возвращаем элемент с наивысшей уверенностью
        if matching_elements:
            best_match = max(matching_elements, key=lambda x: x["confidence"])
            logger.info(f"Найден элемент '{best_match['class_name']}' по запросу '{element_name}'")
            return best_match
        
        logger.warning(f"Элемент '{element_name}' не найден среди детекций")
        return None
    
    def find_element_by_coordinates(self, image, x, y):
        """
        Поиск элемента по координатам точки

        Args:
            image (np.ndarray или str): Входное изображение или путь к файлу
            x (int): Координата X
            y (int): Координата Y

        Returns:
            dict: Информация о найденном элементе или None
        """
        # Получаем все детекции
        detections = self.detect(image)
        
        if not detections:
            logger.warning(f"Элементы не обнаружены при поиске по координатам ({x}, {y})")
            return None
        
        # Поиск элемента, содержащего точку (x, y)
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                logger.info(f"Найден элемент '{detection['class_name']}' по координатам ({x}, {y})")
                return detection
        
        logger.warning(f"Элемент по координатам ({x}, {y}) не найден")
        return None

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Инициализация детектора
    model_path = "models/detector.onnx"
    
    try:
        detector = Detector(model_path)
        
        # Тестовое изображение
        test_image_path = "tests/data/test_screenshot.png"
        
        if os.path.exists(test_image_path):
            # Загрузка тестового изображения
            image = cv2.imread(test_image_path)
            
            # Детекция элементов
            elements = detector.detect(image)
            
            # Вывод результатов
            print(f"Обнаружено элементов: {len(elements)}")
            for i, element in enumerate(elements):
                print(f"{i+1}. {element['class_name']} (conf: {element['confidence']:.2f}), "
                      f"координаты: {element['box']}")
            
            # Поиск конкретного элемента
            button = detector.find_element_by_name(image, "button")
            if button:
                print(f"Найдена кнопка: {button['box']}")
            
            # Визуализация результатов (опционально)
            for element in elements:
                x1, y1, x2, y2 = element["box"]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, element["class_name"], (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite("output.png", image)
            print("Результаты сохранены в output.png")
        else:
            print(f"Тестовое изображение не найдено: {test_image_path}")
    
    except Exception as e:
        print(f"Ошибка: {e}") 