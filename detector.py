import cv2
import numpy as np
import onnxruntime as ort

class Detector:
    """
    Класс для детекции элементов интерфейса с использованием ONNX-модели
    """
    def __init__(self, model_path):
        """
        Инициализация детектора
        
        Args:
            model_path (str): Путь к файлу модели ONNX
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def preprocess_image(self, image):
        """
        Предобработка изображения для модели
        
        Args:
            image: Исходное изображение (numpy array)
            
        Returns:
            Предобработанное изображение
        """
        # Преобразование в нужный размер
        img_height, img_width = self.input_shape[2:] if len(self.input_shape) == 4 else (640, 640)
        img = cv2.resize(image, (img_width, img_height))
        
        # Нормализация и изменение формата
        img = img.transpose(2, 0, 1).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
        
    def predict(self, image):
        """
        Получение предсказаний модели
        
        Args:
            image: Исходное изображение (numpy array)
            
        Returns:
            Список обнаруженных элементов с координатами [x, y, width, height, confidence, class_id]
        """
        # Предобработка изображения
        processed_img = self.preprocess_image(image)
        
        # Получение предсказаний модели
        outputs = self.session.run(self.output_names, {self.input_name: processed_img})
        
        # Обработка выходных данных модели и получение координат
        return self._parse_outputs(outputs, image.shape[1], image.shape[0])
    
    def _parse_outputs(self, outputs, original_width, original_height):
        """
        Преобразование выходных данных модели в координаты объектов
        
        Args:
            outputs: Выходные данные модели
            original_width: Ширина исходного изображения
            original_height: Высота исходного изображения
            
        Returns:
            Список обнаруженных элементов с координатами
        """
        # Для YOLO формата (может отличаться в зависимости от конкретной модели)
        # Предполагаем, что выход имеет формат [batch, num_detections, 5+num_classes]
        detections = []
        
        # Предположим, что outputs[0] содержит наши предсказания
        predictions = outputs[0]
        
        # Фильтрация по уверенности (confidence threshold)
        confidence_threshold = 0.5
        
        for pred in predictions:
            # Для моделей YOLO v5/v8
            if len(pred.shape) == 2:
                for detection in pred:
                    # Получаем координаты, уверенность и класс
                    *xyxy, confidence, class_id = detection
                    
                    if confidence >= confidence_threshold:
                        # Преобразуем формат [x1, y1, x2, y2] в [x, y, width, height]
                        x1, y1, x2, y2 = xyxy
                        
                        # Пересчитываем координаты в исходный размер изображения
                        x1 = int(x1 * original_width / self.input_shape[3])
                        y1 = int(y1 * original_height / self.input_shape[2])
                        x2 = int(x2 * original_width / self.input_shape[3])
                        y2 = int(y2 * original_height / self.input_shape[2])
                        
                        width = x2 - x1
                        height = y2 - y1
                        
                        detections.append({
                            'x': x1,
                            'y': y1,
                            'width': width,
                            'height': height,
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        })
        
        return detections
    
    def detect_elements(self, image, class_names=None):
        """
        Определение элементов интерфейса на изображении
        
        Args:
            image: Исходное изображение
            class_names: Список имен классов
            
        Returns:
            Список элементов интерфейса с их координатами и метками
        """
        elements = self.predict(image)
        
        # Если предоставлены имена классов, добавляем их в результат
        if class_names and len(class_names) > 0:
            for element in elements:
                class_id = element['class_id']
                element['label'] = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        return elements 