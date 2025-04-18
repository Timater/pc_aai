import os
import cv2
import numpy as np
import onnxruntime

class Detector:
    """
    Класс для детектирования UI элементов на экране с использованием модели YOLO
    """
    def __init__(self, model_path):
        """
        Инициализация детектора
        
        Args:
            model_path (str): Путь к ONNX модели
        """
        # Проверяем существование файла модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        # Загружаем модель ONNX
        try:
            self.session = onnxruntime.InferenceSession(model_path)
            
            # Получаем информацию о входном слое
            model_inputs = self.session.get_inputs()
            self.input_name = model_inputs[0].name
            
            # Получаем размерности входного изображения
            input_shape = model_inputs[0].shape
            
            if len(input_shape) == 4:  # NCHW format
                self.input_width = input_shape[3]
                self.input_height = input_shape[2]
            else:
                # Используем стандартный размер для YOLO
                self.input_width = 640
                self.input_height = 640
                
            # Выходные слои
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Порог уверенности для детекции
            self.confidence_threshold = 0.5
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}")
    
    def preprocess_image(self, image):
        """
        Предобработка изображения для подачи в модель
        
        Args:
            image (numpy.ndarray): Исходное изображение BGR
            
        Returns:
            numpy.ndarray: Обработанное изображение
        """
        # Изменяем размер изображения
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Нормализация от 0 до 1
        normalized = resized.astype(np.float32) / 255.0
        
        # Преобразование из BGR в RGB, если необходимо
        if resized.shape[2] == 3:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        
        # Изменение формата для модели (NCHW)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        
        # Добавление размерности батча
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def post_process(self, outputs, original_image, class_names=None):
        """
        Постобработка выходных данных модели
        
        Args:
            outputs (list): Выходные данные модели
            original_image (numpy.ndarray): Исходное изображение
            class_names (list): Список имен классов
            
        Returns:
            list: Список обнаруженных объектов с координатами и классами
        """
        # Получаем размеры исходного изображения
        original_height, original_width = original_image.shape[:2]
        
        # Коэффициенты масштабирования
        scale_width = original_width / self.input_width
        scale_height = original_height / self.input_height
        
        # Извлекаем данные из выходного тензора
        # Формат зависит от конкретной модели YOLO
        detected_objects = []
        
        if len(outputs) == 1:
            # Стандартный формат выходных данных YOLO v8 - одна матрица (batch, objects, params)
            predictions = outputs[0]
            
            # Перебираем результаты для каждого объекта
            for detection in predictions[0]:
                # Извлекаем class_id и confidence
                if len(detection) >= 6:  # box (4) + confidence (1) + classes (n)
                    # Находим индекс класса с максимальной вероятностью
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = float(scores[class_id])
                    
                    # Фильтруем по уверенности
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Извлекаем координаты
                    x, y, w, h = detection[0:4]
                    
                    # Преобразуем в формат [x1, y1, x2, y2]
                    x1 = int((x - w/2) * scale_width)
                    y1 = int((y - h/2) * scale_height)
                    x2 = int((x + w/2) * scale_width)
                    y2 = int((y + h/2) * scale_height)
                    
                    # Ограничиваем координаты границами изображения
                    x1 = max(0, min(x1, original_width))
                    y1 = max(0, min(y1, original_height))
                    x2 = max(0, min(x2, original_width))
                    y2 = max(0, min(y2, original_height))
                    
                    # Получаем имя класса
                    class_name = f"class_{class_id}"
                    if class_names and class_id < len(class_names):
                        class_name = class_names[class_id]
                    
                    detected_objects.append({
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "box": [x1, y1, x2, y2],
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "center_x": (x1 + x2) // 2,
                        "center_y": (y1 + y2) // 2
                    })
        
        return detected_objects
    
    def detect_elements(self, image, class_names=None):
        """
        Обнаружение элементов интерфейса на изображении
        
        Args:
            image (numpy.ndarray): Исходное изображение BGR
            class_names (list): Список имен классов
            
        Returns:
            list: Список обнаруженных элементов
        """
        try:
            # Предобработка изображения
            preprocessed_image = self.preprocess_image(image)
            
            # Запуск инференса
            outputs = self.session.run(self.output_names, {self.input_name: preprocessed_image})
            
            # Постобработка результатов
            detections = self.post_process(outputs, image, class_names)
            
            return detections
        except Exception as e:
            print(f"Ошибка при детектировании: {str(e)}")
            return [] 