# Пакет тестов для системы управления ПК 

import random

class MockDetector:
    """
    Мок-детектор для тестирования без реальной модели.
    Возвращает случайные данные для имитации работы детектора.
    """
    def __init__(self):
        """
        Инициализация мок-детектора
        """
        self.confidence_threshold = 0.5

    def detect_elements(self, image, class_names=None):
        """
        Имитация обнаружения элементов интерфейса на изображении
        
        Args:
            image (numpy.ndarray): Исходное изображение BGR
            class_names (list): Список имен классов
            
        Returns:
            list: Список сгенерированных элементов
        """
        height, width = image.shape[:2]
        
        # Используем список классов, если предоставлен, иначе используем значения по умолчанию
        if class_names is None:
            class_names = ["button", "text_field", "checkbox", "radio", "dropdown", 
                          "menu", "icon", "tab", "scrollbar", "window", "dialog"]
        
        # Генерируем от 3 до 7 случайных элементов
        num_elements = random.randint(3, 7)
        detected_objects = []
        
        for _ in range(num_elements):
            # Случайный класс
            class_id = random.randint(0, len(class_names) - 1)
            class_name = class_names[class_id]
            
            # Случайные координаты
            w = random.randint(50, 300)
            h = random.randint(30, 150)
            x1 = random.randint(0, max(1, width - w))
            y1 = random.randint(0, max(1, height - h))
            x2 = x1 + w
            y2 = y1 + h
            
            # Случайная уверенность
            confidence = random.uniform(0.6, 0.95)
            
            detected_objects.append({
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(confidence),
                "box": [x1, y1, x2, y2],
                "width": w,
                "height": h,
                "center_x": (x1 + x2) // 2,
                "center_y": (y1 + y2) // 2
            })
        
        return detected_objects 