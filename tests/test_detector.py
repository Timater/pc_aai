"""
Тесты для модуля детектора интерфейса
"""

import os
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Импортируем тестируемый модуль
from src.detector import Detector
from tests import MockDetector

class TestDetector:
    """
    Набор тестов для проверки модуля Detector
    """
    
    @pytest.fixture
    def sample_image(self):
        """
        Создает тестовое изображение для тестирования детектора
        """
        # Создаем пустое изображение 640x480 со случайным заполнением
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Добавляем прямоугольники разных цветов для имитации UI элементов
        # Кнопка (красный)
        cv2.rectangle(image, (50, 50), (150, 100), (0, 0, 255), -1)
        # Текстовое поле (зеленый)
        cv2.rectangle(image, (200, 150), (400, 200), (0, 255, 0), -1)
        # Чекбокс (синий)
        cv2.rectangle(image, (50, 200), (80, 230), (255, 0, 0), -1)
        
        return image
    
    @pytest.fixture
    def mock_session(self):
        """
        Создает мок для сессии ONNX Runtime
        """
        mock_session = MagicMock()
        
        # Настраиваем входные данные
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_input.shape = [1, 3, 640, 640]  # NCHW формат
        mock_session.get_inputs.return_value = [mock_input]
        
        # Настраиваем выходные данные
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_outputs.return_value = [mock_output]
        
        # Настраиваем результат инференса
        # Имитируем формат: 1 пакет, 3 обнаружения, 6 значений (x, y, w, h, confidence, class)
        # плюс confidence для каждого класса (предположим, есть 11 классов)
        # [batch, objects, params]
        mock_result = [np.array([[[
            [100, 75, 100, 50, 0.95, 0, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # кнопка
            [300, 175, 200, 50, 0.87, 1, 0.01, 0.87, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # текстовое поле
            [65, 215, 30, 30, 0.78, 2, 0.01, 0.01, 0.78, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]    # чекбокс
        ]])]
        
        mock_session.run.return_value = mock_result
        
        return mock_session
    
    def test_detector_initialization_file_not_found(self):
        """
        Тест на инициализацию детектора с несуществующим файлом модели
        """
        # Используем заведомо несуществующий путь к файлу
        non_existent_path = "non_existent_model.onnx"
        
        # Проверяем, что инициализация с несуществующим файлом вызывает исключение
        with pytest.raises(FileNotFoundError):
            Detector(non_existent_path)
    
    @patch('onnxruntime.InferenceSession')
    @patch('os.path.exists')
    def test_detector_initialization_success(self, mock_exists, mock_inference_session):
        """
        Тест на успешную инициализацию детектора
        """
        # Настраиваем мок для проверки существования файла
        mock_exists.return_value = True
        
        # Настраиваем мок для сессии
        mock_session = self.mock_session()
        mock_inference_session.return_value = mock_session
        
        # Инициализируем детектор
        detector = Detector("fake_model.onnx")
        
        # Проверяем, что атрибуты инициализированы правильно
        assert detector.input_name == "images"
        assert detector.input_width == 640
        assert detector.input_height == 640
        assert detector.confidence_threshold == 0.5
        assert detector.session == mock_session
    
    def test_preprocess_image(self, sample_image):
        """
        Тест на предобработку изображения
        """
        # Инициализируем детектор с использованием патча
        with patch('os.path.exists') as mock_exists, patch('onnxruntime.InferenceSession') as mock_inference_session:
            mock_exists.return_value = True
            mock_inference_session.return_value = self.mock_session()
            
            detector = Detector("fake_model.onnx")
        
        # Выполняем предобработку изображения
        preprocessed = detector.preprocess_image(sample_image)
        
        # Проверяем размерность и тип выходных данных
        assert preprocessed.shape == (1, 3, 640, 640)  # NCHW формат с размером пакета
        assert preprocessed.dtype == np.float32
        assert np.max(preprocessed) <= 1.0  # Проверяем нормализацию
        assert np.min(preprocessed) >= 0.0
    
    def test_post_process(self, sample_image):
        """
        Тест на постобработку результатов инференса
        """
        # Инициализируем детектор с использованием патча
        with patch('os.path.exists') as mock_exists, patch('onnxruntime.InferenceSession') as mock_inference_session:
            mock_exists.return_value = True
            mock_inference_session.return_value = self.mock_session()
            
            detector = Detector("fake_model.onnx")
        
        # Создаем тестовые выходные данные модели
        # Имитируем формат: batch, objects, params
        test_outputs = [np.array([[[
            100, 75, 100, 50, 0.95, 0, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01  # кнопка
        ]]])]
        
        # Список имен классов
        class_names = ["button", "text_field", "checkbox", "radio", "dropdown", 
                      "menu", "icon", "tab", "scrollbar", "window", "dialog"]
        
        # Выполняем постобработку
        detections = detector.post_process(test_outputs, sample_image, class_names)
        
        # Проверяем результаты
        assert len(detections) == 1
        detection = detections[0]
        
        assert detection["class_id"] == 0
        assert detection["class_name"] == "button"
        assert detection["confidence"] == pytest.approx(0.95, 0.01)
        assert "box" in detection
        assert "width" in detection
        assert "height" in detection
        assert "center_x" in detection
        assert "center_y" in detection
    
    def test_detect_elements(self, sample_image):
        """
        Тест на полную работу метода обнаружения элементов
        """
        # Инициализируем детектор с использованием патча
        with patch('os.path.exists') as mock_exists, patch('onnxruntime.InferenceSession') as mock_inference_session:
            mock_exists.return_value = True
            mock_session = self.mock_session()
            mock_inference_session.return_value = mock_session
            
            detector = Detector("fake_model.onnx")
        
        # Список имен классов
        class_names = ["button", "text_field", "checkbox", "radio", "dropdown", 
                      "menu", "icon", "tab", "scrollbar", "window", "dialog"]
        
        # Выполняем обнаружение элементов
        elements = detector.detect_elements(sample_image, class_names)
        
        # Проверяем результаты
        assert len(elements) == 3  # Три элемента, как настроено в mock_session
        
        # Проверяем первый элемент
        assert elements[0]["class_id"] == 0
        assert elements[0]["class_name"] == "button"
        assert elements[0]["confidence"] > 0.9
        
        # Проверяем второй элемент
        assert elements[1]["class_id"] == 1
        assert elements[1]["class_name"] == "text_field"
        assert elements[1]["confidence"] > 0.8
        
        # Проверяем третий элемент
        assert elements[2]["class_id"] == 2
        assert elements[2]["class_name"] == "checkbox"
        assert elements[2]["confidence"] > 0.7
    
    def test_mock_detector(self, sample_image):
        """
        Тест на работу MockDetector
        """
        # Инициализируем мок-детектор
        mock_detector = MockDetector()
        
        # Список имен классов
        class_names = ["button", "text_field", "checkbox", "radio", "dropdown", 
                      "menu", "icon", "tab", "scrollbar", "window", "dialog"]
        
        # Выполняем обнаружение элементов
        elements = mock_detector.detect_elements(sample_image, class_names)
        
        # Проверяем результаты
        assert len(elements) >= 3  # MockDetector генерирует от 3 до 7 элементов
        assert len(elements) <= 7
        
        # Проверяем структуру элементов
        for element in elements:
            assert "class_id" in element
            assert "class_name" in element
            assert "confidence" in element
            assert "box" in element
            assert "width" in element
            assert "height" in element
            assert "center_x" in element
            assert "center_y" in element
            
            # Проверяем типы данных
            assert isinstance(element["class_id"], int)
            assert isinstance(element["class_name"], str)
            assert isinstance(element["confidence"], float)
            assert isinstance(element["box"], list)
            assert len(element["box"]) == 4  # x1, y1, x2, y2
            
            # Проверяем диапазоны
            assert 0 <= element["class_id"] < len(class_names)
            assert element["class_name"] in class_names
            assert 0.5 <= element["confidence"] <= 1.0 