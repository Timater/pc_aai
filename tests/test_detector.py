import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import Detector

class TestDetector(unittest.TestCase):
    """
    Тесты для класса Detector
    """
    @patch('onnxruntime.InferenceSession')
    def test_initialize_detector(self, mock_session):
        """
        Тест инициализации детектора
        """
        # Настройка мока
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 640, 640]
        mock_output = MagicMock()
        mock_output.name = "output"
        
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.get_outputs.return_value = [mock_output]
        mock_session.return_value = mock_session_instance
        
        # Вызов тестируемого метода
        detector = Detector("dummy_model.onnx")
        
        # Проверки
        self.assertEqual(detector.input_name, "input")
        self.assertEqual(detector.input_shape, [1, 3, 640, 640])
        self.assertEqual(detector.output_names, ["output"])
        mock_session.assert_called_once_with("dummy_model.onnx")
    
    @patch('onnxruntime.InferenceSession')
    def test_preprocess_image(self, mock_session):
        """
        Тест предобработки изображения
        """
        # Настройка мока
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 640, 640]
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.get_outputs.return_value = [MagicMock()]
        mock_session.return_value = mock_session_instance
        
        # Создание детектора
        detector = Detector("dummy_model.onnx")
        
        # Создание тестового изображения (RGB)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Вызов тестируемого метода
        processed_image = detector.preprocess_image(test_image)
        
        # Проверки
        self.assertEqual(processed_image.shape, (1, 3, 640, 640))  # Батч, каналы, высота, ширина
        self.assertEqual(processed_image.dtype, np.float32)  # Тип данных float32
        self.assertTrue(np.all(processed_image >= 0.0) and np.all(processed_image <= 1.0))  # Нормализация [0,1]
    
    @patch('onnxruntime.InferenceSession')
    def test_parse_outputs(self, mock_session):
        """
        Тест обработки выходов модели
        """
        # Настройка мока
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 640, 640]
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.get_outputs.return_value = [MagicMock()]
        mock_session.return_value = mock_session_instance
        
        # Создание детектора
        detector = Detector("dummy_model.onnx")
        
        # Создание примера выходных данных YOLO (1 батч, 3 детекции, формат [x1, y1, x2, y2, conf, class_id])
        mock_output = np.array([
            [  # batch
                [  # detections
                    [100, 100, 200, 200, 0.9, 0],  # bbox1: [x1, y1, x2, y2, confidence, class_id]
                    [300, 300, 400, 400, 0.8, 1],  # bbox2
                    [500, 100, 600, 200, 0.7, 2],  # bbox3
                ]
            ]
        ])
        
        # Вызов тестируемого метода
        original_width, original_height = 1920, 1080
        detections = detector._parse_outputs([mock_output], original_width, original_height)
        
        # Проверки
        self.assertEqual(len(detections), 3)  # Должно быть 3 детекции
        
        # Проверяем первую детекцию
        self.assertEqual(detections[0]['x'], 100 * original_width // 640)
        self.assertEqual(detections[0]['y'], 100 * original_height // 640)
        self.assertEqual(detections[0]['width'], 100 * original_width // 640)  # 200-100=100
        self.assertEqual(detections[0]['height'], 100 * original_height // 640)  # 200-100=100
        self.assertEqual(detections[0]['confidence'], 0.9)
        self.assertEqual(detections[0]['class_id'], 0)
    
    @patch('onnxruntime.InferenceSession')
    def test_detect_elements_with_class_names(self, mock_session):
        """
        Тест распознавания элементов с именами классов
        """
        # Настройка мока
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 640, 640]
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.get_outputs.return_value = [MagicMock()]
        mock_session.return_value = mock_session_instance
        
        # Создание детектора
        detector = Detector("dummy_model.onnx")
        
        # Мокаем метод predict, чтобы вернуть предопределенные объекты
        detector.predict = MagicMock(return_value=[
            {'x': 100, 'y': 100, 'width': 100, 'height': 50, 'confidence': 0.9, 'class_id': 0},
            {'x': 300, 'y': 200, 'width': 80, 'height': 30, 'confidence': 0.8, 'class_id': 1}
        ])
        
        # Создание тестового изображения
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Вызов тестируемого метода с классами
        class_names = ["button", "text_field"]
        elements = detector.detect_elements(test_image, class_names)
        
        # Проверки
        self.assertEqual(len(elements), 2)
        self.assertEqual(elements[0]['label'], "button")
        self.assertEqual(elements[1]['label'], "text_field")

if __name__ == "__main__":
    unittest.main() 