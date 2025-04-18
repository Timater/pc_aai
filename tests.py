import unittest
import os
import time
from unittest.mock import patch, MagicMock

# Импортируем модули нашей системы
from detector import Detector
from actions import ActionManager
from nlp_processor import CommandProcessor
from file_manager import FileManager
from logger import Logger

class MockDetector(Detector):
    """
    Мок-класс для детектора элементов интерфейса
    """
    def __init__(self):
        self.mock_elements = []
    
    def set_mock_elements(self, elements):
        self.mock_elements = elements
    
    def detect_elements(self, image, class_names=None):
        return self.mock_elements

class MockActionManager(ActionManager):
    """
    Мок-класс для менеджера действий
    """
    def __init__(self):
        self.actions_log = []
        self.logger = MagicMock()
    
    def click(self, x, y, button='left', clicks=1):
        self.actions_log.append(('click', {'x': x, 'y': y, 'button': button, 'clicks': clicks}))
        return True
    
    def type_text(self, text, interval=0.1):
        self.actions_log.append(('type', {'text': text, 'interval': interval}))
        return True
    
    def press_key(self, key):
        self.actions_log.append(('press', {'key': key}))
        return True
        
    def get_actions_log(self):
        return self.actions_log

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
        mock_session.assert_called_once_with("dummy_model.onnx")

class TestCommandProcessor(unittest.TestCase):
    """
    Тесты для класса CommandProcessor
    """
    def setUp(self):
        self.processor = CommandProcessor()
    
    def test_parse_open_command(self):
        """
        Тест разбора команды открытия файла
        """
        command = "открой файл report.txt"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "open")
        self.assertEqual(result["params"].get("file_name"), "report.txt")
    
    def test_parse_search_command(self):
        """
        Тест разбора команды поиска
        """
        command = "найди папку проекты"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "search")
        self.assertEqual(result["params"].get("folder_name"), "проекты")
    
    def test_parse_type_command(self):
        """
        Тест разбора команды ввода текста
        """
        command = "напиши 'Привет, мир!'"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "type")
        self.assertEqual(result["params"].get("text"), "Привет, мир!")

class TestFileManager(unittest.TestCase):
    """
    Тесты для класса FileManager
    """
    def setUp(self):
        self.file_manager = FileManager()
        self.test_dir = "test_files"
        self.test_file = os.path.join(self.test_dir, "test.txt")
        
        # Создаем временную директорию и файл для тестов
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
    
    def tearDown(self):
        # Удаляем временные файлы после тестов
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
    
    def test_create_file(self):
        """
        Тест создания файла
        """
        content = "Test content"
        result = self.file_manager.create_file(self.test_file, content)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.test_file))
        
        with open(self.test_file, 'r') as f:
            file_content = f.read()
        self.assertEqual(file_content, content)
    
    def test_delete_file(self):
        """
        Тест удаления файла
        """
        # Создаем файл
        self.file_manager.create_file(self.test_file, "Test")
        
        # Проверяем, что файл существует
        self.assertTrue(os.path.exists(self.test_file))
        
        # Удаляем файл
        result = self.file_manager.delete_file(self.test_file)
        
        # Проверяем результат
        self.assertTrue(result)
        self.assertFalse(os.path.exists(self.test_file))

class TestIntegration(unittest.TestCase):
    """
    Интеграционные тесты системы
    """
    def setUp(self):
        # Инициализируем компоненты системы с моками
        self.detector = MockDetector()
        self.action_manager = MockActionManager()
        self.command_processor = CommandProcessor()
        self.file_manager = FileManager()
        
        # Добавляем тестовые элементы интерфейса
        self.mock_elements = [
            {'x': 100, 'y': 100, 'width': 50, 'height': 30, 'confidence': 0.9, 'class_id': 0, 'label': 'button'},
            {'x': 200, 'y': 150, 'width': 100, 'height': 20, 'confidence': 0.8, 'class_id': 1, 'label': 'text_field'}
        ]
        self.detector.set_mock_elements(self.mock_elements)
    
    def test_click_button_command(self):
        """
        Тест выполнения команды нажатия на кнопку
        """
        # Подготовка команды и ее разбор
        command = "нажми на кнопку"
        parsed_command = self.command_processor.parse_command(command)
        
        # Проверка, что команда распознана правильно
        self.assertEqual(parsed_command["action"], "click")
        
        # Находим элемент "кнопка" и кликаем
        button = next((elem for elem in self.mock_elements if elem['label'] == 'button'), None)
        self.assertIsNotNone(button)
        
        self.action_manager.click(button['x'] + button['width']//2, button['y'] + button['height']//2)
        
        # Проверяем журнал действий
        actions_log = self.action_manager.get_actions_log()
        self.assertEqual(len(actions_log), 1)
        self.assertEqual(actions_log[0][0], 'click')
    
    def test_type_text_command(self):
        """
        Тест выполнения команды ввода текста
        """
        # Подготовка команды и ее разбор
        command = "напиши 'Тестовый текст'"
        parsed_command = self.command_processor.parse_command(command)
        
        # Проверка, что команда распознана правильно
        self.assertEqual(parsed_command["action"], "type")
        self.assertEqual(parsed_command["params"]["text"], "Тестовый текст")
        
        # Находим элемент "текстовое поле" и кликаем на него перед вводом
        text_field = next((elem for elem in self.mock_elements if elem['label'] == 'text_field'), None)
        self.assertIsNotNone(text_field)
        
        self.action_manager.click(text_field['x'] + text_field['width']//2, text_field['y'] + text_field['height']//2)
        self.action_manager.type_text(parsed_command["params"]["text"])
        
        # Проверяем журнал действий
        actions_log = self.action_manager.get_actions_log()
        self.assertEqual(len(actions_log), 2)
        self.assertEqual(actions_log[0][0], 'click')
        self.assertEqual(actions_log[1][0], 'type')
        self.assertEqual(actions_log[1][1]['text'], "Тестовый текст")

def run_tests():
    """
    Запуск всех тестов
    """
    unittest.main(argv=[''], exit=False)

if __name__ == "__main__":
    run_tests() 