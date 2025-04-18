import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nlp_processor import CommandProcessor

class TestCommandProcessor(unittest.TestCase):
    """
    Тесты для класса CommandProcessor
    """
    def setUp(self):
        """
        Настройка перед каждым тестом
        """
        self.processor = CommandProcessor()
    
    def test_parse_open_command(self):
        """
        Тест разбора команды открытия файла
        """
        command = "открой файл report.txt"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "open")
        self.assertEqual(result["params"].get("file_name"), "report.txt")
    
    def test_parse_open_app_command(self):
        """
        Тест разбора команды открытия приложения
        """
        command = "запусти программу браузер"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "open")
        self.assertEqual(result["params"].get("app_name"), "браузер")
    
    def test_parse_search_command(self):
        """
        Тест разбора команды поиска
        """
        command = "найди папку проекты"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "search")
        self.assertEqual(result["params"].get("folder_name"), "проекты")
    
    def test_parse_click_command(self):
        """
        Тест разбора команды клика
        """
        command = "нажми на кнопку"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "click")
        self.assertIn("кнопку", result["target"])
    
    def test_parse_click_coordinates_command(self):
        """
        Тест разбора команды клика по координатам
        """
        command = "кликни 500 300"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "click")
        self.assertEqual(result["params"].get("numbers"), [500, 300])
    
    def test_parse_type_command(self):
        """
        Тест разбора команды ввода текста
        """
        command = "напиши 'Привет, мир!'"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "type")
        self.assertEqual(result["params"].get("text"), "Привет, мир!")
    
    def test_parse_simple_type_command(self):
        """
        Тест разбора простой команды ввода текста без кавычек
        """
        command = "введи example@mail.com"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "type")
        self.assertEqual(result["params"].get("text"), "example@mail.com")
    
    def test_parse_create_command(self):
        """
        Тест разбора команды создания файла
        """
        command = "создай файл test.txt"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "create")
        self.assertEqual(result["params"].get("file_name"), "test.txt")
    
    def test_parse_create_folder_command(self):
        """
        Тест разбора команды создания папки
        """
        command = "создай папку Projects"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "create")
        self.assertEqual(result["params"].get("folder_name"), "Projects")
    
    def test_parse_delete_command(self):
        """
        Тест разбора команды удаления файла
        """
        command = "удали файл temp.log"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "delete")
        self.assertEqual(result["params"].get("file_name"), "temp.log")
    
    def test_parse_close_command(self):
        """
        Тест разбора команды закрытия
        """
        command = "закрой окно"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "close")
        self.assertEqual(result["target"], "окно")
    
    def test_parse_scroll_command(self):
        """
        Тест разбора команды прокрутки
        """
        command = "прокрути вниз"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "scroll")
        self.assertEqual(result["target"], "вниз")
    
    def test_parse_scroll_with_number_command(self):
        """
        Тест разбора команды прокрутки с числом
        """
        command = "прокрути на 5 строк вниз"
        result = self.processor.parse_command(command)
        
        self.assertEqual(result["action"], "scroll")
        self.assertEqual(result["params"].get("numbers"), [5])
        self.assertIn("вниз", result["target"])
    
    def test_extract_entity(self):
        """
        Тест извлечения сущности из текста
        """
        text = "обработай документ report.docx на компьютере"
        file_name = self.processor.extract_entity(text, "file")
        
        self.assertEqual(file_name, "report.docx")
    
    def test_extract_folder_entity(self):
        """
        Тест извлечения сущности папки
        """
        text = "перейди в папку Downloads на компьютере"
        folder_name = self.processor.extract_entity(text, "folder")
        
        self.assertEqual(folder_name, "Downloads")
    
    def test_extract_app_entity(self):
        """
        Тест извлечения сущности приложения
        """
        text = "запусти программу Chrome и открой Google"
        app_name = self.processor.extract_entity(text, "app")
        
        self.assertEqual(app_name, "Chrome")
    
    def test_unknown_command(self):
        """
        Тест обработки неизвестной команды
        """
        command = "сделай что-нибудь случайное"
        result = self.processor.parse_command(command)
        
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main() 