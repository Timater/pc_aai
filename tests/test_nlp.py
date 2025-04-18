"""
Тесты для модуля обработки естественно-языковых команд
"""

import pytest
from src.nlp import CommandProcessor

class TestCommandProcessor:
    """
    Набор тестов для проверки класса CommandProcessor
    """
    
    @pytest.fixture
    def processor(self):
        """
        Фикстура для создания экземпляра CommandProcessor
        """
        return CommandProcessor()
    
    def test_parse_click_command(self, processor):
        """
        Тест на распознавание команд клика
        """
        # Тест различных форматов команд клика
        commands = [
            "нажми на кнопку Пуск",
            "кликни на иконку Chrome",
            "щелкни кнопку Применить",
            "клик по кнопке Отмена"
        ]
        
        for cmd in commands:
            result = processor.parse_command(cmd)
            assert result is not None
            assert result["action"] == "click"
            assert result["target"] is not None
            assert isinstance(result["params"], dict)
    
    def test_parse_type_command(self, processor):
        """
        Тест на распознавание команд ввода текста
        """
        # Команда с указанием поля ввода
        cmd = "введи в поле поиска 'Python tutorial'"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "type"
        assert result["target"] == "поле поиска"
        assert result["params"]["text"] == "Python tutorial"
        
        # Команда без указания поля ввода
        cmd = "напиши 'Hello, world!'"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "type"
        assert result["target"] is None
        assert result["params"]["text"] == "Hello, world!"
    
    def test_parse_open_command(self, processor):
        """
        Тест на распознавание команд открытия
        """
        commands = [
            "открой браузер Chrome",
            "запусти Блокнот",
            "активируй программу Word"
        ]
        
        for cmd in commands:
            result = processor.parse_command(cmd)
            assert result is not None
            assert result["action"] == "open"
            assert result["target"] is not None
            assert isinstance(result["params"], dict)
    
    def test_parse_close_command(self, processor):
        """
        Тест на распознавание команд закрытия
        """
        commands = [
            "закрой текущее окно",
            "выключи программу Excel",
            "заверши работу приложения"
        ]
        
        for cmd in commands:
            result = processor.parse_command(cmd)
            assert result is not None
            assert result["action"] == "close"
            assert result["target"] is not None
            assert isinstance(result["params"], dict)
    
    def test_parse_search_command(self, processor):
        """
        Тест на распознавание команд поиска
        """
        # Команда с указанием места поиска
        cmd = "найди в Google 'как обучить YOLO модель'"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "search"
        assert result["target"] == "google"
        assert result["params"]["query"] == "как обучить YOLO модель"
        
        # Команда без указания места поиска
        cmd = "поищи 'погода Москва'"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "search"
        assert result["target"] is None
        assert result["params"]["query"] == "погода Москва"
    
    def test_parse_scroll_command(self, processor):
        """
        Тест на распознавание команд прокрутки
        """
        # Команда с указанием количества
        cmd = "прокрути страницу вниз на 5"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "scroll"
        assert result["target"] == "страницу"
        assert result["params"]["direction"] == "down"
        assert result["params"]["amount"] == 5
        
        # Команда без указания количества
        cmd = "прокрути вверх"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "scroll"
        assert result["target"] is None
        assert result["params"]["direction"] == "up"
        assert result["params"]["amount"] == 1
    
    def test_parse_file_operations(self, processor):
        """
        Тест на распознавание команд для работы с файлами
        """
        # Создание файла
        cmd = "создай файл 'новый документ.txt'"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "create"
        assert result["params"]["type"] == "файл"
        assert result["params"]["path"] == "новый документ.txt"
        
        # Удаление файла
        cmd = "удали файл 'ненужный файл.tmp'"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "delete"
        assert result["params"]["type"] == "файл"
        assert result["params"]["path"] == "ненужный файл.tmp"
        
        # Перемещение файла
        cmd = "перемести файл 'документ.docx' в 'Документы'"
        result = processor.parse_command(cmd)
        
        assert result is not None
        assert result["action"] == "move"
        assert result["params"]["type"] == "файл"
        assert result["params"]["source"] == "документ.docx"
        assert result["params"]["destination"] == "Документы"
    
    def test_unrecognized_command(self, processor):
        """
        Тест на неопознанные команды
        """
        commands = [
            "",  # Пустая строка
            "привет, как дела?",  # Не является командой
            "123456",  # Только цифры
            "!@#$%^&*()"  # Специальные символы
        ]
        
        for cmd in commands:
            result = processor.parse_command(cmd)
            assert result is None
    
    def test_extract_keywords(self, processor):
        """
        Тест на извлечение ключевых слов
        """
        cmd = "открой файл в папке Документы на рабочем столе"
        keywords = processor.extract_keywords(cmd)
        
        # Проверяем, что стоп-слова удалены и остались только ключевые слова
        assert "открой" in keywords
        assert "файл" in keywords
        assert "папке" in keywords
        assert "документы" in keywords
        assert "рабочем" in keywords
        assert "столе" in keywords
        
        # Проверяем, что стоп-слова удалены
        assert "в" not in keywords
        assert "на" not in keywords
    
    def test_suggest_command(self, processor):
        """
        Тест на предложение команд
        """
        # Тест на начало команды
        suggestions = processor.suggest_command("открой")
        assert "открой браузер Chrome" in suggestions
        
        # Тест на ключевое слово
        suggestions = processor.suggest_command("файл")
        assert "создай файл 'новый документ.txt'" in suggestions
        assert "удали файл 'ненужный файл.tmp'" in suggestions
        assert "перемести файл 'документ.docx' в 'Документы'" in suggestions
        
        # Тест на несуществующую команду
        suggestions = processor.suggest_command("небывалая команда")
        assert len(suggestions) == 0 