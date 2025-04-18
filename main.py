import os
import cv2
import time
import numpy as np
import argparse
from datetime import datetime

# Импортируем модули нашей системы
from src.detector import Detector
from src.nlp import CommandProcessor
from actions import ActionManager
from file_manager import FileManager
from logger import Logger

class PCAgent:
    """
    Основной класс агента для управления ПК
    """
    def __init__(self, model_path="models/detector.onnx", class_names=None):
        """
        Инициализация агента
        
        Args:
            model_path (str): Путь к файлу модели ONNX
            class_names (list): Список имен классов для модели
        """
        # Инициализация логгера
        self.logger_manager = Logger(log_dir="logs")
        self.logger = self.logger_manager.get_logger()
        self.logger.info("Инициализация ПК-агента")
        
        try:
            # Инициализация компонентов
            self.logger.info(f"Загрузка модели: {model_path}")
            self.detector = self._init_detector(model_path)
            self.action_manager = ActionManager()
            self.command_processor = CommandProcessor()
            self.file_manager = FileManager()
            
            self.class_names = class_names or [
                "button", "text_field", "checkbox", "radio", "dropdown", 
                "menu", "icon", "tab", "scrollbar", "window", "dialog"
            ]
            
            self.logger.info("ПК-агент успешно инициализирован")
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации агента: {str(e)}")
            raise
    
    def _init_detector(self, model_path):
        """
        Инициализация детектора
        
        Args:
            model_path (str): Путь к файлу модели
            
        Returns:
            Detector: Объект детектора
        """
        # Проверяем наличие файла модели
        if not os.path.exists(model_path):
            self.logger.warning(f"Файл модели не найден: {model_path}. Использование мок-детектора.")
            # Если файл не найден, возвращаем мок-детектор
            from tests import MockDetector
            return MockDetector()
        
        # Если файл найден, инициализируем настоящий детектор
        try:
            return Detector(model_path)
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации детектора: {str(e)}")
            # В случае ошибки возвращаем мок-детектор
            from tests import MockDetector
            return MockDetector()
    
    def make_screenshot(self):
        """
        Создание скриншота экрана
        
        Returns:
            numpy.ndarray: Скриншот в формате RGB
        """
        screenshot = self.action_manager.screenshot()
        
        # Конвертируем из формата PIL в numpy.ndarray
        screenshot_np = np.array(screenshot)
        
        # OpenCV использует BGR, преобразуем из RGB
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        return screenshot_bgr
    
    def detect_elements(self, screenshot=None):
        """
        Распознавание элементов интерфейса на скриншоте
        
        Args:
            screenshot (numpy.ndarray, optional): Скриншот. Если не указан, будет сделан новый.
            
        Returns:
            list: Список обнаруженных элементов
        """
        if screenshot is None:
            screenshot = self.make_screenshot()
        
        elements = self.detector.detect_elements(screenshot, self.class_names)
        self.logger_manager.log_detection(elements)
        
        return elements
    
    def process_command(self, command):
        """
        Обработка текстовой команды
        
        Args:
            command (str): Текстовая команда
            
        Returns:
            dict: Результат обработки команды
        """
        self.logger_manager.log_command(command)
        
        parsed_command = self.command_processor.parse_command(command)
        
        if parsed_command is None:
            self.logger.warning(f"Не удалось распознать команду: {command}")
            return {"status": "error", "message": "Команда не распознана"}
        
        self.logger_manager.log_command(command, parsed_command)
        
        # Выполнение действия в зависимости от типа команды
        action_result = self.execute_action(parsed_command)
        
        return action_result
    
    def execute_action(self, parsed_command):
        """
        Выполнение действия на основе разобранной команды
        
        Args:
            parsed_command (dict): Разобранная команда
            
        Returns:
            dict: Результат выполнения действия
        """
        action_type = parsed_command["action"]
        target = parsed_command["target"]
        params = parsed_command["params"]
        
        self.logger.info(f"Выполнение действия {action_type} для цели {target}")
        
        # Делаем скриншот и распознаем элементы
        screenshot = self.make_screenshot()
        elements = self.detect_elements(screenshot)
        
        try:
            # Обработка различных типов действий
            if action_type == "click":
                return self._handle_click_action(elements, target, params)
            elif action_type == "type":
                return self._handle_type_action(elements, target, params)
            elif action_type == "open":
                return self._handle_open_action(elements, target, params)
            elif action_type == "close":
                return self._handle_close_action(elements, target, params)
            elif action_type == "search":
                return self._handle_search_action(elements, target, params)
            elif action_type == "create":
                return self._handle_create_action(params)
            elif action_type == "delete":
                return self._handle_delete_action(params)
            elif action_type == "move":
                return self._handle_move_action(params)
            elif action_type == "scroll":
                return self._handle_scroll_action(elements, target, params)
            else:
                return {"status": "error", "message": f"Неизвестный тип действия: {action_type}"}
        except Exception as e:
            error_message = f"Ошибка при выполнении действия {action_type}: {str(e)}"
            self.logger.error(error_message)
            self.logger_manager.log_error(error_message, screenshot)
            return {"status": "error", "message": error_message}
    
    def _find_element_by_label(self, elements, label):
        """
        Поиск элемента интерфейса по метке
        
        Args:
            elements (list): Список элементов
            label (str): Метка для поиска
            
        Returns:
            dict: Найденный элемент или None
        """
        if not label:
            return None
        
        label = label.lower()
        
        for element in elements:
            element_label = element.get("class_name", "").lower()
            # Проверяем, содержит ли метка элемента искомую строку
            if label in element_label:
                return element
        
        return None
    
    def _handle_click_action(self, elements, target, params):
        """
        Обработка действия клика
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Целевой элемент
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения действия
        """
        if not target:
            return {"status": "error", "message": "Не указана цель для клика"}
        
        # Поиск элемента по названию
        element = self._find_element_by_label(elements, target)
        
        if element:
            # Получаем координаты центра элемента
            x = element["center_x"]
            y = element["center_y"]
            
            # Выполняем клик
            self.action_manager.click(x, y)
            return {"status": "success", "message": f"Выполнен клик по элементу '{target}' в координатах ({x}, {y})"}
        else:
            return {"status": "error", "message": f"Элемент '{target}' не найден на экране"}
    
    def _handle_type_action(self, elements, target, params):
        """
        Обработка действия ввода текста
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Целевой элемент
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения действия
        """
        text = params.get("text")
        
        if not text:
            return {"status": "error", "message": "Не указан текст для ввода"}
        
        # Если указана цель для ввода текста, сначала кликаем по ней
        if target:
            element = self._find_element_by_label(elements, target)
            
            if element:
                # Получаем координаты центра элемента
                x = element["center_x"]
                y = element["center_y"]
                
                # Выполняем клик
                self.action_manager.click(x, y)
            else:
                return {"status": "error", "message": f"Элемент '{target}' не найден на экране"}
        
        # Вводим текст
        self.action_manager.type_text(text)
        
        if target:
            return {"status": "success", "message": f"Введен текст '{text}' в элемент '{target}'"}
        else:
            return {"status": "success", "message": f"Введен текст '{text}'"}
    
    def _handle_open_action(self, elements, target, params):
        """
        Обработка действия открытия
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Целевой элемент
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения действия
        """
        if not target:
            return {"status": "error", "message": "Не указана цель для открытия"}
        
        # Проверяем, является ли цель программой
        common_apps = {
            "браузер": "chrome.exe",
            "блокнот": "notepad.exe",
            "калькулятор": "calc.exe",
            "проводник": "explorer.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "word": "WINWORD.EXE",
            "excel": "EXCEL.EXE"
        }
        
        # Проверяем, является ли целью одно из известных приложений
        for app_name, app_exe in common_apps.items():
            if app_name in target.lower():
                try:
                    # Запускаем программу
                    import subprocess
                    subprocess.Popen(app_exe)
                    return {"status": "success", "message": f"Запущено приложение '{app_exe}'"}
                except Exception as e:
                    return {"status": "error", "message": f"Ошибка при запуске приложения '{app_exe}': {str(e)}"}
        
        # Если не является программой, пробуем найти элемент на экране
        element = self._find_element_by_label(elements, target)
        
        if element:
            # Получаем координаты центра элемента
            x = element["center_x"]
            y = element["center_y"]
            
            # Выполняем клик
            self.action_manager.click(x, y)
            return {"status": "success", "message": f"Выполнен клик по элементу '{target}' для открытия"}
        
        # Проверяем, является ли целью файл или папка
        if os.path.exists(target):
            try:
                # Открываем файл или папку
                import subprocess
                subprocess.Popen(["explorer", target])
                return {"status": "success", "message": f"Открыт файл или папка '{target}'"}
            except Exception as e:
                return {"status": "error", "message": f"Ошибка при открытии файла или папки '{target}': {str(e)}"}
        
        return {"status": "error", "message": f"Не удалось открыть '{target}'"}
    
    def _handle_close_action(self, elements, target, params):
        """
        Обработка действия закрытия
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Целевой элемент
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения действия
        """
        if not target:
            # Если цель не указана, пробуем найти кнопку закрытия окна
            close_buttons = []
            for element in elements:
                if element.get("class_name", "").lower() in ["close", "закрыть", "крестик", "×", "x"]:
                    close_buttons.append(element)
            
            if close_buttons:
                # Выбираем кнопку с наибольшей уверенностью
                close_button = max(close_buttons, key=lambda x: x.get("confidence", 0))
                
                # Получаем координаты центра элемента
                x = close_button["center_x"]
                y = close_button["center_y"]
                
                # Выполняем клик
                self.action_manager.click(x, y)
                return {"status": "success", "message": "Выполнен клик по кнопке закрытия"}
            
            # Если не нашли кнопку закрытия, отправляем Alt+F4
            self.action_manager.press_key_combination(["alt", "f4"])
            return {"status": "success", "message": "Отправлена комбинация клавиш Alt+F4"}
        
        # Если указана цель, ищем элемент на экране
        element = self._find_element_by_label(elements, target)
        
        if element:
            # Получаем координаты центра элемента
            x = element["center_x"]
            y = element["center_y"]
            
            # Выполняем клик
            self.action_manager.click(x, y)
            return {"status": "success", "message": f"Выполнен клик по элементу '{target}' для закрытия"}
        
        return {"status": "error", "message": f"Не удалось закрыть '{target}'"}
    
    def _handle_search_action(self, elements, target, params):
        """
        Обработка действия поиска
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Место поиска
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения действия
        """
        query = params.get("query")
        
        if not query:
            return {"status": "error", "message": "Не указан запрос для поиска"}
        
        # Если указано место поиска, сначала кликаем по соответствующему элементу
        if target:
            element = self._find_element_by_label(elements, target)
            
            if element:
                # Получаем координаты центра элемента
                x = element["center_x"]
                y = element["center_y"]
                
                # Выполняем клик
                self.action_manager.click(x, y)
            else:
                # Если не нашли указанный элемент, ищем поле поиска
                search_elements = []
                for element in elements:
                    if element.get("class_name", "").lower() in ["search", "поиск", "найти", "text_field"]:
                        search_elements.append(element)
                
                if search_elements:
                    # Выбираем элемент с наибольшей уверенностью
                    search_element = max(search_elements, key=lambda x: x.get("confidence", 0))
                    
                    # Получаем координаты центра элемента
                    x = search_element["center_x"]
                    y = search_element["center_y"]
                    
                    # Выполняем клик
                    self.action_manager.click(x, y)
                else:
                    return {"status": "error", "message": f"Не найдено поле поиска для '{target}'"}
        
        # Вводим запрос
        self.action_manager.type_text(query)
        
        # Нажимаем Enter для выполнения поиска
        self.action_manager.press_key("enter")
        
        if target:
            return {"status": "success", "message": f"Выполнен поиск '{query}' в '{target}'"}
        else:
            return {"status": "success", "message": f"Выполнен поиск '{query}'"}
    
    def _handle_create_action(self, params):
        """
        Обработка действия создания файла или папки
        
        Args:
            params (dict): Параметры действия
            
        Returns:
            dict: Результат выполнения действия
        """
        item_type = params.get("type")
        path = params.get("path")
        
        if not item_type or not path:
            return {"status": "error", "message": "Не указан тип или путь для создания"}
        
        try:
            if item_type.lower() in ["file", "файл"]:
                result = self.file_manager.create_file(path)
                if result:
                    return {"status": "success", "message": f"Создан файл '{path}'"}
                else:
                    return {"status": "error", "message": f"Не удалось создать файл '{path}'"}
            elif item_type.lower() in ["folder", "directory", "папка", "директория"]:
                result = self.file_manager.create_directory(path)
                if result:
                    return {"status": "success", "message": f"Создана папка '{path}'"}
                else:
                    return {"status": "error", "message": f"Не удалось создать папку '{path}'"}
            else:
                return {"status": "error", "message": f"Неизвестный тип '{item_type}' для создания"}
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при создании '{path}': {str(e)}"}
    
    def _handle_delete_action(self, params):
        """
        Обработка действия удаления файла или папки
        
        Args:
            params (dict): Параметры действия
            
        Returns:
            dict: Результат выполнения действия
        """
        item_type = params.get("type")
        path = params.get("path")
        
        if not path:
            return {"status": "error", "message": "Не указан путь для удаления"}
        
        try:
            if item_type.lower() in ["file", "файл"]:
                result = self.file_manager.delete_file(path)
                if result:
                    return {"status": "success", "message": f"Удален файл '{path}'"}
                else:
                    return {"status": "error", "message": f"Не удалось удалить файл '{path}'"}
            elif item_type.lower() in ["folder", "directory", "папка", "директория"]:
                result = self.file_manager.delete_directory(path)
                if result:
                    return {"status": "success", "message": f"Удалена папка '{path}'"}
                else:
                    return {"status": "error", "message": f"Не удалось удалить папку '{path}'"}
            else:
                # Если тип не указан, пробуем удалить и файл, и папку
                if os.path.isfile(path):
                    result = self.file_manager.delete_file(path)
                    if result:
                        return {"status": "success", "message": f"Удален файл '{path}'"}
                    else:
                        return {"status": "error", "message": f"Не удалось удалить файл '{path}'"}
                elif os.path.isdir(path):
                    result = self.file_manager.delete_directory(path)
                    if result:
                        return {"status": "success", "message": f"Удалена папка '{path}'"}
                    else:
                        return {"status": "error", "message": f"Не удалось удалить папку '{path}'"}
                else:
                    return {"status": "error", "message": f"Не найден файл или папка '{path}'"}
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при удалении '{path}': {str(e)}"}
    
    def _handle_move_action(self, params):
        """
        Обработка действия перемещения файла или папки
        
        Args:
            params (dict): Параметры действия
            
        Returns:
            dict: Результат выполнения действия
        """
        item_type = params.get("type")
        source = params.get("source")
        destination = params.get("destination")
        
        if not source or not destination:
            return {"status": "error", "message": "Не указан исходный или целевой путь для перемещения"}
        
        try:
            result = self.file_manager.move(source, destination)
            if result:
                return {"status": "success", "message": f"Перемещено из '{source}' в '{destination}'"}
            else:
                return {"status": "error", "message": f"Не удалось переместить из '{source}' в '{destination}'"}
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при перемещении из '{source}' в '{destination}': {str(e)}"}
    
    def _handle_scroll_action(self, elements, target, params):
        """
        Обработка действия прокрутки
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Целевой элемент
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения действия
        """
        direction = params.get("direction", "down")
        amount = params.get("amount", 1)
        
        # Если указана цель, сначала кликаем по соответствующему элементу
        if target:
            element = self._find_element_by_label(elements, target)
            
            if element:
                # Получаем координаты центра элемента
                x = element["center_x"]
                y = element["center_y"]
                
                # Выполняем клик
                self.action_manager.click(x, y)
            else:
                return {"status": "error", "message": f"Элемент '{target}' не найден на экране"}
        
        # Выполняем прокрутку
        self.action_manager.scroll(direction, amount)
        
        if target:
            return {"status": "success", "message": f"Выполнена прокрутка {direction} на {amount} для элемента '{target}'"}
        else:
            return {"status": "success", "message": f"Выполнена прокрутка {direction} на {amount}"}
    
    def run_console(self):
        """
        Запуск консольного интерфейса
        """
        print("Запуск консольного интерфейса ПК-агента.")
        print("Введите команду:")
        
        while True:
            try:
                command = input("> ")
                
                if command.lower() in ["exit", "quit", "выход", "выйти", "стоп", "закрыть"]:
                    print("Завершение работы.")
                    break
                
                if not command:
                    continue
                
                result = self.process_command(command)
                
                if result["status"] == "success":
                    print(f"✅ {result['message']}")
                else:
                    print(f"❌ {result['message']}")
                
            except KeyboardInterrupt:
                print("\nПрервано пользователем. Завершение работы.")
                break
            except Exception as e:
                print(f"❌ Произошла ошибка: {str(e)}")


def parse_arguments():
    """
    Парсинг аргументов командной строки
    
    Returns:
        argparse.Namespace: Аргументы командной строки
    """
    parser = argparse.ArgumentParser(description="ПК-агент - система для управления ПК с помощью компьютерного зрения и NLP")
    parser.add_argument("--model_path", type=str, default="models/detector.onnx", help="Путь к файлу модели")
    parser.add_argument("--log_dir", type=str, default="logs", help="Директория для сохранения логов")
    
    return parser.parse_args()


def main():
    """
    Основная функция
    """
    args = parse_arguments()
    
    try:
        agent = PCAgent(model_path=args.model_path)
        agent.run_console()
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 