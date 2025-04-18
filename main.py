import os
import cv2
import time
import numpy as np
import argparse
from datetime import datetime

# Импортируем модули нашей системы
from src.detector import Detector
from actions import ActionManager
from nlp_processor import CommandProcessor
from file_manager import FileManager
from logger import Logger

class PCAgent:
    """
    Основной класс агента для управления ПК
    """
    def __init__(self, model_path="detector.onnx", class_names=None):
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
            if 'label' in element and element['label'].lower() in label or label in element['label'].lower():
                return element
        
        return None
    
    def _handle_click_action(self, elements, target, params):
        """
        Обработка действия клика
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Цель действия
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения
        """
        # Найти элемент для клика
        element = self._find_element_by_label(elements, target)
        
        if element:
            # Вычисляем координаты центра элемента
            x = element['x'] + element['width'] // 2
            y = element['y'] + element['height'] // 2
            
            # Выполняем клик
            self.action_manager.click(x, y)
            return {"status": "success", "message": f"Клик выполнен по элементу {element['label']}"}
        else:
            # Если элемент не найден, пробуем извлечь координаты из параметров
            if 'numbers' in params and len(params['numbers']) >= 2:
                x, y = params['numbers'][0], params['numbers'][1]
                self.action_manager.click(x, y)
                return {"status": "success", "message": f"Клик выполнен по координатам ({x}, {y})"}
            
            return {"status": "error", "message": f"Элемент для клика не найден: {target}"}
    
    def _handle_type_action(self, elements, target, params):
        """
        Обработка действия ввода текста
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Цель действия
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения
        """
        # Получаем текст для ввода
        text = params.get('text', target)
        
        if not text:
            return {"status": "error", "message": "Не указан текст для ввода"}
        
        # Находим текстовое поле, если оно указано
        text_field = None
        for element in elements:
            if 'label' in element and element['label'] == 'text_field':
                text_field = element
                break
        
        # Если нашли поле, кликаем по нему перед вводом
        if text_field:
            x = text_field['x'] + text_field['width'] // 2
            y = text_field['y'] + text_field['height'] // 2
            self.action_manager.click(x, y)
        
        # Вводим текст
        self.action_manager.type_text(text)
        
        return {"status": "success", "message": f"Текст '{text}' введен"}
    
    def _handle_open_action(self, elements, target, params):
        """
        Обработка действия открытия файла или приложения
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Цель действия
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения
        """
        app_name = params.get('app_name')
        file_name = params.get('file_name')
        
        if file_name:
            # Поиск файла
            file_path = self.file_manager.find_file(file_name)
            if file_path:
                # Имитируем открытие файла через двойной клик
                # На самом деле, нужно использовать соответствующие команды системы
                import subprocess
                try:
                    subprocess.Popen(['start', file_path], shell=True)
                    return {"status": "success", "message": f"Файл {file_name} открыт"}
                except Exception as e:
                    return {"status": "error", "message": f"Ошибка при открытии файла: {str(e)}"}
            else:
                return {"status": "error", "message": f"Файл {file_name} не найден"}
        
        elif app_name:
            # Запуск приложения
            try:
                # На Windows можно использовать команду start
                subprocess.Popen(['start', app_name], shell=True)
                return {"status": "success", "message": f"Приложение {app_name} запущено"}
            except Exception as e:
                return {"status": "error", "message": f"Ошибка при запуске приложения: {str(e)}"}
        
        # Если не указан ни файл, ни приложение, ищем интерфейсный элемент
        element = self._find_element_by_label(elements, target)
        if element:
            x = element['x'] + element['width'] // 2
            y = element['y'] + element['height'] // 2
            self.action_manager.double_click(x, y)
            return {"status": "success", "message": f"Элемент {element['label']} открыт"}
        
        return {"status": "error", "message": "Не указана цель для открытия"}
    
    def _handle_close_action(self, elements, target, params):
        """
        Обработка действия закрытия
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Цель действия
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения
        """
        # Поиск кнопки закрытия
        close_button = None
        
        # Сначала ищем конкретную кнопку закрытия
        for element in elements:
            if 'label' in element and (element['label'] == 'close_button' or 'close' in element['label'].lower()):
                close_button = element
                break
        
        # Если не нашли, ищем по обычной метке
        if close_button is None:
            close_button = self._find_element_by_label(elements, target)
        
        # Если нашли кнопку, кликаем по ней
        if close_button:
            x = close_button['x'] + close_button['width'] // 2
            y = close_button['y'] + close_button['height'] // 2
            self.action_manager.click(x, y)
            return {"status": "success", "message": f"Элемент {close_button.get('label', 'окно')} закрыт"}
        
        # Если не нашли кнопку, пробуем использовать сочетание клавиш Alt+F4
        self.action_manager.hotkey('alt', 'f4')
        return {"status": "success", "message": "Выполнена комбинация клавиш Alt+F4"}
    
    def _handle_search_action(self, elements, target, params):
        """
        Обработка действия поиска
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Цель действия
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения
        """
        file_name = params.get('file_name')
        folder_name = params.get('folder_name')
        
        if file_name:
            # Поиск файла
            file_path = self.file_manager.find_file(file_name)
            if file_path:
                return {"status": "success", "message": f"Файл найден: {file_path}"}
            else:
                return {"status": "error", "message": f"Файл {file_name} не найден"}
        
        elif folder_name:
            # В реальной системе здесь был бы поиск папки
            return {"status": "error", "message": "Поиск папок пока не реализован"}
        
        elif target:
            # Поиск элемента интерфейса
            element = self._find_element_by_label(elements, target)
            if element:
                return {
                    "status": "success", 
                    "message": f"Элемент {element['label']} найден в позиции ({element['x']}, {element['y']})"
                }
            else:
                return {"status": "error", "message": f"Элемент {target} не найден"}
        
        return {"status": "error", "message": "Не указана цель для поиска"}
    
    def _handle_create_action(self, params):
        """
        Обработка действия создания файла или папки
        
        Args:
            params (dict): Параметры действия
            
        Returns:
            dict: Результат выполнения
        """
        file_name = params.get('file_name')
        folder_name = params.get('folder_name')
        
        if file_name:
            # Создание файла
            desktop_path = self.file_manager.get_default_path("desktop")
            file_path = os.path.join(desktop_path, file_name)
            
            if self.file_manager.create_file(file_path):
                return {"status": "success", "message": f"Файл {file_name} создан на рабочем столе"}
            else:
                return {"status": "error", "message": f"Ошибка при создании файла {file_name}"}
        
        elif folder_name:
            # Создание папки
            desktop_path = self.file_manager.get_default_path("desktop")
            folder_path = os.path.join(desktop_path, folder_name)
            
            if self.file_manager.create_folder(folder_path):
                return {"status": "success", "message": f"Папка {folder_name} создана на рабочем столе"}
            else:
                return {"status": "error", "message": f"Ошибка при создании папки {folder_name}"}
        
        return {"status": "error", "message": "Не указано что создавать (файл или папку)"}
    
    def _handle_delete_action(self, params):
        """
        Обработка действия удаления файла или папки
        
        Args:
            params (dict): Параметры действия
            
        Returns:
            dict: Результат выполнения
        """
        file_name = params.get('file_name')
        folder_name = params.get('folder_name')
        
        if file_name:
            # Поиск и удаление файла
            desktop_path = self.file_manager.get_default_path("desktop")
            file_path = os.path.join(desktop_path, file_name)
            
            if os.path.exists(file_path):
                if self.file_manager.delete_file(file_path):
                    return {"status": "success", "message": f"Файл {file_name} удален"}
                else:
                    return {"status": "error", "message": f"Ошибка при удалении файла {file_name}"}
            else:
                # Поиск файла в других папках
                file_path = self.file_manager.find_file(file_name)
                if file_path and self.file_manager.delete_file(file_path):
                    return {"status": "success", "message": f"Файл {file_name} найден и удален"}
                else:
                    return {"status": "error", "message": f"Файл {file_name} не найден"}
        
        elif folder_name:
            # Удаление папки
            desktop_path = self.file_manager.get_default_path("desktop")
            folder_path = os.path.join(desktop_path, folder_name)
            
            if os.path.exists(folder_path):
                if self.file_manager.delete_folder(folder_path):
                    return {"status": "success", "message": f"Папка {folder_name} удалена"}
                else:
                    return {"status": "error", "message": f"Ошибка при удалении папки {folder_name}"}
            else:
                return {"status": "error", "message": f"Папка {folder_name} не найдена"}
        
        return {"status": "error", "message": "Не указано что удалять (файл или папку)"}
    
    def _handle_move_action(self, params):
        """
        Обработка действия перемещения файла
        
        Args:
            params (dict): Параметры действия
            
        Returns:
            dict: Результат выполнения
        """
        # В реальной системе здесь был бы код для перемещения файлов
        return {"status": "error", "message": "Действие перемещения пока не реализовано"}
    
    def _handle_scroll_action(self, elements, target, params):
        """
        Обработка действия прокрутки
        
        Args:
            elements (list): Список элементов интерфейса
            target (str): Цель действия
            params (dict): Дополнительные параметры
            
        Returns:
            dict: Результат выполнения
        """
        # Получаем количество щелчков колеса мыши
        clicks = -5  # По умолчанию прокручиваем вниз
        
        if 'numbers' in params and params['numbers']:
            clicks = params['numbers'][0]
            # Если указано отрицательное число, то прокручиваем вверх
            if target and ('вверх' in target.lower() or 'наверх' in target.lower()):
                clicks = abs(clicks)
            else:
                clicks = -abs(clicks)
        elif target:
            if 'вверх' in target.lower() or 'наверх' in target.lower():
                clicks = 5
            elif 'вниз' in target.lower():
                clicks = -5
        
        # Выполняем прокрутку
        self.action_manager.scroll(clicks)
        direction = "вверх" if clicks > 0 else "вниз"
        return {"status": "success", "message": f"Выполнена прокрутка {direction} на {abs(clicks)} щелчков"}
    
    def run_console(self):
        """
        Запуск консольного интерфейса для ввода команд
        """
        print("Запуск консольного интерфейса ПК-агента.")
        print("Для выхода введите 'выход' или 'exit'.")
        
        while True:
            try:
                command = input("\nВведите команду: ")
                
                if command.lower() in ['выход', 'exit', 'quit']:
                    print("Завершение работы агента.")
                    break
                
                # Обработка команды
                result = self.process_command(command)
                
                # Вывод результата
                if result.get('status') == 'success':
                    print(f"✅ {result.get('message')}")
                else:
                    print(f"❌ {result.get('message')}")
                    
            except KeyboardInterrupt:
                print("\nПолучен сигнал прерывания. Завершение работы агента.")
                break
            except Exception as e:
                self.logger.error(f"Ошибка в консольном интерфейсе: {str(e)}")
                print(f"❌ Произошла ошибка: {str(e)}")

def parse_arguments():
    """
    Разбор аргументов командной строки
    
    Returns:
        argparse.Namespace: Объект с аргументами
    """
    parser = argparse.ArgumentParser(description='ПК-агент для управления интерфейсом')
    parser.add_argument('--model', type=str, default='detector.onnx',
                        help='Путь к ONNX-модели детектора')
    parser.add_argument('--mode', type=str, choices=['console', 'test'], default='console',
                        help='Режим работы: console - консольный интерфейс, test - запуск тестов')
    
    return parser.parse_args()

def main():
    """
    Основная функция запуска приложения
    """
    args = parse_arguments()
    
    if args.mode == 'test':
        print("Запуск тестов системы...")
        from tests import run_tests
        run_tests()
    else:
        # Создание и запуск агента
        try:
            agent = PCAgent(model_path=args.model)
            agent.run_console()
        except Exception as e:
            print(f"Критическая ошибка: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    main() 