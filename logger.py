import logging
import os
from datetime import datetime
import cv2
import numpy as np

class Logger:
    """
    Класс для логирования действий системы и ошибок
    """
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        """
        Инициализация логгера
        
        Args:
            log_dir (str): Директория для файлов логов
            log_level: Уровень логирования
        """
        self.log_dir = log_dir
        
        # Создаем директорию для логов, если она не существует
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Текущая дата для имени файла лога
        current_date = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"agent_{current_date}.log")
        
        # Настройка логирования
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("pc_agent")
        self.logger.info("Логгер инициализирован")
    
    def log_error(self, message, screenshot=None):
        """
        Логирование ошибки с сохранением скриншота
        
        Args:
            message (str): Сообщение об ошибке
            screenshot: Скриншот экрана (numpy array или PIL.Image)
        """
        self.logger.error(message)
        
        if screenshot is not None:
            # Создаем папку для скриншотов ошибок
            error_screenshots_dir = os.path.join(self.log_dir, "error_screenshots")
            if not os.path.exists(error_screenshots_dir):
                os.makedirs(error_screenshots_dir)
            
            # Генерируем имя файла на основе времени
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(error_screenshots_dir, f"error_{timestamp}.png")
            
            # Конвертируем PIL.Image в numpy array, если необходимо
            if hasattr(screenshot, 'save'):
                screenshot.save(screenshot_path)
            else:
                # Предполагаем, что это numpy array
                cv2.imwrite(screenshot_path, screenshot)
            
            self.logger.error(f"Скриншот сохранен: {screenshot_path}")
    
    def log_action(self, action, details=None):
        """
        Логирование действия
        
        Args:
            action (str): Название действия
            details (dict, optional): Детали действия
        """
        if details:
            self.logger.info(f"Действие: {action}, детали: {details}")
        else:
            self.logger.info(f"Действие: {action}")
    
    def log_command(self, command, parsed_result=None):
        """
        Логирование команды
        
        Args:
            command (str): Исходная команда
            parsed_result (dict, optional): Результат разбора команды
        """
        self.logger.info(f"Команда: {command}")
        if parsed_result:
            self.logger.info(f"Разбор команды: {parsed_result}")
    
    def log_detection(self, elements):
        """
        Логирование результатов детекции
        
        Args:
            elements (list): Список обнаруженных элементов
        """
        self.logger.info(f"Обнаружено элементов: {len(elements)}")
        for i, elem in enumerate(elements):
            self.logger.debug(f"Элемент {i+1}: {elem}")
    
    def get_logger(self):
        """
        Получение объекта логгера
        
        Returns:
            Logger: Объект логгера
        """
        return self.logger 