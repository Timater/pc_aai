import re
import logging

class CommandProcessor:
    """
    Класс для обработки и интерпретации текстовых команд
    """
    def __init__(self):
        """
        Инициализация обработчика команд
        """
        self.logger = logging.getLogger(__name__)
        
        # Словарь ключевых слов и соответствующих действий
        self.action_keywords = {
            # Открытие и запуск
            "открой": "open",
            "запусти": "open",
            "включи": "open",
            "активируй": "open",
            
            # Закрытие и выключение
            "закрой": "close",
            "выключи": "close",
            "деактивируй": "close",
            "останови": "close",
            
            # Поиск
            "найди": "search",
            "отыщи": "search",
            "поиск": "search",
            "где": "search",
            
            # Создание
            "создай": "create",
            "сделай": "create",
            
            # Удаление
            "удали": "delete",
            "убери": "delete",
            "очисти": "delete",
            
            # Перемещение
            "перемести": "move",
            "перенеси": "move",
            
            # Клик
            "нажми": "click",
            "кликни": "click",
            
            # Ввод текста
            "напиши": "type",
            "введи": "type",
            "набери": "type",
            
            # Скроллинг
            "прокрути": "scroll",
            "листай": "scroll"
        }
        
        # Словарь шаблонов для извлечения параметров
        self.patterns = {
            "file_pattern": r"(?:файл|документ)\s+([^\s]+)",
            "folder_pattern": r"(?:папк[ау]|директори[юя])\s+([^\s]+)",
            "app_pattern": r"(?:программ[ау]|приложение)\s+([^\s]+)"
        }
    
    def parse_command(self, command):
        """
        Разбор текстовой команды и извлечение действия и цели
        
        Args:
            command (str): Текстовая команда пользователя
            
        Returns:
            dict: Словарь с ключами 'action' и 'target', или None если команда не распознана
        """
        command = command.lower()
        
        self.logger.info(f"Обработка команды: {command}")
        
        # Определение действия
        action = None
        target = None
        
        for keyword, action_type in self.action_keywords.items():
            if keyword in command:
                action = action_type
                # Извлекаем текст после ключевого слова
                parts = command.split(keyword, 1)
                if len(parts) > 1:
                    target = parts[1].strip()
                break
        
        if not action:
            self.logger.warning("Действие не распознано")
            return None
        
        # Обработка параметров в зависимости от действия
        params = self._extract_parameters(command, action, target)
        
        result = {
            "action": action,
            "target": target,
            "params": params
        }
        
        self.logger.info(f"Результат обработки: {result}")
        return result
    
    def _extract_parameters(self, command, action, target):
        """
        Извлечение дополнительных параметров из команды
        
        Args:
            command (str): Исходная команда
            action (str): Определенное действие
            target (str): Цель действия
            
        Returns:
            dict: Словарь с дополнительными параметрами
        """
        params = {}
        
        # Извлечение имени файла
        file_match = re.search(self.patterns["file_pattern"], command)
        if file_match:
            params["file_name"] = file_match.group(1)
        
        # Извлечение имени папки
        folder_match = re.search(self.patterns["folder_pattern"], command)
        if folder_match:
            params["folder_name"] = folder_match.group(1)
        
        # Извлечение имени приложения
        app_match = re.search(self.patterns["app_pattern"], command)
        if app_match:
            params["app_name"] = app_match.group(1)
        
        # Извлечение числовых параметров
        numbers = re.findall(r'\b\d+\b', command)
        if numbers:
            params["numbers"] = [int(num) for num in numbers]
        
        # Обработка текста для ввода (для действия type)
        if action == "type" and target:
            # Извлекаем текст в кавычках, если есть
            text_match = re.search(r'[\"\'](.+?)[\"\']', target)
            if text_match:
                params["text"] = text_match.group(1)
            else:
                params["text"] = target
        
        return params
    
    def extract_entity(self, text, entity_type):
        """
        Извлечение сущности определенного типа из текста
        
        Args:
            text (str): Текст для анализа
            entity_type (str): Тип сущности ('file', 'folder', 'app', и т.д.)
            
        Returns:
            str: Извлеченная сущность или None
        """
        pattern_key = f"{entity_type}_pattern"
        if pattern_key in self.patterns:
            match = re.search(self.patterns[pattern_key], text)
            if match:
                return match.group(1)
        
        return None 