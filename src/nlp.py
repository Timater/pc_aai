"""
Модуль обработки естественно-языковых команд для системы управления ПК
"""

import re
from typing import Dict, Any, Optional, List, Tuple

class CommandProcessor:
    """
    Класс для обработки и парсинга текстовых команд пользователя
    """
    
    def __init__(self):
        """
        Инициализация процессора команд
        """
        # Патерны для распознавания различных типов команд
        self.patterns = {
            "click": [
                r"(нажми|кликни|щелкни)(?:\s+на)?\s+(.+)",
                r"(клик|щелчок)(?:\s+на|по)?\s+(.+)"
            ],
            "type": [
                r"(введи|напиши|впиши)(?:\s+в\s+(.+?))?\s+[\"'](.+?)[\"']",
                r"(напечатай|ввод)(?:\s+в\s+(.+?))?\s+[\"'](.+?)[\"']"
            ],
            "open": [
                r"(открой|запусти|активируй)\s+(.+)",
                r"(открыть|запустить|активировать)\s+(.+)"
            ],
            "close": [
                r"(закрой|выключи|заверши)\s+(.+)",
                r"(закрыть|выключить|завершить)\s+(.+)"
            ],
            "search": [
                r"(найди|поищи|ищи)(?:\s+в\s+(.+?))?\s+[\"'](.+?)[\"']",
                r"(найти|поиск)(?:\s+в\s+(.+?))?\s+[\"'](.+?)[\"']"
            ],
            "scroll": [
                r"(прокрути|скролл)(?:\s+в\s+(.+?))?\s+(вверх|вниз|влево|вправо)(?:\s+на\s+(\d+))?",
                r"(прокрутка|скроллинг)(?:\s+в\s+(.+?))?\s+(вверх|вниз|влево|вправо)(?:\s+на\s+(\d+))?"
            ],
            "create": [
                r"(создай|сделай)\s+(файл|папку|директорию)\s+[\"'](.+?)[\"']",
                r"(создать|сделать)\s+(файл|папку|директорию)\s+[\"'](.+?)[\"']"
            ],
            "delete": [
                r"(удали|сотри|убери)\s+(файл|папку|директорию)\s+[\"'](.+?)[\"']",
                r"(удалить|стереть|убрать)\s+(файл|папку|директорию)\s+[\"'](.+?)[\"']"
            ],
            "move": [
                r"(перемести|перенеси)\s+(файл|папку)\s+[\"'](.+?)[\"']\s+в\s+[\"'](.+?)[\"']",
                r"(переместить|перенести)\s+(файл|папку)\s+[\"'](.+?)[\"']\s+в\s+[\"'](.+?)[\"']"
            ]
        }
        
        # Словарь направлений прокрутки
        self.scroll_directions = {
            "вверх": "up",
            "вниз": "down",
            "влево": "left",
            "вправо": "right"
        }
        
    def parse_command(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Анализирует текстовую команду и преобразует её в структурированный формат
        
        Args:
            command (str): Текстовая команда пользователя
            
        Returns:
            Optional[Dict[str, Any]]: Структурированная команда или None, если не удалось распознать
        """
        if not command:
            return None
            
        # Приводим к нижнему регистру и убираем лишние пробелы
        command = command.lower().strip()
        
        # Пробуем найти соответствие среди всех типов команд
        for action_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.match(pattern, command)
                if match:
                    return self._process_match(action_type, match)
        
        # Если команда не распознана, возвращаем None
        return None
        
    def _process_match(self, action_type: str, match) -> Dict[str, Any]:
        """
        Обрабатывает найденное соответствие шаблону и возвращает структурированную команду
        
        Args:
            action_type (str): Тип действия
            match: Результат регулярного выражения
            
        Returns:
            Dict[str, Any]: Структурированная команда
        """
        groups = match.groups()
        
        if action_type == "click":
            return {
                "action": "click",
                "target": groups[1],
                "params": {}
            }
            
        elif action_type == "type":
            # Если указано, где вводить текст
            if len(groups) > 2 and groups[1]:
                return {
                    "action": "type",
                    "target": groups[1],
                    "params": {"text": groups[2]}
                }
            # Если не указано, где вводить текст
            else:
                return {
                    "action": "type",
                    "target": None,
                    "params": {"text": groups[2]}
                }
                
        elif action_type == "open":
            return {
                "action": "open",
                "target": groups[1],
                "params": {}
            }
            
        elif action_type == "close":
            return {
                "action": "close",
                "target": groups[1],
                "params": {}
            }
            
        elif action_type == "search":
            # Если указано, где искать
            if len(groups) > 2 and groups[1]:
                return {
                    "action": "search",
                    "target": groups[1],
                    "params": {"query": groups[2]}
                }
            # Если не указано, где искать
            else:
                return {
                    "action": "search",
                    "target": None,
                    "params": {"query": groups[2]}
                }
                
        elif action_type == "scroll":
            direction = self.scroll_directions.get(groups[2], "down")
            amount = int(groups[3]) if len(groups) > 3 and groups[3] else 1
            
            return {
                "action": "scroll",
                "target": groups[1] if len(groups) > 1 and groups[1] else None,
                "params": {"direction": direction, "amount": amount}
            }
            
        elif action_type == "create":
            item_type = groups[1]
            path = groups[2]
            
            return {
                "action": "create",
                "target": None,
                "params": {"type": item_type, "path": path}
            }
            
        elif action_type == "delete":
            item_type = groups[1]
            path = groups[2]
            
            return {
                "action": "delete",
                "target": None,
                "params": {"type": item_type, "path": path}
            }
            
        elif action_type == "move":
            item_type = groups[1]
            source = groups[2]
            destination = groups[3]
            
            return {
                "action": "move",
                "target": None,
                "params": {"type": item_type, "source": source, "destination": destination}
            }
            
        # Если тип действия не обрабатывается
        return {
            "action": action_type,
            "target": None,
            "params": {}
        }

    def extract_keywords(self, command: str) -> List[str]:
        """
        Извлекает ключевые слова из команды
        
        Args:
            command (str): Текстовая команда
            
        Returns:
            List[str]: Список ключевых слов
        """
        # Удаляем стоп-слова и выделяем ключевые слова
        stopwords = ["на", "в", "по", "и", "с", "для", "к", "от", "из", "или", "что", "где", "как", "а"]
        words = command.lower().split()
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords

    def suggest_command(self, partial_command: str) -> List[str]:
        """
        Предлагает возможные команды на основе частичного ввода
        
        Args:
            partial_command (str): Частичная команда
            
        Returns:
            List[str]: Список предлагаемых команд
        """
        suggestions = []
        
        # Примеры наиболее часто используемых команд
        example_commands = [
            "нажми на кнопку Пуск",
            "открой браузер Chrome",
            "введи 'текст для ввода'",
            "закрой текущее окно",
            "найди 'информацию'",
            "прокрути вниз",
            "создай файл 'новый документ.txt'",
            "удали файл 'ненужный файл.tmp'",
            "перемести файл 'документ.docx' в 'Документы'"
        ]
        
        partial_lower = partial_command.lower()
        
        # Находим команды, начинающиеся с частичного ввода
        for cmd in example_commands:
            if cmd.lower().startswith(partial_lower):
                suggestions.append(cmd)
        
        # Если прямых совпадений нет, ищем по ключевым словам
        if not suggestions:
            keywords = self.extract_keywords(partial_command)
            for cmd in example_commands:
                for keyword in keywords:
                    if keyword in cmd.lower():
                        suggestions.append(cmd)
                        break
        
        return suggestions

# Пример использования
if __name__ == "__main__":
    processor = CommandProcessor()
    
    # Примеры команд для тестирования
    test_commands = [
        "нажми на кнопку Пуск",
        "открой браузер Chrome",
        "введи в поле поиска 'Python tutorial'",
        "закрой текущее окно",
        "найди в Google 'как обучить YOLO модель'",
        "прокрути страницу вниз на 5",
        "создай файл 'новый документ.txt'",
        "удали файл 'ненужный файл.tmp'",
        "перемести файл 'документ.docx' в 'Документы'"
    ]
    
    for cmd in test_commands:
        result = processor.parse_command(cmd)
        print(f"Команда: {cmd}")
        print(f"Результат: {result}\n") 