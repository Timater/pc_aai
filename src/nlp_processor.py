"""
Модуль для обработки естественно-языковых команд пользователя
"""

import re
import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Настройка логирования
logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    Класс для обработки команд на естественном языке
    
    Выполняет классификацию команд и извлечение именованных сущностей
    из текстовых запросов пользователя
    """
    
    # Типы команд
    COMMAND_TYPES = {
        'CLICK': ['нажми', 'нажать', 'кликни', 'кликнуть', 'щелкни', 'выбери', 'выбрать', 'активируй'],
        'INPUT': ['введи', 'ввести', 'напиши', 'написать', 'набери', 'напечатай', 'впиши'],
        'SCROLL': ['прокрути', 'прокрутить', 'скролл', 'листай', 'пролистай', 'промотай'],
        'FIND': ['найди', 'поиск', 'покажи', 'найти', 'искать', 'отобразить'],
        'OPEN': ['открой', 'запусти', 'открыть', 'запустить', 'включи', 'включить'],
        'CLOSE': ['закрой', 'выключи', 'закрыть', 'выключить', 'заверши', 'завершить'],
        'DRAG': ['перетащи', 'перетянуть', 'перенеси', 'перенести', 'двигай', 'двигать'],
        'SELECT': ['выдели', 'выделить', 'отметь', 'отметить', 'выбери', 'выбрать'],
        'COPY': ['копируй', 'скопируй', 'копировать', 'скопировать'],
        'PASTE': ['вставь', 'вставить'],
        'BACK': ['назад', 'вернись', 'вернуться'],
        'REFRESH': ['обнови', 'обновить', 'перезагрузи', 'перезагрузить'],
        'SCREENSHOT': ['скриншот', 'снимок', 'сделай скриншот', 'сфотографируй'],
        'HELP': ['помощь', 'помоги', 'справка'],
        'UNKNOWN': []
    }
    
    # Регулярные выражения для извлечения параметров
    PATTERNS = {
        # Для UI элементов
        'ui_element': r'(кнопк[уа]|поле|текст|ссылк[уа]|элемент|флажок|переключатель|чекбокс|список|меню|вкладк[уа]|таб[уа]|окно|значок|иконк[уа]|радиокнопк[уа]|слайдер|ползунок|прогрессбар)(\s+["\']([^"\']+)["\']|\s+(\w+))?',
        # Для текста (с учетом разных кавычек)
        'text': r'(?:["\']([^"\']+)["\']|[«»]([^«»]+)[«»])',
        # Для числовых значений
        'number': r'(\d+)',
        # Для координат
        'coordinates': r'(?:координаты|позиция|положение|точка|место)?\s*\(?(\d+)[.,\s]*(\d+)\)?',
        # Для путей к файлам
        'file_path': r'(?:файл|путь|документ)\s+(?:["\']([^"\']+)["\']|[«»]([^«»]+)[«»]|(\S+\.\w+))',
        # Для URL
        'url': r'(?:сайт|ссылка|страница|url)\s+(?:["\']([^"\']+)["\']|[«»]([^«»]+)[«»]|(?:https?://)?(\S+\.\w+\S*))',
        # Для направлений
        'direction': r'(вверх|вниз|влево|вправо|верх|низ|лево|право)'
    }
    
    def __init__(
        self,
        commands_file: Optional[str] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
        language: str = 'ru'
    ):
        """
        Инициализация процессора NLP
        
        Args:
            commands_file (str, optional): Путь к JSON-файлу с дополнительными командами
            custom_patterns (Dict[str, str], optional): Дополнительные шаблоны для извлечения параметров
            language (str): Язык для обработки ('ru' или 'en')
        """
        self.language = language
        
        # Загрузка пользовательских команд, если указан файл
        if commands_file and os.path.exists(commands_file):
            self._load_custom_commands(commands_file)
        
        # Добавление пользовательских шаблонов
        if custom_patterns:
            for pattern_name, pattern in custom_patterns.items():
                self.PATTERNS[pattern_name] = pattern
                
        # Компиляция регулярных выражений для оптимизации
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE | re.UNICODE) 
            for name, pattern in self.PATTERNS.items()
        }
        
        logger.info(f"NLP-процессор инициализирован, язык: {language}")
    
    def _load_custom_commands(self, commands_file: str) -> None:
        """
        Загрузка пользовательских команд из JSON-файла
        
        Args:
            commands_file (str): Путь к JSON-файлу с командами
        """
        try:
            with open(commands_file, 'r', encoding='utf-8') as f:
                custom_commands = json.load(f)
                
            for command_type, triggers in custom_commands.items():
                if command_type in self.COMMAND_TYPES:
                    # Добавляем к существующему типу команд
                    self.COMMAND_TYPES[command_type].extend(triggers)
                else:
                    # Создаем новый тип команд
                    self.COMMAND_TYPES[command_type] = triggers
                    
            logger.info(f"Загружены пользовательские команды из {commands_file}")
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке пользовательских команд: {e}")
    
    def classify_command(self, text: str) -> Tuple[str, float]:
        """
        Классификация команды по тексту
        
        Args:
            text (str): Текст команды
            
        Returns:
            Tuple[str, float]: Тип команды и уверенность классификации
        """
        text = text.lower().strip()
        
        # Создаем словарь количества совпадений для каждого типа команды
        matches = {cmd_type: 0 for cmd_type in self.COMMAND_TYPES}
        
        # Перебираем все типы команд и ищем совпадения
        for cmd_type, triggers in self.COMMAND_TYPES.items():
            for trigger in triggers:
                # Проверяем, начинается ли текст с триггера
                if text.startswith(trigger + ' ') or text == trigger:
                    matches[cmd_type] += 2  # Дополнительный вес для точного начала
                # Проверяем наличие триггера в тексте
                elif trigger in text.split():
                    matches[cmd_type] += 1
        
        # Если нет совпадений, возвращаем UNKNOWN
        if sum(matches.values()) == 0:
            return 'UNKNOWN', 0.0
        
        # Находим тип команды с наибольшим количеством совпадений
        best_match = max(matches.items(), key=lambda x: x[1])
        command_type, match_count = best_match
        
        # Вычисляем уверенность (нормализуем)
        confidence = min(match_count / 3.0, 1.0)  # Масштабируем до [0,1]
        
        return command_type, confidence
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Извлечение именованных сущностей из текста команды
        
        Args:
            text (str): Текст команды
            
        Returns:
            Dict[str, Any]: Словарь извлеченных сущностей
        """
        entities = {}
        
        # Извлекаем сущности с помощью регулярных выражений
        for entity_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(text)
            extracted = []
            
            for match in matches:
                if entity_type == 'coordinates':
                    # Особая обработка координат (пара чисел)
                    x, y = match.groups()
                    if x and y:
                        extracted.append((int(x), int(y)))
                else:
                    # Берем первую непустую группу
                    groups = [group for group in match.groups() if group]
                    if groups:
                        extracted.append(groups[0])
            
            if extracted:
                # Сохраняем список или один элемент в зависимости от количества
                entities[entity_type] = extracted[0] if len(extracted) == 1 else extracted
        
        # Добавляем очищенную команду (без параметров)
        command_text = text
        for entity_type, entity_value in entities.items():
            if isinstance(entity_value, list):
                for value in entity_value:
                    command_text = self._remove_entity(command_text, value)
            else:
                command_text = self._remove_entity(command_text, entity_value)
        
        entities['clean_command'] = command_text.strip()
        
        return entities
    
    def _remove_entity(self, text: str, entity: Union[str, Tuple]) -> str:
        """
        Удаление сущности из текста
        
        Args:
            text (str): Исходный текст
            entity (Union[str, Tuple]): Сущность для удаления
            
        Returns:
            str: Текст без сущности
        """
        if isinstance(entity, tuple):
            # Если это координаты или другие численные данные
            pattern = r'\(?\s*{0}\s*[,.]?\s*{1}\s*\)?'.format(entity[0], entity[1])
            return re.sub(pattern, '', text)
        else:
            # Для строковых данных
            return re.sub(r'["\']' + re.escape(str(entity)) + r'["\']|[«»]' + re.escape(str(entity)) + r'[«»]', '', text)
    
    def process_command(self, text: str) -> Dict[str, Any]:
        """
        Полная обработка текстовой команды
        
        Args:
            text (str): Текст команды пользователя
            
        Returns:
            Dict[str, Any]: Результат обработки команды
        """
        # Классификация типа команды
        command_type, confidence = self.classify_command(text)
        
        # Извлечение сущностей
        entities = self.extract_entities(text)
        
        # Формирование результата
        result = {
            'text': text,
            'command_type': command_type,
            'confidence': confidence,
            'entities': entities
        }
        
        # Анализ результата для формирования более специфичного действия
        action = self._analyze_result(result)
        result['action'] = action
        
        logger.debug(f"Обработана команда: {text} -> {command_type} (уверенность: {confidence:.2f})")
        
        return result
    
    def _analyze_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ результата распознавания для формирования действия
        
        Args:
            result (Dict[str, Any]): Результат распознавания команды
            
        Returns:
            Dict[str, Any]: Описание действия
        """
        command_type = result['command_type']
        entities = result['entities']
        
        # Формирование базовой структуры действия
        action = {
            'type': command_type,
            'params': {}
        }
        
        # Добавление параметров в зависимости от типа команды
        if command_type == 'CLICK':
            # Обработка команды клика
            if 'ui_element' in entities:
                action['params']['target'] = entities['ui_element']
                action['params']['target_type'] = 'ui_element'
            elif 'coordinates' in entities:
                action['params']['target'] = entities['coordinates']
                action['params']['target_type'] = 'coordinates'
            elif 'text' in entities:
                action['params']['target'] = entities['text']
                action['params']['target_type'] = 'text'
        
        elif command_type == 'INPUT':
            # Обработка команды ввода текста
            if 'text' in entities:
                action['params']['text'] = entities['text']
            if 'ui_element' in entities:
                action['params']['target'] = entities['ui_element']
                action['params']['target_type'] = 'ui_element'
        
        elif command_type == 'SCROLL':
            # Обработка команды прокрутки
            if 'direction' in entities:
                action['params']['direction'] = entities['direction']
            if 'number' in entities:
                action['params']['distance'] = entities['number']
        
        elif command_type == 'OPEN':
            # Обработка команды открытия
            if 'file_path' in entities:
                action['params']['target'] = entities['file_path']
                action['params']['target_type'] = 'file'
            elif 'url' in entities:
                action['params']['target'] = entities['url']
                action['params']['target_type'] = 'url'
            elif 'ui_element' in entities:
                action['params']['target'] = entities['ui_element']
                action['params']['target_type'] = 'ui_element'
        
        # Добавление других обработчиков для разных типов команд
        
        # Если есть другие значимые сущности, добавляем их
        for entity_type, entity_value in entities.items():
            if entity_type not in ['clean_command'] and entity_type not in action['params']:
                action['params'][entity_type] = entity_value
        
        return action


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Инициализация процессора NLP
    processor = NLPProcessor()
    
    # Примеры команд для тестирования
    test_commands = [
        "нажми на кнопку 'Сохранить'",
        "кликни по координатам 100, 200",
        "введи 'Привет, мир!' в поле поиска",
        "прокрути вниз на 5 строк",
        "открой файл 'документ.txt'",
        "найди текст 'пример' на странице",
        "закрой текущее окно",
        "выдели все строки",
        "перетащи иконку в папку",
        "сделай скриншот экрана"
    ]
    
    # Тестирование обработки команд
    for cmd in test_commands:
        result = processor.process_command(cmd)
        print(f"\nКоманда: {cmd}")
        print(f"Тип: {result['command_type']} (уверенность: {result['confidence']:.2f})")
        print(f"Сущности: {result['entities']}")
        print(f"Действие: {result['action']}") 