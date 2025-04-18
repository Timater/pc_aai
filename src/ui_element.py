"""
Модуль для работы с элементами пользовательского интерфейса (UI)
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, ClassVar, Callable
from dataclasses import dataclass
import uuid
import json
import re

logger = logging.getLogger(__name__)

class UIElement:
    """
    Класс для представления UI элемента, обнаруженного на экране
    """
    
    # Типы элементов с описанием на русском
    ELEMENT_TYPES_DESCRIPTION: ClassVar[Dict[str, str]] = {
        "button": "Кнопка",
        "checkbox": "Флажок",
        "input_field": "Поле ввода",
        "dropdown": "Выпадающий список",
        "radio_button": "Радиокнопка",
        "toggle": "Переключатель",
        "slider": "Ползунок",
        "link": "Ссылка",
        "icon": "Иконка",
        "image": "Изображение",
        "text": "Текст",
        "container": "Контейнер",
        "unknown": "Неизвестный элемент"
    }
    
    def __init__(
        self,
        element_id: str,
        element_type: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        text: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация UI элемента
        
        Args:
            element_id (str): Уникальный идентификатор элемента
            element_type (str): Тип элемента (button, checkbox и т.д.)
            x1 (float): Координата X левого верхнего угла
            y1 (float): Координата Y левого верхнего угла
            x2 (float): Координата X правого нижнего угла
            y2 (float): Координата Y правого нижнего угла
            confidence (float): Уверенность детекции (0.0 - 1.0)
            text (str, optional): Текст элемента, если есть
            attributes (Dict[str, Any], optional): Дополнительные атрибуты
        """
        self.element_id = element_id
        self.element_type = element_type
        self.x1 = int(round(x1))
        self.y1 = int(round(y1))
        self.x2 = int(round(x2))
        self.y2 = int(round(y2))
        self.confidence = confidence
        self.text = text
        self.attributes = attributes or {}
        
        # Вычисляем дополнительные параметры
        self._calculate_derived_metrics()
        
        logger.debug(
            f"Создан UI элемент: {self.element_id}, тип: {self.element_type}, "
            f"координаты: ({self.x1}, {self.y1}, {self.x2}, {self.y2}), "
            f"уверенность: {self.confidence:.2f}"
        )
    
    def _calculate_derived_metrics(self):
        """
        Вычисление производных метрик элемента (ширина, высота, центр, площадь)
        """
        # Размеры
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        
        # Центр
        self.center_x = (self.x1 + self.x2) // 2
        self.center_y = (self.y1 + self.y2) // 2
        
        # Площадь
        self.area = self.width * self.height
    
    def get_attribute(self, name: str, default: Any = None) -> Any:
        """
        Получение значения атрибута по имени
        
        Args:
            name (str): Имя атрибута
            default (Any, optional): Значение по умолчанию, если атрибут не найден
            
        Returns:
            Any: Значение атрибута или значение по умолчанию
        """
        return self.attributes.get(name, default)
    
    def has_attribute(self, name: str) -> bool:
        """
        Проверка наличия атрибута
        
        Args:
            name (str): Имя атрибута
            
        Returns:
            bool: True, если атрибут существует, иначе False
        """
        return name in self.attributes
    
    def get_description(self) -> str:
        """
        Получение описания элемента на русском языке
        
        Returns:
            str: Описание элемента
        """
        element_type_desc = self.ELEMENT_TYPES_DESCRIPTION.get(
            self.element_type, f"Элемент '{self.element_type}'"
        )
        
        description = f"{element_type_desc} ({self.width}x{self.height})"
        
        if self.text:
            # Обрезаем слишком длинный текст
            if len(self.text) > 30:
                short_text = self.text[:27] + "..."
            else:
                short_text = self.text
                
            description += f" с текстом '{short_text}'"
            
        return description
    
    @property
    def center(self) -> Tuple[int, int]:
        """
        Получение координат центра элемента
        
        Returns:
            Tuple[int, int]: Координаты центра в формате (x, y)
        """
        return (self.center_x, self.center_y)
    
    @property
    def area(self) -> int:
        """Площадь элемента в пикселях"""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """
        Вычисление соотношения сторон элемента (ширина/высота)
        
        Returns:
            float: Соотношение сторон (ширина/высота)
        """
        return self.width / max(self.height, 1)  # Избегаем деления на ноль
    
    @property
    def height(self) -> int:
        """Высота элемента в пикселях"""
        return self.y2 - self.y1
    
    @property
    def width(self) -> int:
        """Ширина элемента в пикселях"""
        return self.x2 - self.x1
    
    @classmethod
    def from_detection(cls, detection: Dict[str, Any], confidence_threshold: float = 0.5) -> Optional['UIElement']:
        """
        Создание экземпляра UI элемента из данных детекции
        
        Args:
            detection (Dict[str, Any]): Словарь с данными детекции
            confidence_threshold (float, optional): Порог уверенности
            
        Returns:
            Optional[UIElement]: Экземпляр UI элемента или None, если уверенность ниже порога
        """
        if detection.get("confidence", 1.0) < confidence_threshold:
            return None
        
        x1 = int(detection.get("x1", 0))
        y1 = int(detection.get("y1", 0))
        x2 = int(detection.get("x2", 0))
        y2 = int(detection.get("y2", 0))
        
        element_type = detection.get("type", "unknown")
        text = detection.get("text", "")
        confidence = detection.get("confidence", 1.0)
        attributes = detection.get("attributes", {})
        
        return cls(
            element_id=detection.get("id", str(uuid.uuid4())),
            element_type=element_type,
            text=text,
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=confidence,
            attributes=attributes
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование UI элемента в словарь
        
        Returns:
            Dict[str, Any]: Словарь с данными UI элемента
        """
        result = {
            "id": self.element_id,
            "type": self.element_type,
            "text": self.text,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "attributes": self.attributes
        }
        
        # Добавляем только непустые значения для опциональных полей
        if self.has_attribute("state"):
            result["state"] = self.get_attribute("state")
        
        if self.has_attribute("parent_id"):
            result["parent_id"] = self.get_attribute("parent_id")
            
        if self.has_attribute("children_ids") and self.get_attribute("children_ids"):
            result["children_ids"] = self.get_attribute("children_ids")
            
        return result
    
    def contains_point(self, x: int, y: int) -> bool:
        """
        Проверка, содержит ли элемент указанную точку
        
        Args:
            x (int): Координата X точки
            y (int): Координата Y точки
            
        Returns:
            bool: True, если точка находится внутри элемента, иначе False
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def get_relative_position(self, x: int, y: int) -> Tuple[float, float]:
        """
        Получение относительного положения точки внутри элемента в диапазоне [0, 1]
        
        Args:
            x (int): Координата X точки
            y (int): Координата Y точки
            
        Returns:
            Tuple[float, float]: Относительные координаты (x, y) в диапазоне [0, 1]
        """
        if not self.contains_point(x, y):
            raise ValueError(f"Точка ({x}, {y}) находится вне элемента {self.element_id}")
        
        rel_x = (x - self.x1) / max(self.width, 1)
        rel_y = (y - self.y1) / max(self.height, 1)
        
        return (rel_x, rel_y)
    
    def distance_to_point(self, x: int, y: int) -> float:
        """
        Вычисление минимального расстояния от элемента до точки
        
        Args:
            x (int): Координата X точки
            y (int): Координата Y точки
            
        Returns:
            float: Минимальное расстояние до точки
        """
        import math
        
        # Если точка внутри элемента, расстояние равно 0
        if self.contains_point(x, y):
            return 0.0
        
        # Находим ближайшую точку на границе прямоугольника
        closest_x = max(self.x1, min(x, self.x2))
        closest_y = max(self.y1, min(y, self.y2))
        
        # Вычисляем расстояние
        dx = x - closest_x
        dy = y - closest_y
        
        return math.sqrt(dx * dx + dy * dy)
    
    def iou(self, other: 'UIElement') -> float:
        """
        Вычисление IoU (Intersection over Union) с другим элементом
        
        Args:
            other (UIElement): Другой элемент UI
            
        Returns:
            float: Значение IoU в диапазоне [0, 1]
        """
        # Вычисляем координаты пересечения
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        # Если нет пересечения, возвращаем 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Вычисляем площадь пересечения
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Вычисляем площадь объединения
        self_area = self.area
        other_area = other.area
        union_area = self_area + other_area - intersection_area
        
        # Вычисляем IoU
        iou_value = intersection_area / max(union_area, 1)
        
        return iou_value
    
    def update_from_detection(self, detection: Dict[str, Any]) -> None:
        """
        Обновление атрибутов элемента на основе новой детекции
        
        Args:
            detection (Dict[str, Any]): Словарь с результатами детекции
        """
        # Проверка наличия необходимых ключей
        if "box" not in detection:
            raise ValueError("В данных детекции отсутствует обязательный ключ: box")
        
        # Обновляем координаты
        self.x1, self.y1, self.x2, self.y2 = detection["box"]
        
        # Пересчитываем вторичные атрибуты
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.center_x = (self.x1 + self.x2) // 2
        self.center_y = (self.y1 + self.y2) // 2
        
        # Обновляем другие атрибуты, если они есть в детекции
        if "confidence" in detection:
            self.confidence = detection["confidence"]
            
        if "text" in detection:
            self.text = detection["text"]
            
        if "state" in detection:
            self.state = detection["state"]
        
        # Обновляем дополнительные атрибуты
        for key, value in detection.items():
            if key not in ["box", "confidence", "text", "state"]:
                self.attributes[key] = value
    
    def __str__(self) -> str:
        """
        Строковое представление элемента
        
        Returns:
            str: Строковое представление
        """
        return (f"{self.element_type} (id: {self.element_id}): "
                f"box=({self.x1}, {self.y1}, {self.x2}, {self.y2}), "
                f"conf={self.confidence:.2f}"
                + (f", text='{self.text}'" if self.text else ""))

class UIElementCollection:
    """
    Класс для работы с коллекцией элементов UI
    """
    def __init__(self):
        """
        Инициализация пустой коллекции элементов
        """
        self.elements = {}  # Словарь элементов: id -> UIElement
        
    def add_element(self, element: UIElement) -> None:
        """
        Добавление элемента в коллекцию
        
        Args:
            element (UIElement): Элемент для добавления
        """
        self.elements[element.element_id] = element
        
    def add_from_detection(self, detection: Dict[str, Any], element_id: str = None) -> str:
        """
        Создание и добавление элемента из результата детекции
        
        Args:
            detection (Dict[str, Any]): Словарь с результатами детекции
            element_id (str, optional): ID элемента, если не указан - генерируется автоматически
            
        Returns:
            str: ID добавленного элемента
        """
        element = UIElement.from_detection(detection, element_id)
        self.add_element(element)
        return element.element_id
        
    def add_from_detections(self, detections: List[Dict[str, Any]]) -> List[str]:
        """
        Создание и добавление элементов из списка результатов детекции
        
        Args:
            detections (List[Dict[str, Any]]): Список словарей с результатами детекции
            
        Returns:
            List[str]: Список ID добавленных элементов
        """
        element_ids = []
        for detection in detections:
            element_id = self.add_from_detection(detection)
            element_ids.append(element_id)
        return element_ids
    
    def get_element(self, element_id: str) -> Optional[UIElement]:
        """
        Получение элемента по его ID
        
        Args:
            element_id (str): ID элемента
            
        Returns:
            Optional[UIElement]: Элемент или None, если не найден
        """
        return self.elements.get(element_id)
    
    def remove_element(self, element_id: str) -> bool:
        """
        Удаление элемента из коллекции
        
        Args:
            element_id (str): ID элемента для удаления
            
        Returns:
            bool: True, если элемент был удален, иначе False
        """
        if element_id in self.elements:
            del self.elements[element_id]
            
            # Удаляем ссылки на этот элемент из children_ids других элементов
            for element in self.elements.values():
                if element_id in element.children_ids:
                    element.children_ids.remove(element_id)
                    
            return True
        return False
    
    def clear(self) -> None:
        """
        Очистка коллекции (удаление всех элементов)
        """
        self.elements.clear()
    
    def find_by_type(self, element_type: str) -> List[UIElement]:
        """
        Поиск элементов по типу
        
        Args:
            element_type (str): Тип элемента
            
        Returns:
            List[UIElement]: Список найденных элементов
        """
        return [element for element in self.elements.values() 
                if element.element_type == element_type]
    
    def find_by_text(self, text: str, partial_match: bool = True) -> List[UIElement]:
        """
        Поиск элементов по тексту
        
        Args:
            text (str): Текст для поиска
            partial_match (bool): Если True, искать частичное совпадение, иначе точное
            
        Returns:
            List[UIElement]: Список найденных элементов
        """
        if partial_match:
            return [element for element in self.elements.values() 
                    if element.text and text.lower() in element.text.lower()]
        else:
            return [element for element in self.elements.values() 
                    if element.text and text.lower() == element.text.lower()]
    
    def find_by_position(self, x: int, y: int) -> List[UIElement]:
        """
        Поиск элементов, содержащих указанную точку
        
        Args:
            x (int): Координата X точки
            y (int): Координата Y точки
            
        Returns:
            List[UIElement]: Список найденных элементов
        """
        return [element for element in self.elements.values() 
                if element.contains_point(x, y)]
    
    def find_nearest(self, x: int, y: int, max_distance: float = float('inf'),
                    element_type: str = None) -> Optional[UIElement]:
        """
        Поиск ближайшего элемента к указанной точке
        
        Args:
            x (int): Координата X точки
            y (int): Координата Y точки
            max_distance (float): Максимальное расстояние для поиска
            element_type (str, optional): Фильтр по типу элемента
            
        Returns:
            Optional[UIElement]: Ближайший элемент или None, если не найден
        """
        elements = self.elements.values()
        
        # Фильтруем по типу, если указан
        if element_type:
            elements = [e for e in elements if e.element_type == element_type]
        
        if not elements:
            return None
        
        # Находим элемент с минимальным расстоянием
        nearest = min(elements, key=lambda e: e.distance_to_point(x, y))
        
        # Проверяем, что расстояние не превышает максимальное
        if nearest.distance_to_point(x, y) <= max_distance:
            return nearest
        
        return None
    
    def get_elements_in_region(self, x1: int, y1: int, x2: int, y2: int) -> List[UIElement]:
        """
        Получение элементов, находящихся в указанной области
        
        Args:
            x1 (int): Левая координата X области
            y1 (int): Верхняя координата Y области
            x2 (int): Правая координата X области
            y2 (int): Нижняя координата Y области
            
        Returns:
            List[UIElement]: Список элементов в указанной области
        """
        return [element for element in self.elements.values() 
                if (x1 <= element.center_x <= x2 and 
                    y1 <= element.center_y <= y2)]
    
    def set_parent_child_relation(self, parent_id: str, child_id: str) -> bool:
        """
        Установка отношения родитель-потомок между элементами
        
        Args:
            parent_id (str): ID родительского элемента
            child_id (str): ID дочернего элемента
            
        Returns:
            bool: True, если отношение успешно установлено, иначе False
        """
        parent = self.get_element(parent_id)
        child = self.get_element(child_id)
        
        if not parent or not child:
            return False
        
        # Устанавливаем отношение родитель-потомок
        if child_id not in parent.children_ids:
            parent.children_ids.append(child_id)
        
        child.parent_id = parent_id
        
        return True
    
    def get_children(self, element_id: str) -> List[UIElement]:
        """
        Получение дочерних элементов для указанного элемента
        
        Args:
            element_id (str): ID родительского элемента
            
        Returns:
            List[UIElement]: Список дочерних элементов
        """
        element = self.get_element(element_id)
        if not element:
            return []
        
        children = []
        for child_id in element.children_ids:
            child = self.get_element(child_id)
            if child:
                children.append(child)
        
        return children
    
    def get_parent(self, element_id: str) -> Optional[UIElement]:
        """
        Получение родительского элемента для указанного элемента
        
        Args:
            element_id (str): ID дочернего элемента
            
        Returns:
            Optional[UIElement]: Родительский элемент или None
        """
        element = self.get_element(element_id)
        if not element or not element.parent_id:
            return None
        
        return self.get_element(element.parent_id)
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Преобразование коллекции в словарь
        
        Returns:
            Dict[str, Dict[str, Any]]: Словарь с элементами
        """
        return {element_id: element.to_dict() for element_id, element in self.elements.items()}
    
    def __len__(self) -> int:
        """
        Получение количества элементов в коллекции
        
        Returns:
            int: Количество элементов
        """
        return len(self.elements)
    
    def __iter__(self):
        """
        Итератор по элементам коллекции
        
        Returns:
            Iterator[UIElement]: Итератор
        """
        return iter(self.elements.values())
    
    def __contains__(self, element_id: str) -> bool:
        """
        Проверка, содержится ли элемент с указанным ID в коллекции
        
        Args:
            element_id (str): ID элемента для проверки
            
        Returns:
            bool: True, если элемент найден, иначе False
        """
        return element_id in self.elements
    
    @staticmethod
    def filter_by_attributes(elements: List[UIElement], attributes: Dict[str, Any]) -> List[UIElement]:
        """
        Фильтрация элементов UI по атрибутам
        
        Args:
            elements (List[UIElement]): Список элементов для фильтрации
            attributes (Dict[str, Any]): Словарь атрибутов для фильтрации
            
        Returns:
            List[UIElement]: Отфильтрованный список элементов
        """
        filtered_elements = []
        
        for element in elements:
            matches = True
            for attr_name, attr_value in attributes.items():
                if not element.has_attribute(attr_name) or element.get_attribute(attr_name) != attr_value:
                    matches = False
                    break
            
            if matches:
                filtered_elements.append(element)
                
        return filtered_elements
    
    def set_attribute(self, element_id: str, name: str, value: Any) -> bool:
        """
        Установка атрибута для элемента с указанным ID
        
        Args:
            element_id (str): ID элемента
            name (str): Имя атрибута
            value (Any): Значение атрибута
            
        Returns:
            bool: True, если атрибут успешно установлен, иначе False
        """
        element = self.get_element(element_id)
        if element is None:
            return False
            
        element.attributes[name] = value
        return True
    
    @staticmethod
    def filter_by_text_content(elements: List[UIElement], text: str, partial_match: bool = True) -> List[UIElement]:
        """
        Фильтрация элементов UI по содержимому текста
        
        Args:
            elements (List[UIElement]): Список элементов для фильтрации
            text (str): Текст для поиска
            partial_match (bool, optional): Если True, выполняется поиск подстроки, 
                                           иначе - точное совпадение. По умолчанию True
            
        Returns:
            List[UIElement]: Отфильтрованный список элементов
        """
        filtered_elements = []
        
        for element in elements:
            if element.text:
                if partial_match and text.lower() in element.text.lower():
                    filtered_elements.append(element)
                elif not partial_match and text.lower() == element.text.lower():
                    filtered_elements.append(element)
                    
        return filtered_elements
        
    @staticmethod
    def sort_elements(
        elements: List[UIElement],
        criteria: str = 'y',
        reverse: bool = False,
        custom_key: Optional[Callable[[UIElement], Any]] = None
    ) -> List[UIElement]:
        """
        Сортирует список элементов UI по заданному критерию
        
        Args:
            elements (List[UIElement]): Список элементов для сортировки
            criteria (str): Критерий сортировки:
                - 'x': по горизонтальной позиции (слева направо)
                - 'y': по вертикальной позиции (сверху вниз)
                - 'text': по тексту элемента
                - 'type': по типу элемента
                - 'size': по размеру элемента (площади)
                - 'confidence': по уверенности распознавания
                - 'z': по z-index (порядку наложения)
                - 'id': по идентификатору элемента
                - 'reading': по порядку чтения (сверху вниз, затем слева направо)
            reverse (bool): Обратный порядок сортировки
            custom_key (Callable): Пользовательская функция для получения ключа сортировки
            
        Returns:
            List[UIElement]: Отсортированный список элементов
        """
        if not elements:
            return []
            
        # Копируем список, чтобы не изменять оригинал
        result = elements.copy()
        
        # Определяем функцию ключа сортировки
        if custom_key is not None:
            key_func = custom_key
        elif criteria == 'x':
            key_func = lambda e: e.x1
        elif criteria == 'y':
            key_func = lambda e: e.y1
        elif criteria == 'text':
            key_func = lambda e: e.text.lower() if e.text else ""
        elif criteria == 'type':
            key_func = lambda e: e.element_type
        elif criteria == 'size':
            key_func = lambda e: (e.x2 - e.x1) * (e.y2 - e.y1)
        elif criteria == 'confidence':
            key_func = lambda e: e.confidence
        elif criteria == 'z':
            key_func = lambda e: e.z_index if hasattr(e, 'z_index') else 0
        elif criteria == 'id':
            key_func = lambda e: e.element_id
        elif criteria == 'reading':
            # Разделяем экран на строки с определенной высотой
            # и сортируем элементы сначала по строкам, затем внутри строк по X
            row_height = 30  # Примерная высота строки (можно настроить)
            
            def reading_order(e):
                row = e.y1 // row_height
                return (row, e.x1)
                
            key_func = reading_order
        else:
            # По умолчанию сортируем по вертикальной позиции
            key_func = lambda e: e.y1
            
        # Сортируем элементы
        result.sort(key=key_func, reverse=reverse)
        
        return result
        
    @staticmethod
    def filter_by_relative_position(
        elements: List[UIElement],
        reference_element: UIElement,
        position: str,
        max_distance: Optional[float] = None
    ) -> List[UIElement]:
        """
        Фильтрация элементов UI по их относительному положению относительно указанного элемента
        
        Args:
            elements (List[UIElement]): Список элементов для фильтрации
            reference_element (UIElement): Элемент, относительно которого проверяется положение
            position (str): Положение ('above', 'below', 'left', 'right', 'inside', 'contains')
            max_distance (float, optional): Максимальное расстояние между элементами, 
                                           None - без ограничения
            
        Returns:
            List[UIElement]: Отфильтрованный список элементов
        """
        filtered_elements = []
        ref_center_x = reference_element.center[0]
        ref_center_y = reference_element.center[1]
        ref_x1, ref_y1, ref_x2, ref_y2 = reference_element.x1, reference_element.y1, reference_element.x2, reference_element.y2
        
        for element in elements:
            # Пропускаем сам элемент
            if element.element_id == reference_element.element_id:
                continue
                
            center_x = element.center[0]
            center_y = element.center[1]
            x1, y1, x2, y2 = element.x1, element.y1, element.x2, element.y2
            
            matches_position = False
            
            if position == 'above':
                matches_position = y2 < ref_y1 and x1 < ref_x2 and x2 > ref_x1
                distance = ref_y1 - y2 if matches_position else float('inf')
            elif position == 'below':
                matches_position = y1 > ref_y2 and x1 < ref_x2 and x2 > ref_x1
                distance = y1 - ref_y2 if matches_position else float('inf')
            elif position == 'left':
                matches_position = x2 < ref_x1 and y1 < ref_y2 and y2 > ref_y1
                distance = ref_x1 - x2 if matches_position else float('inf')
            elif position == 'right':
                matches_position = x1 > ref_x2 and y1 < ref_y2 and y2 > ref_y1
                distance = x1 - ref_x2 if matches_position else float('inf')
            elif position == 'inside':
                matches_position = x1 >= ref_x1 and y1 >= ref_y1 and x2 <= ref_x2 and y2 <= ref_y2
                distance = 0 if matches_position else float('inf')
            elif position == 'contains':
                matches_position = x1 <= ref_x1 and y1 <= ref_y1 and x2 >= ref_x2 and y2 >= ref_y2
                distance = 0 if matches_position else float('inf')
            else:
                logging.warning(f"Неизвестное положение: {position}. "
                              f"Допустимые значения: 'above', 'below', 'left', 'right', 'inside', 'contains'")
                return []
            
            if matches_position and (max_distance is None or distance <= max_distance):
                filtered_elements.append(element)
                
        return filtered_elements
    
    @staticmethod
    def filter_by_predicate(
        elements: List[UIElement], 
        predicate: Callable[[UIElement], bool]
    ) -> List[UIElement]:
        """
        Фильтрация элементов по пользовательскому предикату
        
        Args:
            elements (List[UIElement]): Список элементов для фильтрации
            predicate (Callable): Функция, принимающая элемент и возвращающая True/False
            
        Returns:
            List[UIElement]: Список отфильтрованных элементов
        """
        if not elements or not callable(predicate):
            return []
            
        return [element for element in elements if predicate(element)]
        
    @staticmethod
    def filter_by_size(
        elements: List[UIElement],
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None
    ) -> List[UIElement]:
        """
        Фильтрация элементов по размеру
        
        Args:
            elements (List[UIElement]): Список элементов для фильтрации
            min_width (int, optional): Минимальная ширина элемента в пикселях
            max_width (int, optional): Максимальная ширина элемента в пикселях
            min_height (int, optional): Минимальная высота элемента в пикселях
            max_height (int, optional): Максимальная высота элемента в пикселях
            min_area (int, optional): Минимальная площадь элемента в пикселях
            max_area (int, optional): Максимальная площадь элемента в пикселях
            
        Returns:
            List[UIElement]: Список отфильтрованных элементов
        """
        if not elements:
            return []
            
        filtered = elements.copy()
        
        # Фильтрация по ширине
        if min_width is not None:
            filtered = [e for e in filtered if e.width >= min_width]
            
        if max_width is not None:
            filtered = [e for e in filtered if e.width <= max_width]
            
        # Фильтрация по высоте
        if min_height is not None:
            filtered = [e for e in filtered if e.height >= min_height]
            
        if max_height is not None:
            filtered = [e for e in filtered if e.height <= max_height]
            
        # Фильтрация по площади
        if min_area is not None:
            filtered = [e for e in filtered if e.area >= min_area]
            
        if max_area is not None:
            filtered = [e for e in filtered if e.area <= max_area]
            
        return filtered
        
    @staticmethod
    def find_nearest_element(
        elements: List[UIElement],
        reference_element: UIElement,
        filter_predicate: callable = None
    ) -> Optional[UIElement]:
        """
        Нахождение ближайшего элемента к указанному опорному элементу
        
        Args:
            elements (List[UIElement]): Список элементов для поиска
            reference_element (UIElement): Опорный элемент
            filter_predicate (callable, optional): Функция-предикат для предварительной фильтрации элементов
            
        Returns:
            Optional[UIElement]: Ближайший элемент или None, если список пуст
        """
        if not elements:
            return None
            
        # Предварительная фильтрация, если предикат указан
        filtered_elements = elements
        if filter_predicate and callable(filter_predicate):
            filtered_elements = [e for e in elements if filter_predicate(e)]
            
        if not filtered_elements:
            return None
            
        # Исключаем сам опорный элемент
        filtered_elements = [e for e in filtered_elements if e.element_id != reference_element.element_id]
        
        if not filtered_elements:
            return None
            
        ref_center = reference_element.center
        
        # Находим элемент с минимальным расстоянием от центра до центра
        nearest = min(filtered_elements, key=lambda e: 
            ((e.center[0] - ref_center[0])**2 + (e.center[1] - ref_center[1])**2)**0.5)
            
        return nearest
        
    @staticmethod
    def group_elements(
        elements: List[UIElement],
        grouping_rule: str,
        max_distance: Optional[float] = None
    ) -> List[List[UIElement]]:
        """
        Группировка элементов UI по заданным правилам
        
        Args:
            elements (List[UIElement]): Список элементов для группировки
            grouping_rule (str): Правило группировки ('horizontal', 'vertical', 'grid', 'proximity')
            max_distance (float, optional): Максимальное расстояние между элементами в группе
            
        Returns:
            List[List[UIElement]]: Список групп элементов
        """
        if not elements:
            return []
            
        if grouping_rule == 'horizontal':
            # Группировка элементов в одной горизонтальной линии
            # Сначала сортируем по y-координате (вертикали)
            sorted_by_y = sorted(elements, key=lambda e: e.center[1])
            
            # Инициализируем группы
            groups = []
            current_group = [sorted_by_y[0]]
            
            # Проходим по элементам и группируем их по близости y-координат
            for i in range(1, len(sorted_by_y)):
                current = sorted_by_y[i]
                prev = current_group[-1]
                
                # Проверяем, находится ли текущий элемент близко по вертикали к предыдущему
                vertical_distance = abs(current.center[1] - prev.center[1])
                threshold = max_distance if max_distance is not None else min(current.height, prev.height) * 0.5
                
                if vertical_distance <= threshold:
                    current_group.append(current)
                else:
                    # Сортируем группу по x-координате (горизонтали)
                    current_group = sorted(current_group, key=lambda e: e.center[0])
                    groups.append(current_group)
                    current_group = [current]
            
            # Добавляем последнюю группу
            if current_group:
                current_group = sorted(current_group, key=lambda e: e.center[0])
                groups.append(current_group)
                
        elif grouping_rule == 'vertical':
            # Группировка элементов в одной вертикальной линии
            # Сначала сортируем по x-координате (горизонтали)
            sorted_by_x = sorted(elements, key=lambda e: e.center[0])
            
            # Инициализируем группы
            groups = []
            current_group = [sorted_by_x[0]]
            
            # Проходим по элементам и группируем их по близости x-координат
            for i in range(1, len(sorted_by_x)):
                current = sorted_by_x[i]
                prev = current_group[-1]
                
                # Проверяем, находится ли текущий элемент близко по горизонтали к предыдущему
                horizontal_distance = abs(current.center[0] - prev.center[0])
                threshold = max_distance if max_distance is not None else min(current.width, prev.width) * 0.5
                
                if horizontal_distance <= threshold:
                    current_group.append(current)
                else:
                    # Сортируем группу по y-координате (вертикали)
                    current_group = sorted(current_group, key=lambda e: e.center[1])
                    groups.append(current_group)
                    current_group = [current]
            
            # Добавляем последнюю группу
            if current_group:
                current_group = sorted(current_group, key=lambda e: e.center[1])
                groups.append(current_group)
                
        elif grouping_rule == 'proximity':
            # Группировка элементов по близости методом кластеризации
            # Используем простой алгоритм с порогом расстояния
            if not elements:
                return []
                
            # Начальные группы - каждый элемент в своей группе
            groups = [[element] for element in elements]
            
            # Порог расстояния между центрами элементов
            threshold = max_distance if max_distance is not None else 50
            
            # Объединяем группы, пока есть изменения
            changed = True
            while changed and len(groups) > 1:
                changed = False
                
                # Проверяем каждую пару групп
                for i in range(len(groups)):
                    if i >= len(groups):
                        continue
                        
                    group1 = groups[i]
                    
                    for j in range(i + 1, len(groups)):
                        if j >= len(groups):
                            continue
                            
                        group2 = groups[j]
                        
                        # Находим минимальное расстояние между элементами в группах
                        min_distance = float('inf')
                        for elem1 in group1:
                            for elem2 in group2:
                                dist = ((elem1.center[0] - elem2.center[0])**2 + 
                                        (elem1.center[1] - elem2.center[1])**2)**0.5
                                min_distance = min(min_distance, dist)
                        
                        # Если группы достаточно близко, объединяем их
                        if min_distance <= threshold:
                            groups[i] = group1 + group2
                            groups.pop(j)
                            changed = True
                            break
                    
                    if changed:
                        break
        else:
            logging.warning(f"Неизвестное правило группировки: {grouping_rule}. "
                          f"Допустимые значения: 'horizontal', 'vertical', 'proximity'")
            return []
            
        return groups
        
    @staticmethod
    def create_page_structure(
        elements: List[UIElement],
        include_container_hierarchy: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Создание структурированного представления страницы на основе UI элементов
        
        Args:
            elements (List[UIElement]): Список элементов UI
            include_container_hierarchy (bool): Включать ли информацию о вложенности элементов
            confidence_threshold (float): Минимальный порог уверенности для включения элемента
            
        Returns:
            Dict: Структурированное представление страницы
        """
        if not elements:
            return {"elements": [], "containers": [], "structure": {}}
            
        # Фильтруем элементы по уверенности
        filtered_elements = [e for e in elements if e.confidence >= confidence_threshold]
        
        # Базовая структура
        page_structure = {
            "elements": [],
            "containers": [],
            "structure": {
                "title": "Root",
                "children": []
            }
        }
        
        # Добавляем информацию о каждом элементе
        for element in filtered_elements:
            element_info = {
                "id": element.element_id,
                "type": element.element_type,
                "text": element.text or "",
                "position": {
                    "x1": element.x1,
                    "y1": element.y1,
                    "x2": element.x2,
                    "y2": element.y2,
                    "center": element.center
                },
                "confidence": element.confidence,
                "properties": element.attributes
            }
            
            page_structure["elements"].append(element_info)
            
            # Если это контейнер, добавляем в список контейнеров
            if element.element_type in ["window", "dialog", "panel", "tab", "frame", "table", "container"]:
                page_structure["containers"].append(element.element_id)
        
        # Если требуется иерархия, строим дерево вложенности
        if include_container_hierarchy and page_structure["containers"]:
            # Сортируем контейнеры от больших к меньшим (предполагая, что большие содержат меньшие)
            containers = [e for e in filtered_elements if e.element_id in page_structure["containers"]]
            containers.sort(key=lambda c: (c.width * c.height), reverse=True)
            
            # Создаем иерархию
            hierarchy = {}
            
            # Функция определения, находится ли элемент внутри контейнера
            def is_inside(element, container):
                return (element.x1 >= container.x1 and 
                        element.y1 >= container.y1 and 
                        element.x2 <= container.x2 and 
                        element.y2 <= container.y2)
            
            # Заполняем иерархию
            for container in containers:
                hierarchy[container.element_id] = {
                    "title": container.text or f"{container.element_type}_{container.element_id}",
                    "type": container.element_type,
                    "children": []
                }
            
            # Определяем вложенность контейнеров
            for container in containers:
                parent_found = False
                for parent in containers:
                    # Пропускаем сам элемент
                    if container.element_id == parent.element_id:
                        continue
                        
                    # Проверяем, находится ли контейнер внутри другого контейнера
                    if is_inside(container, parent):
                        hierarchy[parent.element_id]["children"].append(container.element_id)
                        parent_found = True
                        break
                
                # Если родитель не найден, добавляем к корню
                if not parent_found:
                    page_structure["structure"]["children"].append(container.element_id)
            
            # Распределяем элементы по контейнерам
            for element in filtered_elements:
                # Пропускаем контейнеры (они уже обработаны)
                if element.element_id in page_structure["containers"]:
                    continue
                    
                element_assigned = False
                
                # Ищем самый маленький контейнер, содержащий элемент
                suitable_containers = [c for c in containers if is_inside(element, c)]
                if suitable_containers:
                    # Сортируем по размеру от маленького к большому
                    suitable_containers.sort(key=lambda c: (c.width * c.height))
                    smallest_container = suitable_containers[0]
                    hierarchy[smallest_container.element_id]["children"].append(element.element_id)
                    element_assigned = True
                
                # Если не нашли подходящий контейнер, добавляем к корню
                if not element_assigned:
                    page_structure["structure"]["children"].append(element.element_id)
            
            # Добавляем информацию об иерархии в структуру
            page_structure["hierarchy"] = hierarchy
            
        return page_structure
        
    @staticmethod
    def serialize_elements(elements: List[UIElement], file_path: str) -> bool:
        """
        Сериализация списка UI элементов в JSON файл
        
        Args:
            elements (List[UIElement]): Список элементов для сериализации
            file_path (str): Путь к файлу для сохранения
            
        Returns:
            bool: True в случае успеха, False в случае ошибки
        """
        try:
            elements_data = []
            
            for element in elements:
                element_data = {
                    "id": element.element_id,
                    "element_type": element.element_type,
                    "x1": element.x1,
                    "y1": element.y1,
                    "x2": element.x2,
                    "y2": element.y2,
                    "confidence": element.confidence,
                    "text_content": element.text,
                    "properties": element.attributes
                }
                elements_data.append(element_data)
                
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(elements_data, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Успешно сериализовано {len(elements)} элементов в {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Ошибка при сериализации UI элементов: {str(e)}")
            return False
            
    @staticmethod
    def deserialize_elements(file_path: str) -> List[UIElement]:
        """
        Десериализация списка UI элементов из JSON файла
        
        Args:
            file_path (str): Путь к JSON файлу с сериализованными элементами
            
        Returns:
            List[UIElement]: Список десериализованных элементов
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                elements_data = json.load(f)
                
            elements = []
            
            for data in elements_data:
                element = UIElement(
                    element_id=data["id"],
                    element_type=data["element_type"],
                    x1=data["x1"],
                    y1=data["y1"],
                    x2=data["x2"],
                    y2=data["y2"],
                    confidence=data["confidence"],
                    text=data.get("text_content"),
                    attributes=data.get("properties", {})
                )
                elements.append(element)
                
            logging.info(f"Успешно десериализовано {len(elements)} элементов из {file_path}")
            return elements
            
        except Exception as e:
            logging.error(f"Ошибка при десериализации UI элементов: {str(e)}")
            return []
            
    @staticmethod
    def find_elements_by_relative_position(
        elements: List[UIElement],
        reference_element: UIElement,
        position: str = "below",
        max_distance: Optional[float] = None,
        same_type: bool = False
    ) -> List[UIElement]:
        """
        Поиск элементов по относительному положению относительно заданного элемента
        
        Args:
            elements (List[UIElement]): Список элементов для поиска
            reference_element (UIElement): Элемент относительно которого искать
            position (str): Относительное положение ('above', 'below', 'left', 'right', 
                           'near', 'inside', 'contains')
            max_distance (float, optional): Максимальное расстояние для поиска (пиксели)
            same_type (bool): Искать только элементы того же типа
            
        Returns:
            List[UIElement]: Список найденных элементов
        """
        if not elements or not reference_element:
            return []
            
        result = []
        ref_center_x, ref_center_y = reference_element.center
        
        # Фильтрация по типу, если нужно
        filtered_elements = elements.copy()
        if same_type:
            filtered_elements = [e for e in filtered_elements 
                              if e.element_type == reference_element.element_type]
        
        # Исключаем сам элемент из поиска
        filtered_elements = [e for e in filtered_elements if e.element_id != reference_element.element_id]
        
        # Рассчитываем границы для поиска
        for element in filtered_elements:
            element_center_x, element_center_y = element.center
            
            # Вычисляем расстояние между центрами элементов
            distance = ((element_center_x - ref_center_x) ** 2 + 
                       (element_center_y - ref_center_y) ** 2) ** 0.5
                       
            # Проверяем, не превышает ли расстояние максимальное, если оно задано
            if max_distance is not None and distance > max_distance:
                continue
                
            # Проверяем позицию в зависимости от заданного параметра
            if position == "above":
                # Элемент находится выше референсного
                if element.y2 <= reference_element.y1:
                    result.append(element)
                    
            elif position == "below":
                # Элемент находится ниже референсного
                if element.y1 >= reference_element.y2:
                    result.append(element)
                    
            elif position == "left":
                # Элемент находится слева от референсного
                if element.x2 <= reference_element.x1:
                    result.append(element)
                    
            elif position == "right":
                # Элемент находится справа от референсного
                if element.x1 >= reference_element.x2:
                    result.append(element)
                    
            elif position == "near":
                # Элемент находится рядом с референсным
                # Уже отфильтровано по расстоянию
                result.append(element)
                
            elif position == "inside":
                # Элемент находится внутри референсного
                if (element.x1 >= reference_element.x1 and
                    element.y1 >= reference_element.y1 and
                    element.x2 <= reference_element.x2 and
                    element.y2 <= reference_element.y2):
                    result.append(element)
                    
            elif position == "contains":
                # Элемент содержит в себе референсный
                if (element.x1 <= reference_element.x1 and
                    element.y1 <= reference_element.y1 and
                    element.x2 >= reference_element.x2 and
                    element.y2 >= reference_element.y2):
                    result.append(element)
        
        # Сортируем по расстоянию от референсного элемента
        result.sort(key=lambda e: ((e.center[0] - ref_center_x) ** 2 + 
                                 (e.center[1] - ref_center_y) ** 2) ** 0.5)
                                 
        return result 

    @staticmethod
    def group_by_property(
        elements: List[UIElement],
        property_name: str
    ) -> Dict[Any, List[UIElement]]:
        """
        Группировка элементов по указанному свойству
        
        Args:
            elements (List[UIElement]): Список элементов для группировки
            property_name (str): Имя свойства, по которому группировать
                                 (element_type, text_content, etc.)
            
        Returns:
            Dict[Any, List[UIElement]]: Словарь, где ключ - значение свойства,
                                        значение - список элементов
        """
        if not elements or not property_name:
            return {}
            
        result = {}
        
        for element in elements:
            # Получаем значение свойства
            if hasattr(element, property_name):
                property_value = getattr(element, property_name)
            elif property_name in element.__dict__:
                property_value = element.__dict__[property_name]
            else:
                # Пропускаем элемент, если у него нет такого свойства
                continue
                
            # Используем None как ключ для пустых значений
            key = property_value if property_value is not None else None
            
            # Преобразуем немутабельные типы в строки для использования в качестве ключей
            if isinstance(key, (list, dict, set)):
                key = str(key)
                
            # Добавляем элемент в соответствующую группу
            if key not in result:
                result[key] = []
                
            result[key].append(element)
            
        return result 

    @staticmethod
    def build_hierarchy(
        elements: List[UIElement]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Строит иерархию элементов (родитель-дети) на основе их пространственного расположения
        
        Args:
            elements (List[UIElement]): Список элементов для анализа
            
        Returns:
            Dict[str, Dict[str, Any]]: Словарь, представляющий иерархию, где:
                - Ключ: id элемента
                - Значение: словарь с ключами:
                  - 'element': сам элемент
                  - 'children': список id дочерних элементов
                  - 'parent': id родительского элемента или None
        """
        if not elements:
            return {}
            
        # Результирующая структура
        hierarchy = {}
        
        # Инициализируем иерархию для всех элементов
        for element in elements:
            hierarchy[element.element_id] = {
                'element': element,
                'children': [],
                'parent': None
            }
        
        # Сортируем элементы по площади (от большей к меньшей)
        # Это позволит сначала обработать более крупные контейнеры
        sorted_elements = sorted(elements, key=lambda e: e.area, reverse=True)
        
        # Для каждого элемента ищем потенциального родителя
        for element in sorted_elements:
            # Пропускаем корневые элементы
            if element.element_id not in hierarchy:
                continue
                
            # Проверяем другие элементы как потенциальных родителей
            # Ищем наименьший элемент, который полностью содержит текущий
            best_parent_id = None
            best_parent_area = float('inf')
            
            for potential_parent in sorted_elements:
                # Пропускаем сам элемент и элементы с меньшей площадью
                if (potential_parent.element_id == element.element_id or 
                    potential_parent.area <= element.area):
                    continue
                    
                # Проверяем, содержит ли потенциальный родитель текущий элемент
                if (element.x1 >= potential_parent.x1 and
                    element.y1 >= potential_parent.y1 and
                    element.x2 <= potential_parent.x2 and
                    element.y2 <= potential_parent.y2):
                    
                    # Выбираем наименьшего возможного родителя
                    if potential_parent.area < best_parent_area:
                        best_parent_id = potential_parent.element_id
                        best_parent_area = potential_parent.area
            
            # Если нашли родителя, обновляем иерархию
            if best_parent_id:
                hierarchy[element.element_id]['parent'] = best_parent_id
                hierarchy[best_parent_id]['children'].append(element.element_id)
        
        return hierarchy 

    @staticmethod
    def find_elements_spatial_relation(
        element: UIElement,
        other_elements: List[UIElement],
        relation_type: str = 'below',
        max_distance: Optional[int] = None
    ) -> List[UIElement]:
        """
        Находит элементы в заданном пространственном отношении к указанному элементу
        
        Args:
            element (UIElement): Опорный элемент
            other_elements (List[UIElement]): Список элементов для проверки
            relation_type (str): Тип отношения ('above', 'below', 'left', 'right',
                                 'contains', 'contained_in', 'near')
            max_distance (int, optional): Максимальное расстояние (для отношения 'near')
            
        Returns:
            List[UIElement]: Список элементов, находящихся в указанном отношении
        """
        if not element or not other_elements:
            return []
            
        result = []
        
        for other in other_elements:
            # Пропускаем сам элемент
            if element.element_id == other.element_id:
                continue
                
            # Координаты центров элементов
            ex, ey = element.center
            ox, oy = other.center
            
            # Расстояние между центрами
            distance = int(((ex - ox) ** 2 + (ey - oy) ** 2) ** 0.5)
            
            if relation_type == 'above':
                # Элемент находится над другим
                if (element.y2 <= other.y1 and
                    max(element.x1, other.x1) <= min(element.x2, other.x2)):
                    result.append(other)
                    
            elif relation_type == 'below':
                # Элемент находится под другим
                if (element.y1 >= other.y2 and
                    max(element.x1, other.x1) <= min(element.x2, other.x2)):
                    result.append(other)
                    
            elif relation_type == 'left':
                # Элемент находится слева от другого
                if (element.x2 <= other.x1 and
                    max(element.y1, other.y1) <= min(element.y2, other.y2)):
                    result.append(other)
                    
            elif relation_type == 'right':
                # Элемент находится справа от другого
                if (element.x1 >= other.x2 and
                    max(element.y1, other.y1) <= min(element.y2, other.y2)):
                    result.append(other)
                    
            elif relation_type == 'contains':
                # Элемент содержит другой
                if (element.x1 <= other.x1 and element.y1 <= other.y1 and
                    element.x2 >= other.x2 and element.y2 >= other.y2):
                    result.append(other)
                    
            elif relation_type == 'contained_in':
                # Элемент содержится в другом
                if (element.x1 >= other.x1 and element.y1 >= other.y1 and
                    element.x2 <= other.x2 and element.y2 <= other.y2):
                    result.append(other)
                    
            elif relation_type == 'near':
                # Элемент находится близко к другому
                if max_distance is None or distance <= max_distance:
                    result.append(other)
                    
        # Сортируем по расстоянию (ближайшие сначала)
        if relation_type in ['above', 'below', 'left', 'right', 'near']:
            result.sort(key=lambda e: ((element.center[0] - e.center[0]) ** 2 + 
                                      (element.center[1] - e.center[1]) ** 2) ** 0.5)
                                      
        return result 

    @staticmethod
    def find_nearest_element_by_type(
        reference_element: UIElement,
        elements: List[UIElement],
        element_type: str,
        max_distance: Optional[int] = None
    ) -> Optional[UIElement]:
        """
        Находит ближайший элемент указанного типа относительно опорного элемента
        
        Args:
            reference_element (UIElement): Опорный элемент
            elements (List[UIElement]): Список элементов для поиска
            element_type (str): Тип искомого элемента
            max_distance (int, optional): Максимальное расстояние для поиска
            
        Returns:
            Optional[UIElement]: Ближайший элемент указанного типа или None
        """
        if not reference_element or not elements or not element_type:
            return None
            
        # Фильтруем элементы по типу
        filtered_elements = [e for e in elements 
                            if e.element_type == element_type and 
                            e.element_id != reference_element.element_id]
        
        if not filtered_elements:
            return None
            
        # Координаты центра опорного элемента
        ref_x, ref_y = reference_element.center
        
        # Находим ближайший элемент
        nearest_element = None
        min_distance = float('inf')
        
        for element in filtered_elements:
            # Координаты центра текущего элемента
            elem_x, elem_y = element.center
            
            # Рассчитываем расстояние
            distance = ((ref_x - elem_x) ** 2 + (ref_y - elem_y) ** 2) ** 0.5
            
            # Проверяем, что элемент находится в пределах max_distance
            if max_distance is not None and distance > max_distance:
                continue
                
            if distance < min_distance:
                min_distance = distance
                nearest_element = element
                
        return nearest_element 

    @staticmethod
    def find_elements_by_regex(
        elements: List[UIElement],
        pattern: str,
        case_sensitive: bool = False,
        field: str = 'text'
    ) -> List[UIElement]:
        """
        Находит элементы, текст которых соответствует регулярному выражению
        
        Args:
            elements (List[UIElement]): Список элементов для поиска
            pattern (str): Строка с регулярным выражением
            case_sensitive (bool): Учитывать регистр при поиске
            field (str): Поле элемента для поиска ('text', 'element_type' и т.д.)
            
        Returns:
            List[UIElement]: Список элементов, соответствующих шаблону
        """
        if not elements or not pattern:
            return []
            
        # Компилируем регулярное выражение
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            logger.error(f"Ошибка компиляции регулярного выражения '{pattern}': {e}")
            return []
            
        result = []
        
        for element in elements:
            # Получаем значение поля для поиска
            if field == 'text':
                field_value = element.text or ""
            elif hasattr(element, field):
                field_value = str(getattr(element, field))
            elif field in element.attributes:
                field_value = str(element.attributes[field])
            else:
                continue
                
            # Проверяем соответствие регулярному выражению
            if regex.search(field_value):
                result.append(element)
                
        return result 