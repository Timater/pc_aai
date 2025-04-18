"""
Модуль контроллера компьютера для выполнения действий на основе распознавания UI и команд NLP
"""

import os
import cv2
import time
import logging
import numpy as np
import pyautogui
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Импорт собственных модулей
from .nlp_processor import NLPProcessor
from .ui_detector import UIDetector 
from .onnx_detector import OnnxDetector

# Настройка логирования
logger = logging.getLogger(__name__)

class PCController:
    """
    Контроллер для управления компьютером на основе распознавания UI и команд на естественном языке
    
    Обеспечивает связь между обработкой команд и выполнением действий,
    координирует работу модулей распознавания и управления
    """
    
    def __init__(
        self,
        detector_model_path: str = "models/ui_detector.onnx",
        commands_file: Optional[str] = None,
        delay: float = 0.5,
        move_duration: float = 0.2,
        language: str = 'ru'
    ):
        """
        Инициализация контроллера компьютера
        
        Args:
            detector_model_path (str): Путь к модели ONNX для распознавания UI элементов
            commands_file (Optional[str]): Путь к файлу с дополнительными командами для NLP
            delay (float): Задержка между действиями (в секундах)
            move_duration (float): Продолжительность перемещения курсора
            language (str): Язык для обработки команд ('ru' или 'en')
        """
        self.delay = delay
        self.move_duration = move_duration
        self.language = language
        
        # Классы UI элементов для детектора
        self.class_names = [
            "кнопка", "поле_ввода", "чекбокс", "радиокнопка", "выпадающий_список", 
            "меню", "иконка", "вкладка", "полоса_прокрутки", "окно", "диалог"
        ]
        
        # Инициализация детектора UI элементов
        try:
            logger.info(f"Инициализация детектора UI с моделью: {detector_model_path}")
            self.detector = OnnxDetector(
                model_path=detector_model_path,
                class_names=self.class_names,
                confidence_threshold=0.4
            )
        except Exception as e:
            logger.error(f"Ошибка при инициализации детектора UI: {e}")
            raise
        
        # Инициализация NLP-процессора
        try:
            logger.info("Инициализация NLP-процессора")
            self.nlp_processor = NLPProcessor(
                commands_file=commands_file,
                language=language
            )
        except Exception as e:
            logger.error(f"Ошибка при инициализации NLP-процессора: {e}")
            raise
        
        # Безопасность для операций с мышью и клавиатурой
        pyautogui.FAILSAFE = True
        
        logger.info("Контроллер ПК успешно инициализирован")
    
    def make_screenshot(self) -> np.ndarray:
        """
        Создание скриншота экрана
        
        Returns:
            np.ndarray: Изображение экрана в формате OpenCV (BGR)
        """
        try:
            # Создаем скриншот с помощью PyAutoGUI
            screenshot = pyautogui.screenshot()
            
            # Конвертируем в формат OpenCV
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            
            logger.debug(f"Создан скриншот экрана размером {screenshot.shape[1]}x{screenshot.shape[0]}")
            return screenshot
        except Exception as e:
            logger.error(f"Ошибка при создании скриншота: {e}")
            raise
    
    def detect_ui_elements(self, screenshot: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Обнаружение UI элементов на экране
        
        Args:
            screenshot (Optional[np.ndarray]): Скриншот экрана или None для создания нового
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных UI элементов
        """
        if screenshot is None:
            screenshot = self.make_screenshot()
        
        # Выполняем распознавание
        detections = self.detector.detect(screenshot)
        
        logger.info(f"Обнаружено {len(detections)} UI элементов")
        return detections
    
    def process_command(self, command_text: str) -> Dict[str, Any]:
        """
        Обработка текстовой команды и выполнение соответствующего действия
        
        Args:
            command_text (str): Текст команды пользователя
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        try:
            # Обработка команды NLP-процессором
            logger.info(f"Обработка команды: '{command_text}'")
            parsed_command = self.nlp_processor.process_command(command_text)
            
            if parsed_command['confidence'] < 0.3:
                logger.warning(f"Низкая уверенность распознавания команды: {parsed_command['confidence']:.2f}")
                return {
                    "status": "error",
                    "message": "Не удалось распознать команду с достаточной уверенностью",
                    "parsed_command": parsed_command
                }
            
            # Получаем действие и его параметры
            action = parsed_command['action']
            action_type = action['type']
            params = action['params']
            
            # Выполняем действие в зависимости от типа
            return self.execute_action(action_type, params)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке команды '{command_text}': {e}")
            return {
                "status": "error",
                "message": f"Ошибка при обработке команды: {str(e)}"
            }
    
    def execute_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение действия на основе типа и параметров
        
        Args:
            action_type (str): Тип действия (CLICK, INPUT, SCROLL и т.д.)
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        logger.info(f"Выполнение действия типа '{action_type}' с параметрами: {params}")
        
        # Делаем скриншот для поиска UI элементов
        screenshot = self.make_screenshot()
        ui_elements = self.detect_ui_elements(screenshot)
        
        # Обработка различных типов действий
        try:
            if action_type == "CLICK":
                return self._handle_click(ui_elements, params)
            elif action_type == "INPUT":
                return self._handle_input(ui_elements, params)
            elif action_type == "SCROLL":
                return self._handle_scroll(params)
            elif action_type == "OPEN":
                return self._handle_open(ui_elements, params)
            elif action_type == "CLOSE":
                return self._handle_close(ui_elements, params)
            elif action_type == "DRAG":
                return self._handle_drag(ui_elements, params)
            elif action_type == "SCREENSHOT":
                return self._handle_screenshot(params)
            elif action_type == "BACK":
                return self._handle_back()
            elif action_type == "REFRESH":
                return self._handle_refresh()
            elif action_type == "HELP":
                return self._handle_help()
            elif action_type == "UNKNOWN":
                return {
                    "status": "error",
                    "message": "Неизвестная команда. Используйте справку для получения списка доступных команд."
                }
            else:
                return {
                    "status": "error",
                    "message": f"Неподдерживаемый тип действия: {action_type}"
                }
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении действия {action_type}: {e}")
            return {
                "status": "error",
                "message": f"Ошибка при выполнении действия {action_type}: {str(e)}"
            }
    
    def _find_ui_element(self, ui_elements: List[Dict[str, Any]], target: str) -> Optional[Dict[str, Any]]:
        """
        Поиск UI элемента по тексту или другим параметрам
        
        Args:
            ui_elements (List[Dict[str, Any]]): Список обнаруженных UI элементов
            target (str): Текст или описание целевого элемента
            
        Returns:
            Optional[Dict[str, Any]]: Найденный элемент или None
        """
        if not target or not ui_elements:
            return None
        
        target = target.lower()
        best_match = None
        best_score = 0
        
        for element in ui_elements:
            # Проверяем текст элемента
            element_text = element.get('text', '').lower()
            
            # Проверяем класс элемента
            element_class = element.get('class', '').lower()
            
            # Рассчитываем метрику совпадения
            score = 0
            
            # Точное совпадение текста
            if element_text == target:
                score += 1.0
            # Частичное совпадение текста
            elif target in element_text:
                score += 0.7
            elif element_text in target:
                score += 0.5
                
            # Совпадение класса (если требуется найти определенный тип элемента)
            if element_class in target or target in element_class:
                score += 0.3
            
            # Сохраняем лучшее совпадение
            if score > best_score:
                best_score = score
                best_match = element
        
        # Возвращаем результат, если уверенность достаточная
        if best_score >= 0.5:
            logger.debug(f"Найден элемент '{best_match.get('text', '')}' с уверенностью {best_score:.2f}")
            return best_match
        
        return None
    
    def _handle_click(self, ui_elements: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка действия клика
        
        Args:
            ui_elements (List[Dict[str, Any]]): Список UI элементов
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        # Определяем цель для клика
        target_type = params.get('target_type')
        target = params.get('target')
        
        if not target:
            return {"status": "error", "message": "Не указана цель для клика"}
        
        # Клик по координатам
        if target_type == 'coordinates':
            x, y = target if isinstance(target, tuple) else (0, 0)
            try:
                pyautogui.moveTo(x, y, duration=self.move_duration)
                time.sleep(self.delay)
                pyautogui.click(x=x, y=y)
                return {"status": "success", "message": f"Выполнен клик по координатам ({x}, {y})"}
            except Exception as e:
                return {"status": "error", "message": f"Ошибка при клике по координатам: {str(e)}"}
        
        # Клик по UI элементу
        elif target_type in ['ui_element', 'text']:
            element = self._find_ui_element(ui_elements, target)
            
            if element:
                try:
                    # Получаем координаты центра элемента
                    x = element.get("center_x") or (element.get("x1") + element.get("x2")) // 2
                    y = element.get("center_y") or (element.get("y1") + element.get("y2")) // 2
                    
                    # Выполняем клик
                    pyautogui.moveTo(x, y, duration=self.move_duration)
                    time.sleep(self.delay)
                    pyautogui.click(x=x, y=y)
                    
                    return {
                        "status": "success", 
                        "message": f"Выполнен клик по элементу '{target}' в координатах ({x}, {y})"
                    }
                except Exception as e:
                    return {"status": "error", "message": f"Ошибка при клике по элементу: {str(e)}"}
            else:
                return {"status": "error", "message": f"Элемент '{target}' не найден на экране"}
        
        return {"status": "error", "message": "Неизвестный тип цели для клика"}
    
    def _handle_input(self, ui_elements: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка действия ввода текста
        
        Args:
            ui_elements (List[Dict[str, Any]]): Список UI элементов
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        text = params.get('text')
        target = params.get('target')
        target_type = params.get('target_type')
        
        if not text:
            return {"status": "error", "message": "Не указан текст для ввода"}
        
        # Если указан целевой элемент, сначала кликаем на него
        if target and target_type in ['ui_element', 'text']:
            element = self._find_ui_element(ui_elements, target)
            
            if element:
                # Получаем координаты центра элемента
                x = element.get("center_x") or (element.get("x1") + element.get("x2")) // 2
                y = element.get("center_y") or (element.get("y1") + element.get("y2")) // 2
                
                # Кликаем по элементу
                pyautogui.moveTo(x, y, duration=self.move_duration)
                time.sleep(self.delay)
                pyautogui.click(x=x, y=y)
                time.sleep(self.delay)
            else:
                return {"status": "error", "message": f"Поле ввода '{target}' не найдено на экране"}
        
        # Вводим текст
        try:
            pyautogui.write(text, interval=0.05)
            
            return {
                "status": "success", 
                "message": f"Выполнен ввод текста: '{text}'" + 
                           (f" в поле '{target}'" if target else "")
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при вводе текста: {str(e)}"}
    
    def _handle_scroll(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка действия прокрутки
        
        Args:
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        direction = params.get('direction', 'вниз')
        distance = int(params.get('distance', 3))
        
        # Определяем множитель для направления
        direction_multiplier = -1 if direction.lower() in ['вниз', 'down', 'низ'] else 1
        
        # Рассчитываем количество кликов колеса мыши
        scroll_amount = direction_multiplier * distance * 100
        
        try:
            pyautogui.scroll(scroll_amount)
            
            return {
                "status": "success", 
                "message": f"Выполнена прокрутка {direction} на {distance} единиц"
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при прокрутке: {str(e)}"}
    
    def _handle_open(self, ui_elements: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка действия открытия файла или программы
        
        Args:
            ui_elements (List[Dict[str, Any]]): Список UI элементов
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        target = params.get('target')
        target_type = params.get('target_type')
        
        if not target:
            return {"status": "error", "message": "Не указана цель для открытия"}
        
        # Открытие файла
        if target_type == 'file':
            try:
                os.startfile(target)
                return {"status": "success", "message": f"Открыт файл: {target}"}
            except Exception as e:
                return {"status": "error", "message": f"Ошибка при открытии файла: {str(e)}"}
        
        # Открытие URL
        elif target_type == 'url':
            try:
                # Дополняем URL, если нужно
                if not target.startswith(('http://', 'https://')):
                    target = f"https://{target}"
                
                import webbrowser
                webbrowser.open(target)
                return {"status": "success", "message": f"Открыт URL: {target}"}
            except Exception as e:
                return {"status": "error", "message": f"Ошибка при открытии URL: {str(e)}"}
        
        # Клик по UI элементу
        elif target_type == 'ui_element':
            element = self._find_ui_element(ui_elements, target)
            
            if element:
                # Получаем координаты центра элемента
                x = element.get("center_x") or (element.get("x1") + element.get("x2")) // 2
                y = element.get("center_y") or (element.get("y1") + element.get("y2")) // 2
                
                # Выполняем клик
                pyautogui.moveTo(x, y, duration=self.move_duration)
                time.sleep(self.delay)
                pyautogui.click(x=x, y=y)
                
                return {
                    "status": "success", 
                    "message": f"Открыт элемент: {target}"
                }
            else:
                return {"status": "error", "message": f"Элемент '{target}' не найден на экране"}
        
        return {"status": "error", "message": "Неизвестный тип цели для открытия"}
    
    def _handle_close(self, ui_elements: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка действия закрытия
        
        Args:
            ui_elements (List[Dict[str, Any]]): Список UI элементов
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        target = params.get('target')
        
        # Поиск кнопки закрытия, если цель не указана
        if not target:
            # Ищем кнопки закрытия (крестики)
            close_elements = []
            for element in ui_elements:
                element_text = element.get('text', '').lower()
                element_class = element.get('class', '').lower()
                
                # Проверяем, похоже ли это на кнопку закрытия
                if (element_text in ['закрыть', 'close', 'x', '×'] or
                    element_class in ['close', 'закрыть', 'close-button', 'button-close']):
                    close_elements.append(element)
            
            if close_elements:
                # Выбираем элемент с наивысшей уверенностью
                element = max(close_elements, key=lambda e: e.get('confidence', 0))
                
                # Получаем координаты центра элемента
                x = element.get("center_x") or (element.get("x1") + element.get("x2")) // 2
                y = element.get("center_y") or (element.get("y1") + element.get("y2")) // 2
                
                # Выполняем клик
                pyautogui.moveTo(x, y, duration=self.move_duration)
                time.sleep(self.delay)
                pyautogui.click(x=x, y=y)
                
                return {
                    "status": "success", 
                    "message": "Закрыто активное окно через кнопку закрытия"
                }
            
            # Если не нашли кнопку закрытия, используем Alt+F4
            try:
                pyautogui.hotkey('alt', 'f4')
                return {
                    "status": "success", 
                    "message": "Закрыто активное окно с помощью Alt+F4"
                }
            except Exception as e:
                return {"status": "error", "message": f"Ошибка при закрытии окна: {str(e)}"}
        
        # Если указана цель, ищем конкретный элемент
        element = self._find_ui_element(ui_elements, target)
        
        if element:
            # Получаем координаты центра элемента
            x = element.get("center_x") or (element.get("x1") + element.get("x2")) // 2
            y = element.get("center_y") or (element.get("y1") + element.get("y2")) // 2
            
            # Выполняем клик
            pyautogui.moveTo(x, y, duration=self.move_duration)
            time.sleep(self.delay)
            pyautogui.click(x=x, y=y)
            
            return {
                "status": "success", 
                "message": f"Закрыт элемент: {target}"
            }
        else:
            return {"status": "error", "message": f"Элемент '{target}' не найден на экране"}
    
    def _handle_drag(self, ui_elements: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка действия перетаскивания
        
        Args:
            ui_elements (List[Dict[str, Any]]): Список UI элементов
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        source = params.get('source')
        target = params.get('target')
        
        if not source:
            return {"status": "error", "message": "Не указан исходный элемент для перетаскивания"}
        
        # Находим исходный элемент
        source_element = self._find_ui_element(ui_elements, source)
        
        if not source_element:
            return {"status": "error", "message": f"Исходный элемент '{source}' не найден на экране"}
        
        # Получаем координаты центра исходного элемента
        source_x = source_element.get("center_x") or (source_element.get("x1") + source_element.get("x2")) // 2
        source_y = source_element.get("center_y") or (source_element.get("y1") + source_element.get("y2")) // 2
        
        # Если указан целевой элемент, находим его координаты
        if target:
            target_element = self._find_ui_element(ui_elements, target)
            
            if not target_element:
                return {"status": "error", "message": f"Целевой элемент '{target}' не найден на экране"}
            
            # Получаем координаты центра целевого элемента
            target_x = target_element.get("center_x") or (target_element.get("x1") + target_element.get("x2")) // 2
            target_y = target_element.get("center_y") or (target_element.get("y1") + target_element.get("y2")) // 2
        else:
            # Если целевой элемент не указан, используем относительные координаты
            # Или координаты, указанные в параметрах
            coordinates = params.get('coordinates', (100, 100))  # По умолчанию перемещаем на 100 пикселей
            
            if isinstance(coordinates, tuple) and len(coordinates) == 2:
                target_x, target_y = coordinates
            else:
                target_x = source_x + 100
                target_y = source_y + 100
        
        try:
            # Перемещаем мышь к исходному элементу
            pyautogui.moveTo(source_x, source_y, duration=self.move_duration)
            time.sleep(self.delay)
            
            # Начинаем перетаскивание
            pyautogui.mouseDown()
            time.sleep(self.delay)
            
            # Перемещаем к цели
            pyautogui.moveTo(target_x, target_y, duration=self.move_duration * 2)
            time.sleep(self.delay)
            
            # Отпускаем кнопку мыши
            pyautogui.mouseUp()
            
            return {
                "status": "success", 
                "message": f"Выполнено перетаскивание '{source}' в '{target or 'указанное положение'}'"
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при перетаскивании: {str(e)}"}
    
    def _handle_screenshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка действия создания скриншота
        
        Args:
            params (Dict[str, Any]): Параметры действия
            
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        filename = params.get('filename', f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        try:
            # Создаем скриншот
            screenshot = self.make_screenshot()
            
            # Создаем директорию screenshots, если не существует
            os.makedirs("screenshots", exist_ok=True)
            
            # Сохраняем скриншот
            file_path = os.path.join("screenshots", filename)
            cv2.imwrite(file_path, screenshot)
            
            return {
                "status": "success", 
                "message": f"Скриншот сохранен в файл: {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при создании скриншота: {str(e)}"}
    
    def _handle_back(self) -> Dict[str, Any]:
        """
        Обработка действия "назад" (обычно Alt+Left)
        
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        try:
            pyautogui.hotkey('alt', 'left')
            return {
                "status": "success", 
                "message": "Выполнен переход назад"
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при выполнении перехода назад: {str(e)}"}
    
    def _handle_refresh(self) -> Dict[str, Any]:
        """
        Обработка действия обновления (F5)
        
        Returns:
            Dict[str, Any]: Результат выполнения действия
        """
        try:
            pyautogui.press('f5')
            return {
                "status": "success", 
                "message": "Выполнено обновление страницы"
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при обновлении страницы: {str(e)}"}
    
    def _handle_help(self) -> Dict[str, Any]:
        """
        Обработка запроса справки
        
        Returns:
            Dict[str, Any]: Информация о доступных командах
        """
        available_commands = {
            "CLICK": "нажать на элемент (например: 'нажми на кнопку Сохранить')",
            "INPUT": "ввести текст (например: 'введи привет в поле поиска')",
            "SCROLL": "прокрутить экран (например: 'прокрути вниз')",
            "OPEN": "открыть файл или ссылку (например: 'открой файл документ.txt')",
            "CLOSE": "закрыть окно (например: 'закрой текущее окно')",
            "DRAG": "перетащить элемент (например: 'перетащи этот файл в корзину')",
            "SCREENSHOT": "сделать скриншот (например: 'сделай скриншот экрана')",
            "BACK": "вернуться назад (например: 'вернись на предыдущую страницу')",
            "REFRESH": "обновить страницу (например: 'обнови страницу')",
            "HELP": "показать справку (например: 'помощь', 'справка')"
        }
        
        help_text = "Доступные команды:\n\n"
        
        for cmd_type, description in available_commands.items():
            help_text += f"- {cmd_type}: {description}\n"
        
        return {
            "status": "success", 
            "message": help_text
        }


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Создаем экземпляр контроллера
    controller = PCController()
    
    # Примеры команд для тестирования
    test_commands = [
        "нажми на кнопку 'Сохранить'",
        "введи 'Привет, мир!' в поле поиска",
        "прокрути вниз на 5 строк",
        "сделай скриншот экрана"
    ]
    
    # Выполняем тестовые команды
    for cmd in test_commands:
        print(f"\nКоманда: {cmd}")
        result = controller.process_command(cmd)
        print(f"Результат: {result['status']}")
        print(f"Сообщение: {result['message']}") 