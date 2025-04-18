"""
Модуль для детектирования UI элементов интерфейса при помощи ONNX-модели
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import List, Dict, Tuple, Union, Optional, Any

from src.image_processor import ImageProcessor
from src.ui_element import UIElement

logger = logging.getLogger(__name__)

class UIDetector:
    """
    Класс для детектирования UI элементов на скриншотах при помощи ONNX-модели
    """
    
    # Список поддерживаемых типов UI элементов
    SUPPORTED_ELEMENT_TYPES = [
        "button", 
        "checkbox",
        "input_field",
        "dropdown",
        "radio_button",
        "toggle",
        "slider",
        "link",
        "icon",
        "image",
        "text",
        "container"
    ]
    
    def __init__(
        self, 
        model_path: str,
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None
    ):
        """
        Инициализация детектора UI элементов
        
        Args:
            model_path (str): Путь к ONNX-модели
            confidence_threshold (float): Порог уверенности для фильтрации детекций
            input_size (Tuple[int, int]): Размер входного изображения для модели (ширина, высота)
            providers (List[str], optional): Список провайдеров для ONNX Runtime
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Инициализация процессора изображений
        self.image_processor = ImageProcessor(default_size=input_size)
        
        # Настройка провайдеров ONNX Runtime
        if providers is None:
            # Автоматический выбор провайдеров
            self.providers = ort.get_available_providers()
        else:
            self.providers = providers
            
        # Создание сессии ONNX Runtime
        self._create_session()
        
        # Получение информации о модели
        self._get_model_info()
        
        logger.info(
            f"UIDetector инициализирован с моделью: {model_path}, "
            f"порог уверенности: {confidence_threshold}, "
            f"размер входа: {input_size}, "
            f"провайдеры: {self.providers}"
        )
    
    def _create_session(self):
        """
        Создание сессии ONNX Runtime
        """
        try:
            session_options = ort.SessionOptions()
            # Оптимизация графа
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Количество потоков
            session_options.intra_op_num_threads = os.cpu_count() or 1
            
            # Создание сессии
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=self.providers
            )
            
            logger.debug(f"Сессия ONNX Runtime успешно создана с провайдерами: {self.providers}")
            
        except Exception as e:
            logger.error(f"Ошибка при создании сессии ONNX Runtime: {str(e)}")
            raise
    
    def _get_model_info(self):
        """
        Получение информации о входах и выходах модели
        """
        # Получение информации о входах
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}
        
        # Получение информации о выходах
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.debug(f"Входы модели: {self.input_names}, формы: {self.input_shapes}")
        logger.debug(f"Выходы модели: {self.output_names}")
    
    def detect(
        self, 
        image: Union[str, np.ndarray],
        enhance_image: bool = False,
        return_processed_image: bool = False
    ) -> Dict[str, Any]:
        """
        Детектирование UI элементов на изображении
        
        Args:
            image (Union[str, np.ndarray]): Путь к изображению или изображение в формате numpy array
            enhance_image (bool): Применять ли улучшение изображения перед детекцией
            return_processed_image (bool): Возвращать ли предобработанное изображение
            
        Returns:
            Dict[str, Any]: Результаты детекции 
                {
                    "ui_elements": List[UIElement],
                    "processed_image": np.ndarray (опционально),
                    "detection_time": float
                }
        """
        # Предобработка изображения
        processed_image, preprocess_info = self.image_processor.preprocess_for_detection(
            image=image,
            target_size=self.input_size,
            normalize=True,
            to_rgb=True,
            enhance_contrast=enhance_image
        )
        
        # Получение имени входного тензора (обычно "images" или "input")
        input_name = self.input_names[0]
        
        # Запуск инференса
        try:
            # Измерение времени выполнения
            start_time = cv2.getTickCount()
            
            # Подготовка входных данных
            input_data = {input_name: processed_image}
            
            # Запуск инференса
            outputs = self.session.run(self.output_names, input_data)
            
            # Вычисление времени
            end_time = cv2.getTickCount()
            inference_time = (end_time - start_time) / cv2.getTickFrequency()
            
            logger.debug(f"Время инференса: {inference_time:.4f} секунд")
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении инференса: {str(e)}")
            raise
        
        # Постобработка результатов
        detections = self._postprocess_results(outputs, preprocess_info)
        
        # Формирование результата
        result = {
            "ui_elements": detections,
            "detection_time": inference_time
        }
        
        # Если требуется, добавляем предобработанное изображение
        if return_processed_image:
            if isinstance(image, str):
                original_image = self.image_processor.load_image(image)
            else:
                original_image = image.copy()
                
            # Получаем боксы для визуализации
            boxes = [[elem.x1, elem.y1, elem.x2, elem.y2] for elem in detections]
            labels = [elem.element_type for elem in detections]
            confidences = [elem.confidence for elem in detections]
            
            # Отрисовываем боксы на изображении
            result["processed_image"] = self.image_processor.draw_boxes(
                original_image, 
                boxes, 
                labels=labels, 
                confidences=confidences
            )
        
        return result
    
    def _postprocess_results(
        self, 
        outputs: List[np.ndarray],
        preprocess_info: Dict[str, Any]
    ) -> List[UIElement]:
        """
        Постобработка результатов детекции
        
        Args:
            outputs (List[np.ndarray]): Выходы модели
            preprocess_info (Dict[str, Any]): Информация о предобработке изображения
            
        Returns:
            List[UIElement]: Список обнаруженных UI элементов
        """
        # Обработка выходов в зависимости от формата модели (YOLO, SSD и т.д.)
        # Здесь предполагается формат YOLO, который возвращает один выход
        # с формой [1, num_classes + 5, num_detections]
        
        # Получаем выход модели
        output = outputs[0]
        
        # Получаем размеры выхода
        num_classes = len(self.SUPPORTED_ELEMENT_TYPES)
        
        # Преобразуем выход в удобный формат
        # Предполагаем, что выход имеет формат [batch, num_detections, 5 + num_classes]
        # где 5 - это [x, y, width, height, confidence]
        
        # Список для хранения детекций
        ui_elements = []
        
        # Получаем детекции
        if len(output.shape) == 3:  # [batch, num_detections, 5 + num_classes]
            detections = output[0]  # Берем только первый батч
            
            # Обрабатываем каждую детекцию
            for detection in detections:
                # Получаем уверенность
                confidence = detection[4]
                
                # Фильтруем по порогу уверенности
                if confidence < self.confidence_threshold:
                    continue
                
                # Получаем класс с максимальной вероятностью
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                
                # Применяем дополнительную фильтрацию по уверенности класса
                if class_score < self.confidence_threshold:
                    continue
                
                # Получаем координаты центра и размеры бокса
                cx, cy, width, height = detection[:4]
                
                # Преобразуем в формат [x1, y1, x2, y2]
                x1 = cx - width / 2
                y1 = cy - height / 2
                x2 = cx + width / 2
                y2 = cy + height / 2
                
                # Масштабируем координаты обратно к исходному размеру изображения
                boxes = self.image_processor.scale_box_coordinates(
                    [[x1, y1, x2, y2]], 
                    preprocess_info
                )
                
                # Получаем масштабированные координаты
                x1, y1, x2, y2 = boxes[0]
                
                # Получаем тип элемента
                if class_id < len(self.SUPPORTED_ELEMENT_TYPES):
                    element_type = self.SUPPORTED_ELEMENT_TYPES[class_id]
                else:
                    element_type = "unknown"
                
                # Создаем объект UI элемента
                ui_element = UIElement(
                    element_id=f"{element_type}_{len(ui_elements)}",
                    element_type=element_type,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(confidence)
                )
                
                # Добавляем элемент в список
                ui_elements.append(ui_element)
                
        elif len(output.shape) == 2:  # [num_detections, 6] или [num_detections, 5 + num_classes]
            # Предполагаем формат [num_detections, x, y, w, h, confidence, class_id, ...]
            
            for detection in output:
                # Минимальная длина [x, y, w, h, confidence, class_id]
                if len(detection) < 6:
                    continue
                
                # Получаем уверенность
                confidence = detection[4]
                
                # Фильтруем по порогу уверенности
                if confidence < self.confidence_threshold:
                    continue
                
                # Получаем id класса
                if len(detection) == 6:  # [x, y, w, h, confidence, class_id]
                    class_id = int(detection[5])
                    class_score = confidence
                else:  # [x, y, w, h, confidence, class_scores...]
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_score = class_scores[class_id]
                
                # Применяем дополнительную фильтрацию по уверенности класса
                if class_score < self.confidence_threshold:
                    continue
                
                # Получаем координаты центра и размеры бокса
                cx, cy, width, height = detection[:4]
                
                # Преобразуем в формат [x1, y1, x2, y2]
                x1 = cx - width / 2
                y1 = cy - height / 2
                x2 = cx + width / 2
                y2 = cy + height / 2
                
                # Масштабируем координаты обратно к исходному размеру изображения
                boxes = self.image_processor.scale_box_coordinates(
                    [[x1, y1, x2, y2]], 
                    preprocess_info
                )
                
                # Получаем масштабированные координаты
                x1, y1, x2, y2 = boxes[0]
                
                # Получаем тип элемента
                if class_id < len(self.SUPPORTED_ELEMENT_TYPES):
                    element_type = self.SUPPORTED_ELEMENT_TYPES[class_id]
                else:
                    element_type = "unknown"
                
                # Создаем объект UI элемента
                ui_element = UIElement(
                    element_id=f"{element_type}_{len(ui_elements)}",
                    element_type=element_type,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(confidence)
                )
                
                # Добавляем элемент в список
                ui_elements.append(ui_element)
        
        # Сортируем элементы по уверенности (от большей к меньшей)
        ui_elements.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Обнаружено {len(ui_elements)} UI элементов")
        
        return ui_elements
    
    def export_detections_to_file(
        self, 
        detections: List[UIElement], 
        output_path: str, 
        format: str = "json"
    ) -> bool:
        """
        Экспорт результатов детекции в файл
        
        Args:
            detections (List[UIElement]): Список обнаруженных UI элементов
            output_path (str): Путь для сохранения результатов
            format (str): Формат экспорта (json, csv, xml)
            
        Returns:
            bool: True в случае успеха, False в случае неудачи
        """
        if not detections:
            logger.warning("Нет данных для экспорта")
            return False
            
        format = format.lower()
        
        try:
            # Конвертируем элементы в словари
            elements_data = [element.to_dict() for element in detections]
            
            # Экспортируем в зависимости от формата
            if format == "json":
                import json
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "version": "1.0",
                        "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
                        "ui_elements": elements_data
                    }, f, ensure_ascii=False, indent=2)
                    
            elif format == "csv":
                import csv
                
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ["element_id", "element_type", "x1", "y1", "x2", "y2", "confidence"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for element in detections:
                        writer.writerow({
                            "element_id": element.element_id,
                            "element_type": element.element_type,
                            "x1": element.x1,
                            "y1": element.y1,
                            "x2": element.x2,
                            "y2": element.y2,
                            "confidence": element.confidence
                        })
                        
            elif format == "xml":
                import xml.etree.ElementTree as ET
                from xml.dom import minidom
                
                root = ET.Element("ui_detections")
                root.set("version", "1.0")
                root.set("timestamp", str(cv2.getTickCount() / cv2.getTickFrequency()))
                
                elements_node = ET.SubElement(root, "ui_elements")
                
                for element in detections:
                    elem_node = ET.SubElement(elements_node, "ui_element")
                    elem_node.set("id", element.element_id)
                    
                    ET.SubElement(elem_node, "type").text = element.element_type
                    ET.SubElement(elem_node, "x1").text = str(element.x1)
                    ET.SubElement(elem_node, "y1").text = str(element.y1)
                    ET.SubElement(elem_node, "x2").text = str(element.x2)
                    ET.SubElement(elem_node, "y2").text = str(element.y2)
                    ET.SubElement(elem_node, "confidence").text = str(element.confidence)
                
                # Преобразуем в красивый XML
                xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(xmlstr)
                    
            else:
                logger.error(f"Неподдерживаемый формат экспорта: {format}")
                return False
                
            logger.info(f"Результаты детекции успешно экспортированы в {output_path} (формат: {format})")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте результатов детекции: {str(e)}")
            return False
            
    def get_element_by_type(
        self, 
        detections: List[UIElement], 
        element_type: str,
        min_confidence: float = 0.0
    ) -> List[UIElement]:
        """
        Получение элементов определенного типа из списка детекций
        
        Args:
            detections (List[UIElement]): Список обнаруженных UI элементов
            element_type (str): Тип элемента для фильтрации
            min_confidence (float): Минимальный порог уверенности
            
        Returns:
            List[UIElement]: Отфильтрованный список элементов
        """
        return [
            elem for elem in detections 
            if elem.element_type == element_type and elem.confidence >= min_confidence
        ]
    
    def get_element_by_position(
        self, 
        detections: List[UIElement], 
        x: int, 
        y: int
    ) -> Optional[UIElement]:
        """
        Получение элемента, содержащего указанную позицию
        
        Args:
            detections (List[UIElement]): Список обнаруженных UI элементов
            x (int): Координата X
            y (int): Координата Y
            
        Returns:
            Optional[UIElement]: Найденный элемент или None
        """
        matching_elements = []
        
        for elem in detections:
            if elem.x1 <= x <= elem.x2 and elem.y1 <= y <= elem.y2:
                matching_elements.append(elem)
        
        if not matching_elements:
            return None
            
        # Если несколько элементов содержат точку, возвращаем элемент с наименьшей площадью
        # (обычно самый специфичный элемент)
        return min(matching_elements, key=lambda e: e.area)
    
    def non_max_suppression(
        self, 
        detections: List[UIElement], 
        iou_threshold: float = 0.5
    ) -> List[UIElement]:
        """
        Применение алгоритма подавления не-максимумов (NMS) для устранения дублирующихся детекций
        
        Args:
            detections (List[UIElement]): Список обнаруженных UI элементов
            iou_threshold (float): Порог IoU для определения перекрытия
            
        Returns:
            List[UIElement]: Список элементов после применения NMS
        """
        if not detections:
            return []
            
        # Сортируем элементы по уверенности (от большей к меньшей)
        sorted_elements = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        selected_elements = []
        
        while sorted_elements:
            # Выбираем элемент с наибольшей уверенностью
            current = sorted_elements.pop(0)
            selected_elements.append(current)
            
            # Фильтруем оставшиеся элементы
            remaining_elements = []
            
            for element in sorted_elements:
                # Если элементы разного типа, сохраняем оба
                if element.element_type != current.element_type:
                    remaining_elements.append(element)
                    continue
                
                # Вычисляем IoU
                iou = UIElement.calculate_iou(current, element)
                
                # Если IoU меньше порога, сохраняем элемент
                if iou < iou_threshold:
                    remaining_elements.append(element)
            
            # Обновляем список оставшихся элементов
            sorted_elements = remaining_elements
        
        return selected_elements 