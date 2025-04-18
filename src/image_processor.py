"""
Модуль для предобработки и обработки изображений перед детекцией UI элементов
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Union, Optional, List

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Класс для предобработки и обработки изображений
    """
    
    def __init__(self, default_size: Tuple[int, int] = (640, 640)):
        """
        Инициализация процессора изображений
        
        Args:
            default_size (Tuple[int, int]): Размер по умолчанию для преобразования изображений (ширина, высота)
        """
        self.default_size = default_size
        logger.info(f"Инициализирован процессор изображений с размером по умолчанию {default_size}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Загрузка изображения из файла
        
        Args:
            image_path (str): Путь к файлу изображения
            
        Returns:
            np.ndarray: Загруженное изображение в формате BGR
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если файл не является валидным изображением
        """
        if not image_path:
            raise ValueError("Путь к изображению не может быть пустым")
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
                
            logger.debug(f"Изображение успешно загружено: {image_path}, размер: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения {image_path}: {str(e)}")
            raise
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Изменение размера изображения
        
        Args:
            image (np.ndarray): Исходное изображение
            target_size (Tuple[int, int], optional): Целевой размер (ширина, высота)
                                                    Если None, используется default_size
            
        Returns:
            np.ndarray: Изображение с измененным размером
        """
        if image is None or image.size == 0:
            raise ValueError("Входное изображение пустое или некорректное")
            
        if target_size is None:
            target_size = self.default_size
            
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        logger.debug(f"Изображение изменено с {image.shape} на {resized.shape}")
        return resized
    
    def normalize_image(self, image: np.ndarray, scale: float = 255.0) -> np.ndarray:
        """
        Нормализация изображения
        
        Args:
            image (np.ndarray): Исходное изображение
            scale (float): Масштаб нормализации
            
        Returns:
            np.ndarray: Нормализованное изображение в диапазоне [0, 1]
        """
        normalized = image.astype(np.float32) / scale
        return normalized
    
    def convert_color(self, image: np.ndarray, conversion_code: int = cv2.COLOR_BGR2RGB) -> np.ndarray:
        """
        Конвертация цветового пространства изображения
        
        Args:
            image (np.ndarray): Исходное изображение
            conversion_code (int): Код конвертации (cv2.COLOR_*)
            
        Returns:
            np.ndarray: Изображение в новом цветовом пространстве
        """
        if image is None or image.size == 0:
            raise ValueError("Входное изображение пустое или некорректное")
            
        converted = cv2.cvtColor(image, conversion_code)
        return converted
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
        для улучшения контраста
        
        Args:
            image (np.ndarray): Исходное изображение
            clip_limit (float): Предел контраста для CLAHE
            grid_size (Tuple[int, int]): Размер сетки для CLAHE
            
        Returns:
            np.ndarray: Изображение с улучшенным контрастом
        """
        # Проверяем, цветное ли изображение
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Конвертируем в LAB для обработки только канала яркости
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Применяем CLAHE только к каналу яркости
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_clahe = clahe.apply(l)
            
            # Объединяем каналы обратно
            lab_clahe = cv2.merge((l_clahe, a, b))
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            # Если изображение одноканальное (оттенки серого)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            result = clahe.apply(image)
            
        return result
    
    def denoise_image(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Уменьшение шума на изображении
        
        Args:
            image (np.ndarray): Исходное изображение
            strength (int): Сила шумоподавления
            
        Returns:
            np.ndarray: Изображение с уменьшенным шумом
        """
        if len(image.shape) == 3:
            # Цветное изображение
            denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            # Одноканальное изображение
            denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
            
        return denoised
    
    def sharpen_image(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        Увеличение резкости изображения
        
        Args:
            image (np.ndarray): Исходное изображение
            alpha (float): Коэффициент резкости
            
        Returns:
            np.ndarray: Изображение с повышенной резкостью
        """
        # Создаем ядро для увеличения резкости
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
                           
        # Применяем фильтр свертки с ядром
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Комбинируем с оригинальным изображением для регулирования силы эффекта
        result = cv2.addWeighted(image, 1.0, sharpened, alpha - 1.0, 0)
        
        return result
    
    def preprocess_for_detection(self, image: Union[str, np.ndarray], 
                                target_size: Optional[Tuple[int, int]] = None,
                                normalize: bool = True,
                                to_rgb: bool = True,
                                enhance_contrast: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Комплексная предобработка изображения для детекции
        
        Args:
            image (Union[str, np.ndarray]): Путь к изображению или само изображение
            target_size (Tuple[int, int], optional): Целевой размер (ширина, высота)
            normalize (bool): Нормализовать ли изображение
            to_rgb (bool): Конвертировать ли в RGB
            enhance_contrast (bool): Улучшать ли контраст
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (Предобработанное изображение, Информация о предобработке)
        """
        # Загружаем изображение, если передан путь
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Сохраняем информацию о размерах исходного изображения
        original_height, original_width = image.shape[:2]
        preprocess_info = {
            "original_height": original_height,
            "original_width": original_width,
        }
        
        # Улучшаем контраст, если требуется
        if enhance_contrast:
            image = self.apply_clahe(image)
        
        # Изменяем размер изображения
        if target_size is None:
            target_size = self.default_size
            
        resized = self.resize_image(image, target_size)
        preprocess_info["target_height"] = target_size[1]
        preprocess_info["target_width"] = target_size[0]
        
        # Конвертируем в RGB, если требуется
        if to_rgb:
            resized = self.convert_color(resized, cv2.COLOR_BGR2RGB)
        
        # Нормализуем изображение, если требуется
        if normalize:
            processed = self.normalize_image(resized)
        else:
            processed = resized
        
        # Меняем формат для совместимости с нейронными сетями (NCHW)
        if len(processed.shape) == 3:
            # HWC -> CHW
            processed = processed.transpose((2, 0, 1))
            # Добавляем размерность батча: CHW -> NCHW
            processed = np.expand_dims(processed, axis=0)
        else:
            # HW -> 1HW
            processed = np.expand_dims(processed, axis=0)
            # 1HW -> NCHW (с одним каналом)
            processed = np.expand_dims(processed, axis=0)
        
        return processed, preprocess_info
    
    def scale_box_coordinates(self, 
                             boxes: List[List[float]], 
                             preprocess_info: Dict[str, Any]) -> List[List[int]]:
        """
        Масштабирование координат ограничивающих рамок обратно к исходному размеру изображения
        
        Args:
            boxes (List[List[float]]): Список боксов с координатами [x1, y1, x2, y2]
            preprocess_info (Dict[str, Any]): Информация о предобработке
            
        Returns:
            List[List[int]]: Масштабированные координаты боксов
        """
        orig_height = preprocess_info["original_height"]
        orig_width = preprocess_info["original_width"]
        target_height = preprocess_info["target_height"]
        target_width = preprocess_info["target_width"]
        
        scale_x = orig_width / target_width
        scale_y = orig_height / target_height
        
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Масштабируем координаты
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y)
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y)
            
            # Ограничиваем координаты размерами изображения
            scaled_x1 = max(0, min(scaled_x1, orig_width - 1))
            scaled_y1 = max(0, min(scaled_y1, orig_height - 1))
            scaled_x2 = max(0, min(scaled_x2, orig_width - 1))
            scaled_y2 = max(0, min(scaled_y2, orig_height - 1))
            
            scaled_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
            
        return scaled_boxes
    
    def draw_boxes(self, 
                  image: np.ndarray, 
                  boxes: List[List[int]],
                  labels: Optional[List[str]] = None,
                  confidences: Optional[List[float]] = None, 
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2,
                  font_scale: float = 0.5) -> np.ndarray:
        """
        Отрисовка ограничивающих рамок на изображении
        
        Args:
            image (np.ndarray): Исходное изображение
            boxes (List[List[int]]): Список боксов с координатами [x1, y1, x2, y2]
            labels (List[str], optional): Список меток для каждого бокса
            confidences (List[float], optional): Список значений уверенности для каждого бокса
            color (Tuple[int, int, int]): Цвет рамки в формате BGR
            thickness (int): Толщина линии рамки
            font_scale (float): Масштаб шрифта для текста
            
        Returns:
            np.ndarray: Изображение с отрисованными боксами
        """
        # Создаем копию изображения для отрисовки
        result = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Рисуем прямоугольник
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Если есть метки, добавляем их
            if labels is not None and i < len(labels):
                label = labels[i]
                
                # Добавляем уверенность, если она доступна
                if confidences is not None and i < len(confidences):
                    label = f"{label}: {confidences[i]:.2f}"
                
                # Рисуем фон для текста
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                
                # Рисуем текст
                cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), thickness)
        
        return result
    
    def crop_image(self, image: np.ndarray, box: List[int]) -> np.ndarray:
        """
        Вырезание части изображения по координатам бокса
        
        Args:
            image (np.ndarray): Исходное изображение
            box (List[int]): Координаты бокса [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Вырезанная часть изображения
        """
        x1, y1, x2, y2 = box
        
        # Проверяем границы
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Вырезаем часть изображения
        cropped = image[y1:y2, x1:x2]
        
        return cropped 