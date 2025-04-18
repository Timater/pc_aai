import pyautogui
import time
import logging

class ActionManager:
    """
    Класс для управления действиями мыши и клавиатуры
    """
    def __init__(self, delay=0.5, move_duration=0.3):
        """
        Инициализация менеджера действий
        
        Args:
            delay (float): Задержка перед кликом
            move_duration (float): Время перемещения курсора
        """
        self.delay = delay
        self.move_duration = move_duration
        self.logger = logging.getLogger(__name__)
        
        # Настройка безопасности PyAutoGUI
        pyautogui.FAILSAFE = True
        
    def click(self, x, y, button='left', clicks=1):
        """
        Клик по координатам
        
        Args:
            x (int): Координата x
            y (int): Координата y
            button (str): Кнопка мыши ('left', 'right', 'middle')
            clicks (int): Количество кликов
        """
        try:
            self.logger.info(f"Перемещение курсора в позицию ({x}, {y})")
            pyautogui.moveTo(x, y, duration=self.move_duration)
            time.sleep(self.delay)
            
            self.logger.info(f"Клик кнопкой {button}, количество кликов: {clicks}")
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при клике: {str(e)}")
            return False
    
    def double_click(self, x, y):
        """
        Двойной клик по координатам
        
        Args:
            x (int): Координата x
            y (int): Координата y
        """
        return self.click(x, y, clicks=2)
    
    def right_click(self, x, y):
        """
        Правый клик по координатам
        
        Args:
            x (int): Координата x
            y (int): Координата y
        """
        return self.click(x, y, button='right')
    
    def type_text(self, text, interval=0.1):
        """
        Ввод текста
        
        Args:
            text (str): Текст для ввода
            interval (float): Интервал между нажатиями клавиш
        """
        try:
            self.logger.info(f"Ввод текста: {text}")
            pyautogui.write(text, interval=interval)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при вводе текста: {str(e)}")
            return False
    
    def press_key(self, key):
        """
        Нажатие клавиши
        
        Args:
            key (str): Название клавиши
        """
        try:
            self.logger.info(f"Нажатие клавиши: {key}")
            pyautogui.press(key)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при нажатии клавиши: {str(e)}")
            return False
    
    def hotkey(self, *keys):
        """
        Комбинация клавиш
        
        Args:
            *keys: Список клавиш
        """
        try:
            self.logger.info(f"Комбинация клавиш: {keys}")
            pyautogui.hotkey(*keys)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при комбинации клавиш: {str(e)}")
            return False
    
    def scroll(self, clicks):
        """
        Прокрутка колеса мыши
        
        Args:
            clicks (int): Количество щелчков колеса (отрицательные значения - вниз)
        """
        try:
            self.logger.info(f"Прокрутка: {clicks}")
            pyautogui.scroll(clicks)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при прокрутке: {str(e)}")
            return False
    
    def drag_to(self, x1, y1, x2, y2, duration=0.5):
        """
        Перетаскивание мышью
        
        Args:
            x1 (int): Начальная координата x
            y1 (int): Начальная координата y
            x2 (int): Конечная координата x
            y2 (int): Конечная координата y
            duration (float): Продолжительность перетаскивания
        """
        try:
            self.logger.info(f"Перетаскивание с ({x1}, {y1}) в ({x2}, {y2})")
            pyautogui.moveTo(x1, y1, duration=self.move_duration)
            pyautogui.dragTo(x2, y2, duration=duration, button='left')
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при перетаскивании: {str(e)}")
            return False
    
    def screenshot(self, region=None):
        """
        Создание скриншота
        
        Args:
            region (tuple): Область для скриншота (x, y, width, height)
            
        Returns:
            PIL.Image: Объект изображения
        """
        try:
            self.logger.info("Создание скриншота")
            return pyautogui.screenshot(region=region)
        except Exception as e:
            self.logger.error(f"Ошибка при создании скриншота: {str(e)}")
            return None 