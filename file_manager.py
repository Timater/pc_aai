import os
import shutil
import glob
import logging
import fnmatch
from pathlib import Path

class FileManager:
    """
    Класс для работы с файловой системой
    """
    def __init__(self):
        """
        Инициализация менеджера файлов
        """
        self.logger = logging.getLogger(__name__)
        self.desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        self.documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        self.downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
    def find_file(self, filename, search_path=None):
        """
        Поиск файла по имени в указанной директории и её подпапках
        
        Args:
            filename (str): Имя файла для поиска
            search_path (str, optional): Путь для поиска. По умолчанию используется текущий каталог.
            
        Returns:
            str: Полный путь к найденному файлу или None, если не найден
        """
        if search_path is None:
            search_path = os.getcwd()
        
        self.logger.info(f"Поиск файла '{filename}' в '{search_path}'")
        
        try:
            for root, dirs, files in os.walk(search_path):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    self.logger.info(f"Файл найден: {found_path}")
                    return found_path
                
            # Если файл не найден по точному имени, ищем с использованием шаблона
            pattern = f"*{filename}*"
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if fnmatch.fnmatch(file.lower(), pattern.lower()):
                        found_path = os.path.join(root, file)
                        self.logger.info(f"Файл найден (по шаблону): {found_path}")
                        return found_path
                        
            self.logger.warning(f"Файл '{filename}' не найден")
            return None
        except Exception as e:
            self.logger.error(f"Ошибка при поиске файла: {str(e)}")
            return None
    
    def create_folder(self, folder_path):
        """
        Создание папки
        
        Args:
            folder_path (str): Путь к создаваемой папке
            
        Returns:
            bool: True, если папка создана успешно, иначе False
        """
        try:
            self.logger.info(f"Создание папки: {folder_path}")
            os.makedirs(folder_path, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при создании папки: {str(e)}")
            return False
    
    def delete_file(self, file_path):
        """
        Удаление файла
        
        Args:
            file_path (str): Путь к удаляемому файлу
            
        Returns:
            bool: True, если файл удален успешно, иначе False
        """
        try:
            if os.path.exists(file_path):
                self.logger.info(f"Удаление файла: {file_path}")
                os.remove(file_path)
                return True
            else:
                self.logger.warning(f"Файл не существует: {file_path}")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при удалении файла: {str(e)}")
            return False
    
    def delete_folder(self, folder_path):
        """
        Удаление папки
        
        Args:
            folder_path (str): Путь к удаляемой папке
            
        Returns:
            bool: True, если папка удалена успешно, иначе False
        """
        try:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                self.logger.info(f"Удаление папки: {folder_path}")
                shutil.rmtree(folder_path)
                return True
            else:
                self.logger.warning(f"Папка не существует: {folder_path}")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при удалении папки: {str(e)}")
            return False
    
    def move_file(self, source, destination):
        """
        Перемещение файла
        
        Args:
            source (str): Исходный путь файла
            destination (str): Путь назначения
            
        Returns:
            bool: True, если файл перемещен успешно, иначе False
        """
        try:
            if os.path.exists(source):
                self.logger.info(f"Перемещение файла из {source} в {destination}")
                shutil.move(source, destination)
                return True
            else:
                self.logger.warning(f"Исходный файл не существует: {source}")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при перемещении файла: {str(e)}")
            return False
    
    def copy_file(self, source, destination):
        """
        Копирование файла
        
        Args:
            source (str): Исходный путь файла
            destination (str): Путь назначения
            
        Returns:
            bool: True, если файл скопирован успешно, иначе False
        """
        try:
            if os.path.exists(source):
                self.logger.info(f"Копирование файла из {source} в {destination}")
                shutil.copy2(source, destination)
                return True
            else:
                self.logger.warning(f"Исходный файл не существует: {source}")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка при копировании файла: {str(e)}")
            return False
    
    def list_files(self, directory, pattern=None):
        """
        Получение списка файлов в директории
        
        Args:
            directory (str): Путь к директории
            pattern (str, optional): Шаблон для фильтрации файлов
            
        Returns:
            list: Список файлов в директории
        """
        try:
            if not os.path.exists(directory):
                self.logger.warning(f"Директория не существует: {directory}")
                return []
            
            if pattern:
                self.logger.info(f"Получение списка файлов в {directory} по шаблону {pattern}")
                return glob.glob(os.path.join(directory, pattern))
            else:
                self.logger.info(f"Получение списка файлов в {directory}")
                return [os.path.join(directory, f) for f in os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, f))]
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка файлов: {str(e)}")
            return []
    
    def create_file(self, file_path, content=""):
        """
        Создание файла с указанным содержимым
        
        Args:
            file_path (str): Путь к создаваемому файлу
            content (str, optional): Содержимое файла
            
        Returns:
            bool: True, если файл создан успешно, иначе False
        """
        try:
            self.logger.info(f"Создание файла: {file_path}")
            directory = os.path.dirname(file_path)
            
            # Создаем директорию, если она не существует
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при создании файла: {str(e)}")
            return False
    
    def read_file(self, file_path):
        """
        Чтение содержимого файла
        
        Args:
            file_path (str): Путь к файлу
            
        Returns:
            str: Содержимое файла или None в случае ошибки
        """
        try:
            if os.path.exists(file_path):
                self.logger.info(f"Чтение файла: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                self.logger.warning(f"Файл не существует: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Ошибка при чтении файла: {str(e)}")
            return None
    
    def get_default_path(self, folder_type="desktop"):
        """
        Получение пути к стандартным системным папкам
        
        Args:
            folder_type (str): Тип папки (desktop, documents, downloads)
            
        Returns:
            str: Путь к папке
        """
        if folder_type.lower() == "desktop":
            return self.desktop_path
        elif folder_type.lower() == "documents":
            return self.documents_path
        elif folder_type.lower() == "downloads":
            return self.downloads_path
        else:
            return None 