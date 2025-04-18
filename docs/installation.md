# Установка и настройка системы управления ПК

В этом документе описаны шаги по установке, настройке и запуску системы управления ПК.

## Требования к системе

- Python 3.8 или выше
- Операционная система: Windows, Linux или macOS
- Память: не менее 4 ГБ RAM
- Пространство на диске: не менее 500 МБ

## Подготовка окружения

### Windows

1. Установка Python:
   - Скачайте и установите Python с [официального сайта](https://www.python.org/downloads/)
   - Убедитесь, что опция "Add Python to PATH" отмечена во время установки
   - Проверьте установку, выполнив в командной строке:
     ```
     python --version
     ```

2. Клонирование репозитория:
   ```
   git clone https://github.com/Timater/PC_AI.git
   cd PC_AI
   ```

3. Установка зависимостей:
   ```
   pip install -r requirements.txt
   ```

### Linux

1. Установка Python и необходимых пакетов:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv git
   ```

2. Клонирование репозитория:
   ```bash
   git clone https://github.com/Timater/PC_AI.git
   cd PC_AI
   ```

3. Создание и активация виртуального окружения:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Установка зависимостей:
   ```bash
   pip install -r requirements.txt
   ```

### macOS

1. Установка Python и необходимых инструментов:
   ```bash
   # Установите Homebrew, если его нет
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Установите Python
   brew install python3 git
   ```

2. Клонирование репозитория:
   ```bash
   git clone https://github.com/Timater/PC_AI.git
   cd PC_AI
   ```

3. Создание и активация виртуального окружения:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Установка зависимостей:
   ```bash
   pip install -r requirements.txt
   ```

## Настройка модели

1. Разместите вашу ONNX модель в директории `models/`
2. Если у вас нет собственной модели, вы можете использовать одну из предобученных:
   ```bash
   # Загрузка предобученной модели YOLOv8n
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
   mv yolov8n.onnx models/
   ```

## Возможные проблемы и их решения

### Ошибка с OpenCV на Windows

Если вы получаете ошибку `ImportError: DLL load failed: The specified module could not be found.` при импорте OpenCV:

1. Проверьте версию Visual C++ Redistributable:
   ```
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

### Ошибки с PyAutoGUI на Linux

Для работы PyAutoGUI на Linux требуются дополнительные библиотеки:

```bash
sudo apt-get install python3-tk python3-dev scrot
sudo apt-get install python3-xlib
```

### Ошибки с ONNX Runtime

Если возникают проблемы с ONNX Runtime, попробуйте установить версию для CPU:

```bash
pip uninstall onnxruntime
pip install onnxruntime-cpu
``` 