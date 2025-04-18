# Модели для проекта PC AI

В этой директории должны находиться обученные модели для детекции элементов интерфейса.

## Требуемые модели

1. **detector.onnx** - основная модель детекции элементов интерфейса (формат ONNX)
2. **detector.pt** - модель в формате PyTorch (опционально)

## Получение моделей

Существует несколько способов получить необходимые модели:

### Способ 1: Использование предобученной модели

1. Скачайте предобученную модель YOLOv8 с официального репозитория:
   ```bash
   pip install ultralytics
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
   ```

2. Переместите полученный файл `yolov8n.onnx` в эту директорию и переименуйте в `detector.onnx`

### Способ 2: Обучение собственной модели

1. Подготовьте датасет с аннотациями GUI элементов в формате YOLO:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

2. Создайте файл конфигурации `data.yaml`:
   ```yaml
   path: ./dataset
   train: train/images
   val: val/images
   
   nc: 11
   names: ['button', 'text_field', 'checkbox', 'radio', 'dropdown', 
           'menu', 'icon', 'tab', 'scrollbar', 'window', 'dialog']
   ```

3. Обучите модель:
   ```bash
   pip install ultralytics
   yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
   ```

4. Конвертируйте модель в ONNX:
   ```bash
   yolo export model=runs/detect/train/weights/best.pt format=onnx
   ```

5. Переместите полученный файл в эту директорию и переименуйте в `detector.onnx`

## Структура модели

Модель должна обнаруживать следующие классы элементов интерфейса:
1. button - кнопки
2. text_field - текстовые поля
3. checkbox - флажки
4. radio - радиокнопки
5. dropdown - выпадающие списки
6. menu - меню
7. icon - иконки
8. tab - вкладки
9. scrollbar - полосы прокрутки
10. window - окна
11. dialog - диалоговые окна

## Проверка модели

Чтобы проверить работу модели, выполните:

```python
from src.detector import Detector
import cv2

# Инициализация детектора
detector = Detector('models/detector.onnx')

# Загрузка тестового изображения
image = cv2.imread('test_screenshot.png')

# Обнаружение элементов
elements = detector.detect_elements(image)
print(f"Обнаружено элементов: {len(elements)}")
for elem in elements:
    print(f"Класс: {elem['class_name']}, Уверенность: {elem['confidence']:.2f}")
```

## Дополнительные ресурсы

- [Документация YOLO](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Создание собственного датасета для YOLO](https://docs.ultralytics.com/datasets/)

## Рекомендации

1. Для разметки данных рекомендуется использовать [Roboflow](https://roboflow.com/) или [LabelImg](https://github.com/tzutalin/labelImg)
2. Собирайте скриншоты с разными темами интерфейса (светлая/темная) для лучшего обобщения
3. Рекомендуемые классы UI элементов:
   - button
   - text_field
   - checkbox
   - radio
   - dropdown
   - menu
   - icon
   - tab
   - scrollbar
   - window
   - dialog

## Использование модели

После добавления файла модели в эту директорию укажите путь к ней при запуске системы:

```bash
python src/main.py --model models/your_model.onnx
``` 