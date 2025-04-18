# Модели для системы управления ПК

В эту директорию следует поместить обученные модели в формате ONNX для распознавания элементов интерфейса.

## Требования к моделям

- Формат: ONNX (`.onnx`)
- Входные данные: изображение размером 640x640 пикселей, нормализованное до [0, 1]
- Выходные данные: массив предсказаний в формате YOLO [x1, y1, x2, y2, confidence, class_id]

## Поддерживаемые модели

1. **YOLO v5/v8** - модели для распознавания объектов, конвертированные в ONNX формат
2. **ViT (Vision Transformer)** - модели на основе трансформеров (требуется дополнительная обработка выходных данных)

## Пример обучения и конвертации модели YOLO

```python
# Обучение модели YOLOv8 на собственном датасете
!pip install ultralytics
from ultralytics import YOLO

# Загрузка предобученной модели
model = YOLO('yolov8n.pt')

# Обучение на собственном датасете
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='ui_elements_detector'
)

# Экспорт в ONNX формат
model.export(format='onnx')
```

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