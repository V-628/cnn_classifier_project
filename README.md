"# cnn_classifier_project" 
# CNN Classifier Project

Классификатор рукописных геометрических фигур (круг, треугольник, прямоугольник)
на PyTorch. Пользователь рисует фигуру мышкой, модель предсказывает класс.

## Структура

- `packages/core/` — переиспользуемое ядро (генерация датасета, модель, обучение, инференс)
- `app/cli/` — запускаемое приложение (UI + оркестрация)
- `tests/` — smoke и unit тесты
- `scripts/` — обёртки для запуска
- `docs/diagrams/` — редактируемые диаграммы (Mermaid)

## Быстрый старт

Установка зависимостей:

    scripts\setup.bat          (Windows)
    ./scripts/setup.sh         (Linux/Mac)

Запуск приложения:

    scripts\run-app.bat
    ./scripts/run-app.sh

Тесты:

    scripts\run-tests.bat
    ./scripts/run-tests.sh

Через Makefile (Linux/Mac):

    make setup
    make run
    make test
    make clean

## Что делает приложение

1. При первом запуске генерирует датасет из 3000 синтетических изображений
   (по 1000 на класс) в `dataset/`.
2. Обучает свёрточную сеть `ShapeCNN` и сохраняет веса в `model.pth`.
3. Открывает окно рисования: ЛКМ — рисовать, Enter — распознать, C — очистить.

## Архитектура

См. `docs/diagrams/ml-pipeline.mmd`.

## Зависимости

`requirements.txt`. Основные: torch, torchvision, scikit-image, imageio, matplotlib, numpy.