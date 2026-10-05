# main.py
from pathlib import Path
from dataset_generation import main_generation, OUTPUT_FOLDER
from train import train, MODEL_PATH
import interface


def ensure_dataset_and_model():
    # Если датасета нет — генерируем
    if not OUTPUT_FOLDER.exists():
        print("Датасет не найден, генерирую...")
        main_generation()
    else:
        print(f"Датасет уже есть: {OUTPUT_FOLDER.resolve()}")

    # Если модели нет — обучаем
    if not MODEL_PATH.exists():
        print("Модель не найдена, обучаю...")
        train(epochs=10)
    else:
        print(f"Модель найдена: {MODEL_PATH.resolve()}")


def on_recognize(image_rgb):
    from predict import predict
    name, probs = predict(image_rgb)
    pretty = {"circles": "Круг", "rectangles": "Прямоугольник", "triangles": "Треугольник"}
    print("\n=== Результат ===")
    print(f"Фигура: {pretty.get(name, name)}")
    for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {pretty.get(c, c):>15}: {p*100:5.1f}%")


def main():
    ensure_dataset_and_model()

    print("\nОткрываю окно рисования.")
    print("  ЛКМ — рисовать, Enter — распознать, C — очистить.")
    interface.run(on_recognize)


if __name__ == "__main__":
    main()