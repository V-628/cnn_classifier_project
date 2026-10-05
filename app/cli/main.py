from pathlib import Path

from app.cli import interface
from packages.core.dataset import generate_dataset
from packages.core.inference import predict
from packages.core.model import DEFAULT_MODEL_PATH
from packages.core.training import train

DATASET_DIR = Path("dataset")

_PRETTY = {
    "circles": "Круг",
    "rectangles": "Прямоугольник",
    "triangles": "Треугольник",
}


def ensure_dataset_and_model(
    dataset_dir: Path = DATASET_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    epochs: int = 10,
) -> None:
    if not dataset_dir.exists():
        print("Датасет не найден, генерирую...")
        generate_dataset(output_folder=dataset_dir)
    else:
        print(f"Датасет уже есть: {dataset_dir.resolve()}")

    if not model_path.exists():
        print("Модель не найдена, обучаю...")
        train(epochs=epochs, data_dir=dataset_dir, model_path=model_path)
    else:
        print(f"Модель найдена: {model_path.resolve()}")


def on_recognize(image_rgb):
    name, probs = predict(image_rgb)
    print("\n=== Результат ===")
    print(f"Фигура: {_PRETTY.get(name, name)}")
    for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {_PRETTY.get(c, c):>15}: {p*100:5.1f}%")


def main() -> None:
    ensure_dataset_and_model()
    print("\nОткрываю окно рисования.")
    print("  ЛКМ — рисовать, Enter — распознать, C — очистить.")
    interface.run(on_recognize)


if __name__ == "__main__":
    main()