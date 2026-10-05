"""Smoke tests for current behaviour after refactoring."""
import importlib


def test_main_module_importable():
    mod = importlib.import_module("main")
    assert hasattr(mod, "main")


def test_app_entry_importable():
    from app.cli.main import main, on_recognize, ensure_dataset_and_model
    assert callable(main)
    assert callable(on_recognize)
    assert callable(ensure_dataset_and_model)


def test_dataset_module_importable():
    from packages.core.dataset import generate_dataset
    assert callable(generate_dataset)


def test_training_module_importable():
    from packages.core.training import train, get_loaders
    assert callable(train)
    assert callable(get_loaders)


def test_inference_module_importable():
    from packages.core.inference import predict
    assert callable(predict)


def test_model_module_importable():
    from packages.core.model import ShapeCNN, load_model, save_model, CLASSES
    assert CLASSES == ["circles", "rectangles", "triangles"]