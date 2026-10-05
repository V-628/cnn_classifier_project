"""Smoke tests for current behaviour (before refactoring)."""
import importlib
from pathlib import Path


def test_main_module_importable():
    mod = importlib.import_module("main")
    assert hasattr(mod, "main")
    assert hasattr(mod, "on_recognize")
    assert hasattr(mod, "ensure_dataset_and_model")


def test_dataset_module_importable():
    import dataset_generation as dg
    assert hasattr(dg, "main_generation")
    assert dg.OUTPUT_FOLDER == Path("dataset")


def test_train_module_importable():
    import train as tr
    assert hasattr(tr, "train")
    assert hasattr(tr, "load_model")
    assert tr.CLASSES == ["circles", "rectangles", "triangles"]


def test_predict_module_importable():
    import predict as pr
    assert hasattr(pr, "predict")