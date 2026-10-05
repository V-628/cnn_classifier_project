import numpy as np
import pytest

from packages.core.inference import predict
from packages.core.model import ShapeCNN


def test_predict_returns_class_and_probs():
    model = ShapeCNN(num_classes=3).eval()
    classes = ["circles", "rectangles", "triangles"]
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

    name, probs = predict(img, model=model, classes=classes)

    assert name in classes
    assert set(probs) == set(classes)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_probs_non_negative():
    model = ShapeCNN(num_classes=3).eval()
    classes = ["circles", "rectangles", "triangles"]
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

    _, probs = predict(img, model=model, classes=classes)
    assert all(p >= 0 for p in probs.values())