import torch

from packages.core.model import ShapeCNN, CLASSES


def test_model_forward_output_shape():
    model = ShapeCNN(num_classes=3).eval()
    x = torch.randn(4, 1, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 3)


def test_classes_constant():
    assert CLASSES == ["circles", "rectangles", "triangles"]