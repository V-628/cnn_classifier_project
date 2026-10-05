import numpy as np
import torch
from torchvision import transforms

from packages.core.model import load_model, DEVICE

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def predict(image_rgb: np.ndarray, model=None, classes=None):
    """
    image_rgb: np.uint8 HxWx3 (как в app/cli/interface.py).
    Возвращает (class_name, probabilities_dict).
    """
    if model is None:
        model, classes = load_model()

    x = _transform(image_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    return classes[idx], {c: float(p) for c, p in zip(classes, probs)}