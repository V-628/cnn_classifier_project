from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from packages.core.model import DEVICE, DEFAULT_MODEL_PATH, ShapeCNN, save_model

DATA_DIR = Path("dataset")


def get_loaders(data_dir=DATA_DIR, batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full = datasets.ImageFolder(data_dir, transform=transform)
    n_val = int(0.2 * len(full))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, full.classes


def train(epochs=10, lr=1e-3, data_dir=DATA_DIR, model_path=DEFAULT_MODEL_PATH):
    train_loader, val_loader, classes = get_loaders(data_dir)
    print("Классы:", classes)

    model = ShapeCNN(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()

        acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch:2d} | loss={total_loss/len(train_loader.dataset):.4f} | val_acc={acc:.4f}")

    save_model(model, classes, model_path)
    print(f"Модель сохранена: {model_path}")
    return model, classes


if __name__ == "__main__":
    train()