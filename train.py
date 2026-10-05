# Устанавливаем фреймворки
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# Задаем константы для дальнейшей работы
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Проверяем и выбираем на чем обучать
DATA_DIR = Path("dataset")
MODEL_PATH = Path("model.pth")
CLASSES = ["circles", "rectangles", "triangles"]

# Создаем класс nn.Module - база любых нейронных сетей
class ShapeCNN(nn.Module):
    
    # Инициализируем класс и передаем количесво
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders(batch_size=64):
    # Ч/б, нормализация
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full = datasets.ImageFolder(DATA_DIR, transform=transform)
    n_val = int(0.2 * len(full))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, full.classes


def train(epochs=10, lr=1e-3):
    train_loader, val_loader, classes = get_loaders()
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
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        # Валидация
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()

        acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch:2d} | loss={total_loss/len(train_loader.dataset):.4f} | val_acc={acc:.4f}")

    torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")
    return model, classes


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = ShapeCNN(num_classes=len(ckpt["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]


if __name__ == "__main__":
    train()