from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
from skimage.draw import (
    ellipse_perimeter,
    polygon_perimeter,
    rectangle_perimeter,
)
from skimage.morphology import dilation, disk


# ==========================
# Значения по умолчанию
# ==========================
DEFAULT_IMAGE_SIZE = 128
DEFAULT_IMAGES_PER_CLASS = 1000

BACKGROUND = 255
LINE_COLOR = 0

MIN_SIZE = 25
MAX_SIZE = 80

DEFAULT_SEED = 42


# ==========================
# Вспомогательные функции
# ==========================
def rotate_point(x, y, center_x, center_y, angle_degrees):
    """Поворачивает точку вокруг центра."""
    angle = math.radians(angle_degrees)

    x -= center_x
    y -= center_y

    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)

    return new_x + center_x, new_y + center_y


def create_white_image(image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Создаёт белое чёрно-белое изображение."""
    return np.full(
        (image_size, image_size),
        BACKGROUND,
        dtype=np.uint8,
    )


def make_line_thicker(image, thickness):
    """Увеличивает толщину чёрного контура."""
    black_pixels = image == LINE_COLOR
    thicker_black_pixels = dilation(black_pixels, disk(thickness))
    image[thicker_black_pixels] = LINE_COLOR
    return image


def get_center_and_size(image_size: int = DEFAULT_IMAGE_SIZE):
    """Выбирает случайные размер и положение фигуры."""
    size = random.randint(MIN_SIZE, MAX_SIZE)

    margin = size // 2 + 5

    min_x = margin
    max_x = image_size - margin
    if min_x >= max_x:
        min_x = image_size // 4
        max_x = (image_size // 4) * 3

    center_x = random.randint(min_x, max_x)
    center_y = random.randint(min_x, max_x)

    return center_x, center_y, size


# ==========================
# Рисование фигур
# ==========================
def draw_circle(image, image_size: int = DEFAULT_IMAGE_SIZE):
    """Рисует контур круга."""
    center_x, center_y, size = get_center_and_size(image_size)

    radius = size // 2

    rr, cc = ellipse_perimeter(
        center_y,
        center_x,
        radius,
        radius,
        shape=image.shape,
    )

    image[rr, cc] = LINE_COLOR

    thickness = random.choice([1, 2, 3])
    return make_line_thicker(image, thickness)


def draw_triangle(image, image_size: int = DEFAULT_IMAGE_SIZE):
    """Рисует контур треугольника со случайными углами."""
    center_x, center_y, size = get_center_and_size(image_size)

    left_x = center_x - random.uniform(size * 0.25, size * 0.60)
    left_y = center_y + random.uniform(size * 0.15, size * 0.45)

    right_x = center_x + random.uniform(size * 0.25, size * 0.60)
    right_y = center_y + random.uniform(size * 0.15, size * 0.45)

    top_x = center_x + random.uniform(-size * 0.35, size * 0.35)
    top_y = center_y - random.uniform(size * 0.25, size * 0.65)

    rows = np.array([top_y, left_y, right_y], dtype=int)
    columns = np.array([top_x, left_x, right_x], dtype=int)

    rr, cc = polygon_perimeter(rows, columns, shape=image.shape, clip=True)

    image[rr, cc] = LINE_COLOR

    thickness = random.choice([1, 2, 3])
    return make_line_thicker(image, thickness)


def draw_rectangle(image, image_size: int = DEFAULT_IMAGE_SIZE):
    """Рисует контур прямоугольника."""
    center_x, center_y, size = get_center_and_size(image_size)

    width = size
    height = random.randint(int(size * 0.45), int(size * 0.85))

    if random.random() < 0.5:
        start = (int(center_y - height / 2), int(center_x - width / 2))
        end = (int(center_y + height / 2), int(center_x + width / 2))

        rr, cc = rectangle_perimeter(
            start=start,
            end=end,
            shape=image.shape,
            clip=True,
        )
    else:
        points = [
            (center_x - width / 2, center_y - height / 2),
            (center_x + width / 2, center_y - height / 2),
            (center_x + width / 2, center_y + height / 2),
            (center_x - width / 2, center_y + height / 2),
        ]

        angle = random.uniform(0, 360)

        rotated_points = [
            rotate_point(x, y, center_x, center_y, angle)
            for x, y in points
        ]

        columns = np.array([point[0] for point in rotated_points])
        rows = np.array([point[1] for point in rotated_points])

        rr, cc = polygon_perimeter(rows, columns, shape=image.shape, clip=True)

    image[rr, cc] = LINE_COLOR

    thickness = random.choice([1, 2, 3])
    return make_line_thicker(image, thickness)


# ==========================
# Генерация файлов
# ==========================
def generate_class(
    folder_name: str,
    draw_function: Callable,
    output_folder: Path,
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> None:
    """Создаёт N изображений для одного класса."""
    folder = output_folder / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    for index in range(images_per_class):
        image = create_white_image(image_size)
        image = draw_function(image, image_size)

        filename = f"{folder_name}_{index + 1:04d}.png"
        iio.imwrite(folder / filename, image)

        if (index + 1) % 100 == 0:
            print(f"{folder_name}: {index + 1}/{images_per_class}")


def generate_dataset(
    output_folder: Path = Path("dataset"),
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Генерирует полный датасет из трёх классов фигур.

    Возвращает путь к папке с датасетом.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("Создание датасета...")

    generate_class("triangles", draw_triangle, output_folder, images_per_class, image_size)
    generate_class("circles", draw_circle, output_folder, images_per_class, image_size)
    generate_class("rectangles", draw_rectangle, output_folder, images_per_class, image_size)

    print("\nГотово.")
    print(f"Папка: {output_folder.resolve()}")
    print(f"Создано файлов: {images_per_class * 3}")
    return output_folder


if __name__ == "__main__":
    generate_dataset()