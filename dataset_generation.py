from pathlib import Path
import random
import math

import imageio.v3 as iio
import numpy as np

from skimage.draw import (
    ellipse_perimeter,
    polygon_perimeter,
    rectangle_perimeter
)
from skimage.morphology import dilation, disk


# ==========================
# Настройки
# ==========================
OUTPUT_FOLDER = Path("dataset")

IMAGES_PER_CLASS = 1000
IMAGE_SIZE = 128

BACKGROUND = 255  # Белый фон
LINE_COLOR = 0    # Чёрная линия

MIN_SIZE = 25
MAX_SIZE = 80

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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


def create_white_image():
    """Создаёт белое чёрно-белое изображение."""
    return np.full(
        (IMAGE_SIZE, IMAGE_SIZE),
        BACKGROUND,
        dtype=np.uint8
    )


def make_line_thicker(image, thickness):
    """
    Увеличивает толщину чёрного контура.

    skimage.draw создаёт контур толщиной 1 пиксель.
    dilation расширяет чёрные пиксели.
    """
    black_pixels = image == LINE_COLOR

    thicker_black_pixels = dilation(
        black_pixels,
        disk(thickness)
    )

    image[thicker_black_pixels] = LINE_COLOR

    return image


def get_center_and_size():
    """Выбирает случайные размер и положение фигуры."""
    size = random.randint(MIN_SIZE, MAX_SIZE)

    # Динамический отступ: половина размера фигуры плюс небольшой буфер
    margin = size // 2 + 5 

    # Защитная проверка: если margin сожрал всё пространство, 
    # сдвигаем границы к центру, чтобы randint не упал
    min_x = margin
    max_x = IMAGE_SIZE - margin
    if min_x >= max_x:
        min_x = IMAGE_SIZE // 4
        max_x = (IMAGE_SIZE // 4) * 3

    center_x = random.randint(min_x, max_x)
    center_y = random.randint(min_x, max_x)

    return center_x, center_y, size


# ==========================
# Рисование фигур
# ==========================
def draw_circle(image):
    """Рисует контур круга."""
    center_x, center_y, size = get_center_and_size()

    radius = size // 2

    rr, cc = ellipse_perimeter(
        center_y,
        center_x,
        radius,
        radius,
        shape=image.shape
    )

    image[rr, cc] = LINE_COLOR

    thickness = random.choice([1, 2, 3])
    return make_line_thicker(image, thickness)


def draw_triangle(image):
    """Рисует контур треугольника со случайными углами."""
    center_x, center_y, size = get_center_and_size()

    # Нижние вершины.
    # Из-за разных значений ширины основание каждый раз другое.
    left_x = center_x - random.uniform(size * 0.25, size * 0.60)
    left_y = center_y + random.uniform(size * 0.15, size * 0.45)

    right_x = center_x + random.uniform(size * 0.25, size * 0.60)
    right_y = center_y + random.uniform(size * 0.15, size * 0.45)

    # Верхняя вершина может быть не по центру.
    # Поэтому треугольники получаются разной формы.
    top_x = center_x + random.uniform(-size * 0.35, size * 0.35)
    top_y = center_y - random.uniform(size * 0.25, size * 0.65)

    # polygon_perimeter использует:
    # rr — строки, то есть y
    # cc — столбцы, то есть x
    rows = np.array([
        top_y,
        left_y,
        right_y
    ], dtype=int)

    columns = np.array([
        top_x,
        left_x,
        right_x
    ], dtype=int)

    rr, cc = polygon_perimeter(
        rows,
        columns,
        shape=image.shape,
        clip=True
    )

    image[rr, cc] = LINE_COLOR

    thickness = random.choice([1, 2, 3])

    return make_line_thicker(image, thickness)

def draw_rectangle(image):
    """Рисует контур прямоугольника."""
    center_x, center_y, size = get_center_and_size()

    width = size
    height = random.randint(
        int(size * 0.45),
        int(size * 0.85)
    )

    # Иногда прямоугольник рисуется без поворота.
    # Это позволяет использовать rectangle_perimeter.
    if random.random() < 0.5:
        start = (
            int(center_y - height / 2),
            int(center_x - width / 2)
        )

        end = (
            int(center_y + height / 2),
            int(center_x + width / 2)
        )

        rr, cc = rectangle_perimeter(
            start=start,
            end=end,
            shape=image.shape,
            clip=True
        )

    else:
        # Поворотный прямоугольник строится как контур многоугольника.
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

        rr, cc = polygon_perimeter(
            rows,
            columns,
            shape=image.shape,
            clip=True
        )

    image[rr, cc] = LINE_COLOR

    thickness = random.choice([1, 2, 3])
    return make_line_thicker(image, thickness)


# ==========================
# Генерация файлов
# ==========================
def generate_class(folder_name, draw_function):
    """Создаёт 1000 изображений для одного класса."""
    folder = OUTPUT_FOLDER / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    for index in range(IMAGES_PER_CLASS):
        image = create_white_image()

        image = draw_function(image)

        filename = f"{folder_name}_{index + 1:04d}.png"
        filepath = folder / filename

        iio.imwrite(filepath, image)

        if (index + 1) % 100 == 0:
            print(f"{folder_name}: {index + 1}/{IMAGES_PER_CLASS}")


def main_generation():
    print("Создание датасета...")

    generate_class("triangles", draw_triangle)
    generate_class("circles", draw_circle)
    generate_class("rectangles", draw_rectangle)

    print("\nГотово.")
    print(f"Папка: {OUTPUT_FOLDER.resolve()}")
    print(f"Создано файлов: {IMAGES_PER_CLASS * 3}")


if __name__ == "__main__":
    main_generation()