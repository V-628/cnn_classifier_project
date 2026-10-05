import numpy as np
import pytest

from packages.core.dataset import (
    DEFAULT_IMAGE_SIZE,
    create_white_image,
    draw_circle,
    draw_rectangle,
    draw_triangle,
)


@pytest.mark.parametrize("drawer", [draw_circle, draw_triangle, draw_rectangle])
def test_drawer_puts_black_pixels_on_white(drawer):
    img = create_white_image(DEFAULT_IMAGE_SIZE)
    result = drawer(img, DEFAULT_IMAGE_SIZE)
    assert np.any(result == 0), "shape must leave black pixels"
    assert np.any(result == 255), "background must remain"


def test_white_image_shape_and_color():
    img = create_white_image(32)
    assert img.shape == (32, 32)
    assert np.all(img == 255)


def test_white_image_default_size():
    img = create_white_image()
    assert img.shape == (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)