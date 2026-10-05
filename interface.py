# interface.py
import matplotlib.pyplot as plt
import numpy as np

# 1. Настройки холста
W, H = 280, 280  # ближе к размеру, который ждёт модель (например, 128)
img = np.ones((H, W, 3), dtype=np.uint8) * 255

# 2. Предрасчёт кисти
R = 6
brush_side = 2 * R + 1
by, bx = np.ogrid[-R:R + 1, -R:R + 1]
brush_mask = by**2 + bx**2 <= R**2
brush_color = np.array([0, 0, 0], dtype=np.uint8)  # чёрная кисть

fig, ax = plt.subplots()
im = ax.imshow(img, animated=True)
ax.set_title("Нарисуйте фигуру и нажмите Enter")
ax.set_xticks([]); ax.set_yticks([])

bg = None
last_point = None  # для интерполяции


def on_draw(event):
    global bg
    bg = fig.canvas.copy_from_bbox(ax.bbox)


fig.canvas.mpl_connect("draw_event", on_draw)


def stamp(cx, cy):
    """Ставит отпечаток кисти в точке (cx, cy) на img."""
    y1, y2 = max(0, cy - R), min(H, cy + R + 1)
    x1, x2 = max(0, cx - R), min(W, cx + R + 1)
    by1, by2 = y1 - cy + R, y2 - cy + R
    bx1, bx2 = x1 - cx + R, x2 - cx + R
    mask = brush_mask[by1:by2, bx1:bx2]
    img[y1:y2, x1:x2][mask] = brush_color


def redraw():
    if bg is not None:
        fig.canvas.restore_region(bg)
        im.set_data(img)
        ax.draw_artist(im)
        fig.canvas.blit(ax.bbox)
    else:
        fig.canvas.draw_idle()


def draw_on_move(event):
    global last_point
    if event.button != 1 or event.xdata is None or event.ydata is None:
        return

    x, y = int(event.xdata), int(event.ydata)

    # Интерполяция между предыдущей и текущей точкой
    if last_point is not None:
        x0, y0 = last_point
        dist = max(abs(x - x0), abs(y - y0))
        for i in range(1, dist + 1):
            t = i / dist
            stamp(int(x0 + (x - x0) * t), int(y0 + (y - y0) * t))
    else:
        stamp(x, y)

    last_point = (x, y)
    redraw()


def on_release(event):
    global last_point
    last_point = None


def on_key(event):
    """Enter — распознать, C — очистить."""
    if event.key == "enter":
        recognize_callback(img.copy())
    elif event.key == "c":
        img[:] = 255
        redraw()


fig.canvas.mpl_connect("motion_notify_event", draw_on_move)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)

recognize_callback = lambda x: None  # будет задан извне


def run(on_recognize):
    """Запуск интерфейса. on_recognize(image_rgb) вызывается при Enter."""
    global recognize_callback
    recognize_callback = on_recognize
    plt.show()