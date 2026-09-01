# -*- coding: utf-8 -*-
"""Проверка объёмного вида В ОКНЕ. Запускать ОТДЕЛЬНО, из корня проекта:

    py -3.9 editor/check_gl_window.py

Почему отдельно, а не в check_gui.py. Там к этому месту живут сразу несколько контекстов
OpenGL — закадровые рисовальщики из других проверок, — и чтение кадра ИЗ ОКНА перестаёт быть
надёжным: читается пустота, хотя нарисовано верно. Признак такой беды приметный — расхождение
с образцом и с его зеркалом выходит ОДИНАКОВЫМ (перевернуть однотонное поле нельзя), и на него
уже дважды удавалось поймать себя за руку. В чистом процессе всё честно.

Что здесь проверяется — ориентация. Отказ тихий и виден только глазом: карта встаёт вверх
ногами. Вся отрисовка живёт в координатах PIL (строка 0 сверху); закадровый путь мирит это с
OpenGL при чтении кадра, а окну читать нечего, и переворот делается в самом конце —
четырёхугольником на весь экран. Матрицей его сделать НЕЛЬЗЯ: небо рисуется прямо в клипе и
матрице не подчиняется, местность перевернётся, а градиент останется.
"""
import os
import sys
import tkinter as tk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import view3d                                                # noqa: E402
import view3d_gpu                                            # noqa: E402

FAILED = []


def ok(cond, what):
    print(("  ok  " if cond else "ПРОВАЛ ") + what)
    if not cond:
        FAILED.append(what)


def main():
    try:
        from pyopengltk import OpenGLFrame
    except Exception as ex:                                  # noqa: BLE001
        print("pyopengltk не поставлен (%s) — объём в окне не проверяется, "
              "редактор работает через холст" % ex)
        return 0
    import moderngl

    W, H, G = 400, 300, 48
    # НЕСИММЕТРИЧНАЯ карта: лес в одном углу, вода в противоположном. На симметричной
    # перевёрнутый кадр от правильного не отличить.
    mark = np.zeros((G, G), dtype=np.int32)
    mark[:G // 3, :G // 2] = 1
    mark[-G // 4:, -G // 3:] = 3
    flat = np.zeros((G, G), dtype=np.float32)
    cell = 2550.0 / G
    cam = view3d.Camera(target=(1275.0, 1275.0), dist=5200.0, yaw=0.0, pitch=30.0)
    draw = [(0.0, 0.0, 2550.0, 2550.0, 0.0, 0.0, 1.0, 1.0, "t", 48)]
    tile = view3d.surface_rgb(mark).transpose(1, 0, 2).astype(np.uint8)

    # образец — закадровый путь: он сверяется с программным рисовальщиком в check_gui
    off = view3d_gpu.GLView((W, H), samples=4)
    off.set_height(flat, cell, key="проба")
    off.upload_tile("t", tile)
    ref = np.asarray(off.frame(cam, draw, size=(W, H), ground=True)).astype(np.int16)
    ok(float(np.abs(ref - ref.mean()).mean()) > 2.0, "образец нарисован, а не однотонен")

    state = {}

    class View(OpenGLFrame):
        def initgl(self):
            if "gl" not in state:
                gl = view3d_gpu.GLView((W, H), samples=4, ctx=moderngl.create_context())
                gl.set_height(flat, cell, key="проба")
                gl.upload_tile("t", tile)
                state["gl"] = gl
            state["gl"].ctx.viewport = (0, 0, self.width, self.height)

        def redraw(self):
            gl = state["gl"]
            gl.frame(cam, draw, size=(W, H), ground=True)
            fb = gl._screen()
            raw = np.frombuffer(fb.read(components=3), dtype=np.uint8)
            # OpenGL отдаёт строки снизу вверх — переворачиваем к виду PIL
            state["shot"] = raw.reshape(fb.size[1], fb.size[0], 3)[::-1].astype(np.int16)

    root = tk.Tk()
    root.geometry("%dx%d" % (W + 20, H + 20))
    view = View(root, width=W, height=H)
    view.pack()
    root.update()
    view.animate = 0
    for _ in range(8):
        view.tkExpose(None)
        root.update()

    shot = state.get("shot")
    ok(shot is not None, "кадр из окна снят")
    if shot is not None:
        ok(float(np.abs(shot - shot.mean()).mean()) > 2.0,
           "кадр из окна не однотонен (иначе читаем не то, что нарисовали)")
        a = shot[:ref.shape[0], :ref.shape[1]]
        b = ref[:a.shape[0], :a.shape[1]]
        same = float(np.abs(a - b).mean())
        flip = float(np.abs(a[::-1] - b).mean())
        ok(same < flip and same < 8.0,
           "кадр в окне не перевёрнут: расхождение с закадровым %.1f, "
           "а с его зеркалом %.1f из 255" % (same, flip))
    root.destroy()
    print("\nПРОВАЛОВ: " + (str(len(FAILED)) + " — " + "; ".join(FAILED) if FAILED else "нет"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
