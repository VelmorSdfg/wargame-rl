"""Рендерит превью сценарной карты нашей ОБСТВЕННОЙ палитрой (play.py TERRAIN_COLORS) в PNG."""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gen_lakeside as _gl  # noqa: E402 (запускать из папки maps/, см. __main__)

COLORS = {0: (58, 74, 40), 1: (24, 46, 26), 2: (90, 90, 100), 3: (35, 70, 110)}  # open/forest/building/water


def render(tm, out_path, scale=3):
    surf = pygame.Surface((tm.Gx * scale, tm.Gy * scale))
    for gx in range(tm.Gx):
        col_top = tm.Gy * scale
        for gy in range(tm.Gy):
            c = COLORS[int(tm.grid[gx, gy])]
            pygame.draw.rect(surf, c, (gx * scale, col_top - (gy + 1) * scale, scale, scale))
    pygame.image.save(surf, out_path)


if __name__ == "__main__":
    pygame.init()
    tm, villages = _gl.build()
    out = os.path.join(os.path.dirname(__file__), "lakeside_25km_preview.png")
    render(tm, out)
    print(f"превью сохранено: {out}  ({tm.Gx}x{tm.Gy} клеток x3 = {tm.Gx*3}x{tm.Gy*3}px)")
