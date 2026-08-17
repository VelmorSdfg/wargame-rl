"""Строит TerrainMap из reference_grid.npy и рендерит превью нашей палитрой + зоны-метки."""
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import terrain

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {0: (58, 74, 40), 1: (24, 46, 26), 2: (90, 90, 100), 3: (35, 70, 110), 4: (150, 140, 90)}
MARKER_COLORS = {"team_a_home": (220, 60, 60), "team_b_home": (70, 100, 220), "contested": (230, 210, 90)}


def main():
    grid = np.load(os.path.join(HERE, "reference_grid.npy"))
    with open(os.path.join(HERE, "reference_zones.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    cap = {int(k): v for k, v in meta.get("building_capacity", {}).items()}
    tm = terrain.from_grid(grid, meta["cell_m"], cap)
    print(f"TerrainMap: {tm.Gx}x{tm.Gy}, {tm.width_m/1000:.1f}x{tm.height_m/1000:.1f} км, "
          f"зданий(компонент): {len(tm.building_centers)}")

    pygame.init()
    scale = 2
    surf = pygame.Surface((tm.Gx * scale, tm.Gy * scale))
    for gx in range(tm.Gx):
        top = tm.Gy * scale
        for gy in range(tm.Gy):
            pygame.draw.rect(surf, COLORS[int(tm.grid[gx, gy])],
                             (gx * scale, top - (gy + 1) * scale, scale, scale))

    def to_screen(mx, my):
        return int(mx / tm.cell * scale), int(tm.Gy * scale - my / tm.cell * scale)

    for name, regs in meta["markers"].items():
        col = MARKER_COLORS[name]
        for r in regs:
            mx, my = r["center_m"]; sw, sh = r["size_m"]  # реальный размер зоны (bounding box), не фикс-радиус
            x0, y0 = to_screen(mx - sw / 2, my + sh / 2)
            x1, y1 = to_screen(mx + sw / 2, my - sh / 2)
            pygame.draw.rect(surf, col, (x0, y0, max(1, x1 - x0), max(1, y1 - y0)), 3)

    # --- масштабная линейка (1 км) ---
    px_per_km = 1000.0 / tm.cell * scale
    bx, by = 20, tm.Gy * scale - 24
    pygame.draw.line(surf, (240, 240, 240), (bx, by), (bx + px_per_km, by), 3)
    pygame.draw.line(surf, (240, 240, 240), (bx, by - 5), (bx, by + 5), 2)
    pygame.draw.line(surf, (240, 240, 240), (bx + px_per_km, by - 5), (bx + px_per_km, by + 5), 2)
    font = pygame.font.SysFont("consolas", 16)
    surf.blit(font.render("1 km", True, (240, 240, 240)), (bx, by - 22))

    # --- рамка нашей тактической арены (7x7 км = ARENA*M_PER_UNIT) поверх одной спорной точки ---
    ARENA_M = 100.0 * 70.0  # см. wargame_env.py: ARENA=100, M_PER_UNIT=70
    contested = meta["markers"].get("contested", [])
    if contested:
        acx, acy = contested[0]["center_m"]  # центрируем на первой спорной точке
        half = ARENA_M / 2
        x0, y0 = to_screen(acx - half, acy + half)
        x1, y1 = to_screen(acx + half, acy - half)
        pygame.draw.rect(surf, (255, 255, 255), (x0, y0, x1 - x0, y1 - y0), 2)
        surf.blit(font.render("наш тактический бой (7x7 км)", True, (255, 255, 255)), (x0, y0 - 20))

        # --- NATO-иконка юнита (ОБТ) — того же визуального размера, что в play.py (не истинный масштаб!) ---
        # ВАЖНО: масштаб px/м здесь другой, чем в play.py (25км/1000px vs 7км/1080px) — пиксельный
        # размер иконки НЕ означает то же мировое расстояние. Это ориентировочная метка-булавка
        # (для видимости на карте 25км), а не масштабный значок. Настоящий танк (~10м) был бы <1px.
        icx, icy = to_screen(acx, acy)
        w, h = 3, 2
        pygame.draw.rect(surf, (24, 26, 32), (icx - w, icy - h, 2 * w, 2 * h))
        pygame.draw.rect(surf, (90, 150, 255), (icx - w, icy - h, 2 * w, 2 * h), 2)
        pygame.draw.ellipse(surf, (245, 245, 245), (icx - w + 3, icy - h + 3, 2 * w - 6, 2 * h - 6), 0)
        surf.blit(font.render("1 юнит (метка-булавка, НЕ в масштабе)", True, (90, 150, 255)), (icx + w + 6, icy - 6))

    out = os.path.join(HERE, "reference_preview.png")
    pygame.image.save(surf, out)
    print(f"превью: {out}  ({tm.Gx*scale}x{tm.Gy*scale}px)")


if __name__ == "__main__":
    main()
