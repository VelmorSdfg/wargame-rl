"""
Вырезает тактический участок 7x7 км (= наша ARENA=100 * M_PER_UNIT=70) из карты reference_map.png,
центрируясь на заданной точке (в тех же "мировых метрах", что и tm._cell()/building_centers —
той же системе координат, что использует TerrainMap). Пересчитывает вместимость зданий ЗАНОВО
для вырезанного куска (той же функцией, что и полная сегментация), не переносит старые id компонент
(они всё равно другие после локального пересчёта связности).

Результат — обычный TerrainMap 140x140 клеток по 50м = 7000м = ARENA(100)*M_PER_UNIT(70), поэтому
подключается к WarGameEnv БЕЗ прямоугольной арены — совпадает с текущей square ARENA один в один.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import terrain
from segment_reference import PALETTE, nearest_class, build_grid, compute_building_capacity, MARKERS

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_SPAN_M = 25000.0
CELL_M = float(__import__("os").environ.get("CROP_CELL_M", 50.0))
# ARENA больше не зашита числом: поле выросло со 100 до 170 единиц, и вырезки должны
# следовать за ним, иначе загруженная карта не совпадёт по размеру с полем.
CROP_SPAN_M = (float(__import__("os").environ.get("CROP_ARENA", 100.0))
               * float(__import__("os").environ.get("CROP_M_PER_UNIT", 70.0)))


def crop_at(center_m, out_name="tactical_crop"):
    im = Image.open(os.path.join(HERE, "reference_map.png")).convert("RGB")
    arr = np.array(im)
    H, W = arr.shape[:2]
    m_per_px = MAP_SPAN_M / W
    px_per_cell = int(round(CELL_M / m_per_px))
    cells_span = int(round(CROP_SPAN_M / CELL_M))  # 140

    # гасим метки в белый ДО кропа — та же логика, что в основном пайплайне
    arr_terrain = arr.copy()
    for rgb in MARKERS.values():
        dist = np.abs(arr.astype(np.int32) - np.array(rgb)).sum(axis=-1)
        arr_terrain[dist < 60] = (255, 255, 255)

    # ВАЖНО: center_m приходит из extract_markers(), где y ХРАНИТСЯ ФЛИПНУТЫМ: y=(H-row)*m_per_px
    # (чтобы верх картинки = большой Y — для рендера через to_screen). А сетка/пиксели строятся
    # НЕфлипнутыми (gy=row//px_per_cell напрямую, см. build_grid/_building_components). Если тут
    # взять center_m[1] как есть — вырежем зеркально не то место. Разворачиваем обратно в row.
    col_center = center_m[0] / m_per_px
    row_center = H - center_m[1] / m_per_px
    gx0 = int(col_center // px_per_cell) - cells_span // 2
    gy0 = int(row_center // px_per_cell) - cells_span // 2
    Gmax = W // px_per_cell
    gx0 = max(0, min(gx0, Gmax - cells_span))
    gy0 = max(0, min(gy0, Gmax - cells_span))
    row0, col0 = gy0 * px_per_cell, gx0 * px_per_cell
    size_px = cells_span * px_per_cell

    arr_crop = arr_terrain[row0:row0 + size_px, col0:col0 + size_px]
    cls_idx, names = nearest_class(arr_crop, PALETTE)
    grid, Gx, Gy = build_grid(cls_idx, names, m_per_px, CELL_M)
    capacity = compute_building_capacity(arr_crop, grid, m_per_px, CELL_M)

    u, c = np.unique(grid, return_counts=True)
    types, *_ = terrain._type_tables()
    id2name = {t["id"]: n for n, t in types.items()}
    print(f"crop {Gx}x{Gy} клеток ({Gx*CELL_M:.0f}x{Gy*CELL_M:.0f}м), "
          f"тайлы: {({id2name[int(k)]: round(float(v)/grid.size,3) for k,v in zip(u,c)})}")
    print(f"зданий: {len(capacity)}, вместимость мин={min(capacity.values()) if capacity else 0} "
          f"макс={max(capacity.values()) if capacity else 0}")

    np.save(os.path.join(HERE, f"{out_name}.npy"), grid)
    with open(os.path.join(HERE, f"{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump({"cell_m": CELL_M, "center_m": list(center_m), "building_capacity": capacity},
                  f, ensure_ascii=False, indent=2)
    print(f"сохранено: {out_name}.npy, {out_name}.json")
    return grid, capacity


if __name__ == "__main__":
    # центр — ТА САМАЯ первая спорная точка (contested[0]), вокруг которой рисовали
    # белую рамку "наш тактический бой 7x7км" на reference_preview.png. Не произвольная точка.
    crop_at((20154.7, 18771.9))
