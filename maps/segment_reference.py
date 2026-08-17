"""
Сегментация авторской карты пользователя (maps/reference_map.png) в наш тайловый формат
по точной палитре цветов, заданной пользователем. Реальный цветовой анализ (nearest-color
классификация по RGB), не приближение на глаз.

Палитра (пользователем):
  525a6f water | 252820 forest (заливка и пунктир-лесопосадки) | 353535 road
  462d2d building | fb0606 team_a home | 2d43a8 team_b home | fff573 contested

Масштаб: карта ~25x25 км (пользователь), изображение 8000x8000 px -> 3.125 м/px.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import terrain

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "reference_map.png")
MAP_SPAN_M = 25000.0          # ~25 км по пользовательской оценке
CELL_M = 50.0                  # размер выходной клетки

PALETTE = {  # terrain-класс -> RGB
    "water":    (0x52, 0x5a, 0x6f),
    "forest":   (0x25, 0x28, 0x20),
    "road":     (0x35, 0x35, 0x35),
    "building": (0x46, 0x2d, 0x2d),
    "open":     (0xff, 0xff, 0xff),
}
MARKERS = {  # метка -> RGB (не террейн, извлекаются отдельно как точки/зоны)
    "team_a_home": (0xfb, 0x06, 0x06),
    "team_b_home": (0x2d, 0x43, 0xa8),
    "contested":   (0xff, 0xf5, 0x73),
}


def nearest_class(arr, palette):
    """arr (H,W,3) uint8 -> (H,W) индекс класса по минимальной евклидовой дистанции в RGB."""
    names = list(palette.keys())
    refs = np.array([palette[n] for n in names], dtype=np.int32)  # (K,3)
    flat = arr.reshape(-1, 3).astype(np.int32)
    # считаем блоками, чтобы не раздувать память (H*W x K)
    idx = np.empty(flat.shape[0], dtype=np.int8)
    B = 2_000_000
    for i in range(0, flat.shape[0], B):
        chunk = flat[i:i + B]                                    # (b,3)
        d = ((chunk[:, None, :] - refs[None, :, :]) ** 2).sum(-1)  # (b,K)
        idx[i:i + B] = d.argmin(axis=1)
    return idx.reshape(arr.shape[:2]), names


def extract_markers(arr, m_per_px, tol=40):
    """Связные области под цвета MARKERS -> список {center_m, size_m} в мировых метрах."""
    out = {}
    for name, rgb in MARKERS.items():
        dist = np.abs(arr.astype(np.int32) - np.array(rgb)).sum(axis=-1)
        mask = dist < tol
        lbl, n = label(mask)
        regions = []
        for sl in find_objects(lbl):
            if sl is None:
                continue
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            if (y1 - y0) * (x1 - x0) < 25:  # шум/антиалиасинг — отсеять мелочь
                continue
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            regions.append({
                "center_m": [round(cx * m_per_px, 1), round((arr.shape[0] - cy) * m_per_px, 1)],
                "size_m": [round((x1 - x0) * m_per_px, 1), round((y1 - y0) * m_per_px, 1)],
            })
        out[name] = regions
    return out


def build_grid(cls_idx, names, m_per_px, cell_m):
    """(H,W) индекс класса -> (Gx,Gy) id наших тайлов, агрегируя блоки с приоритетом
    building > water > road > forest > open (мелкие объекты не теряются при даунсемпле)."""
    H, W = cls_idx.shape
    px_per_cell = int(round(cell_m / m_per_px))
    Gy, Gx = H // px_per_cell, W // px_per_cell
    cls_idx = cls_idx[:Gy * px_per_cell, :Gx * px_per_cell]
    blocks = cls_idx.reshape(Gy, px_per_cell, Gx, px_per_cell).transpose(0, 2, 1, 3).reshape(Gy, Gx, -1)
    n_px = px_per_cell * px_per_cell

    types, *_ = terrain._type_tables()
    frac = {name: (blocks == names.index(name)).sum(-1) / n_px for name in PALETTE}
    grid = np.full((Gx, Gy), types["open"]["id"], dtype=np.int8)  # (Gx,Gy) — наш порядок осей
    # приоритет: building > water > forest > road > open. Road — САМЫЙ слабый (кроме open):
    # дорога — тонкая линия и порог у неё низкий (5%), поэтому её нельзя пускать ПЕРЕД
    # forest/water/building — иначе тропинка, чиркнувшая по лесу, стирала весь лес в клетке
    # ("вкрапления дорог" в лесах/зданиях). Road побеждает только на иначе-open клетках (поля).
    g = grid.T  # временно (Gy,Gx) для удобства с frac
    g[frac["road"] > 0.05] = types["road"]["id"]
    g[frac["forest"] > 0.15] = types["forest"]["id"]
    g[frac["water"] > 0.30] = types["water"]["id"]
    g[frac["building"] > 0.03] = types["building"]["id"]
    grid = g.T
    return grid, Gx, Gy


def compute_building_capacity(arr_terrain, grid, m_per_px, cell_m):
    """Вместимость (число отрядов) каждого СЛИПШЕГОСЯ grid-компонента здания = число РЕАЛЬНЫХ
    зданий (по центроидам сырых пиксельных blob'ов исходника), которые в него попали.
    Минимум 1 на любой компонент, даже если центроид случайно не совпал с его клетками."""
    dist = np.abs(arr_terrain.astype(np.int32) - np.array(PALETTE["building"])).sum(axis=-1)
    mask = dist < 60
    lbl, n = label(mask)
    if n == 0:
        return {}
    sizes = np.bincount(lbl.ravel())
    valid_ids = np.where(sizes[1:] >= 5)[0] + 1  # отсечь однопиксельный шум/антиалиасинг
    from scipy.ndimage import center_of_mass
    centroids = center_of_mass(mask, lbl, valid_ids)  # [(row,col), ...] в ПИКСЕЛЯХ исходника

    tmp = terrain.from_grid(grid, cell_m)  # временная карта — узнать building_id/building_comp
    types, *_ = terrain._type_tables()
    bid = types["building"]["id"]
    px_per_cell = int(round(cell_m / m_per_px))

    counts = {}
    for row, col in centroids:
        # ВАЖНО: та же (нефлипнутая) привязка pixel-row -> gy, что использует build_grid —
        # иначе центроиды разъедутся с building_comp, посчитанным из того же grid.
        gx, gy = int(col // px_per_cell), int(row // px_per_cell)
        if 0 <= gx < tmp.Gx and 0 <= gy < tmp.Gy and tmp.grid[gx, gy] == bid:
            comp = int(tmp.building_comp[gx, gy])
            if comp > 0:
                counts[comp] = counts.get(comp, 0) + 1

    all_comps = set(int(c) for c in np.unique(tmp.building_comp) if c > 0)
    return {c: max(1, counts.get(c, 0)) for c in all_comps}


def main():
    im = Image.open(IMG_PATH).convert("RGB")
    arr = np.array(im)
    H, W = arr.shape[:2]
    m_per_px = MAP_SPAN_M / W
    print(f"изображение {W}x{H}px, {m_per_px:.3f} м/px, поле {W*m_per_px/1000:.1f}x{H*m_per_px/1000:.1f} км")

    # ВАЖНО: пиксели меток-зон (red/blue/yellow) НЕЛЬЗЯ пускать в классификацию террейна — они
    # тонкие обводки, но по ближайшему цвету попадают в building/water (бага: "серые прямоугольники"
    # на месте домашних точек). Гасим их в белый (open) ПЕРЕД классификацией, а не после.
    arr_terrain = arr.copy()
    for rgb in MARKERS.values():
        dist = np.abs(arr.astype(np.int32) - np.array(rgb)).sum(axis=-1)
        arr_terrain[dist < 60] = (255, 255, 255)

    cls_idx, names = nearest_class(arr_terrain, PALETTE)
    u, c = np.unique(cls_idx, return_counts=True)
    print("доля классов:", {names[k]: round(float(v) / cls_idx.size, 3) for k, v in zip(u, c)})

    grid, Gx, Gy = build_grid(cls_idx, names, m_per_px, CELL_M)
    print(f"выходная сетка {Gx}x{Gy} клеток по {CELL_M:.0f} м")
    u2, c2 = np.unique(grid, return_counts=True)
    types, *_ = terrain._type_tables()
    id2name = {t["id"]: n for n, t in types.items()}
    print("доля тайлов после агрегации:", {id2name[int(k)]: round(float(v) / grid.size, 3) for k, v in zip(u2, c2)})

    markers = extract_markers(arr, m_per_px)
    # ручная коррекция: contested[2] распознался целиком в поле — двигаем в ближайший город
    # ВЫШЕ (comp 122, dx всего 97м) по просьбе пользователя. Держим здесь, а не патчим JSON
    # руками, чтобы правка переживала перезапуски пайплайна.
    if len(markers.get("contested", [])) > 2:
        # (17492, 13208) — координаты НЕфлипнутой grid-системы (building_centers); markers хранит
        # Y ФЛИПНУТЫМ (см. extract_markers: y=(H-row)*m_per_px) — конвертируем, иначе рассинхрон
        # с остальными точками (ровно баг, что уронил crop_tactical.py на неверное место).
        markers["contested"][2]["center_m"] = [17492.0, MAP_SPAN_M - 13208.0]
    for name, regs in markers.items():
        print(f"{name}: {len(regs)} шт -> {[r['center_m'] for r in regs]}")

    capacity = compute_building_capacity(arr_terrain, grid, m_per_px, CELL_M)
    cap_vals = np.array(list(capacity.values()))
    print(f"вместимость зданий: {len(capacity)} компонент, "
          f"мин={cap_vals.min()} медиана={int(np.median(cap_vals))} макс={cap_vals.max()}")

    np.save(os.path.join(HERE, "reference_grid.npy"), grid)
    with open(os.path.join(HERE, "reference_zones.json"), "w", encoding="utf-8") as f:
        json.dump({"cell_m": CELL_M, "width_m": Gx * CELL_M, "height_m": Gy * CELL_M,
                   "markers": markers, "building_capacity": capacity}, f, ensure_ascii=False, indent=2)
    print("сохранено: reference_grid.npy, reference_zones.json (+ building_capacity)")
    return grid


if __name__ == "__main__":
    main()
