"""
Оригинальная (авторская) сценарная карта в тактическом духе "приозёрное фермерье поле боя":
крупный водоём вдоль одного края, редкая лесная кромка/перелески, несколько разбросанных
деревень, основной массив — открытые поля. Это НЕ копия конкретной карты Eugen Systems —
собственная раскладка на тех же общих, неавторских элементах ландшафта (озеро, лес, деревни,
поля), под наш масштаб. Сгенерированное здесь используется как отдельный сценарный ассет,
не подключённый пока к тренировочному циклу (WarGameEnv сейчас квадратная арена).

Масштаб: ~25 x 18 км (M_PER_UNIT=70 в wargame_env, но здесь считаем в метрах напрямую),
клетка 100 м -> сетка 250 x 180.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import terrain

CELL = 100.0
WIDTH_M, HEIGHT_M = 25000.0, 18000.0
GX, GY = int(WIDTH_M / CELL), int(HEIGHT_M / CELL)


def build(seed=42):
    rng = np.random.default_rng(seed)
    types, *_ = terrain._type_tables()
    open_id, forest_id, bld_id, water_id = (types[t]["id"] for t in ("open", "forest", "building", "water"))
    grid = np.full((GX, GY), open_id, dtype=np.int8)
    xs, ys = np.meshgrid(np.arange(GX), np.arange(GY), indexing="ij")

    # --- водоём вдоль левого края: неровная береговая линия через шум по Y ---
    shore_x = 22 + 6 * np.sin(np.arange(GY) / 14.0) + rng.normal(0, 1.5, GY)
    for gy in range(GY):
        grid[: max(1, int(shore_x[gy])), gy] = water_id

    # --- редкая лесная кромка вдоль верхнего и правого края (перелески, не сплошная стена) ---
    edge_band = 10
    edge_mask = (ys > GY - edge_band) | (xs > GX - edge_band)
    fringe_noise = rng.random((GX, GY)) < 0.35
    grid[edge_mask & fringe_noise & (grid == open_id)] = forest_id

    # --- разбросанные перелески по полю (негустая пятнистость, не блок) ---
    for _ in range(rng.integers(10, 16)):
        cx, cy = rng.integers(30, GX - 5), rng.integers(5, GY - 5)
        r = rng.integers(3, 9)
        m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        grid[m & (grid == open_id)] = forest_id

    # --- несколько разбросанных деревень (компактные прямоугольные кластеры построек) ---
    villages = []
    for _ in range(rng.integers(5, 8)):
        for _try in range(20):
            cx, cy = rng.integers(28, GX - 8), rng.integers(8, GY - 8)
            if grid[cx, cy] == water_id:
                continue
            w, h = rng.integers(3, 7), rng.integers(3, 7)
            grid[cx:cx + w, cy:cy + h] = bld_id
            villages.append((cx, cy))
            break

    return terrain.from_grid(grid, CELL), villages


if __name__ == "__main__":
    tm, villages = build()
    print(f"карта {tm.Gx}x{tm.Gy} клеток, {tm.cell:.0f} м/клетка, "
          f"{tm.width_m/1000:.1f}x{tm.height_m/1000:.1f} км")
    print(f"деревень: {len(villages)}")
    u, c = np.unique(tm.grid, return_counts=True)
    names = {0: "open", 1: "forest", 2: "building", 3: "water"}
    print("доля тайлов:", {names[int(k)]: round(float(v) / tm.grid.size, 3) for k, v in zip(u, c)})
    np.save(os.path.join(os.path.dirname(__file__), "lakeside_25km.npy"), tm.grid)
    print("сохранено: maps/lakeside_25km.npy")
