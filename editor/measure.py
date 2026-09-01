"""Мерка карты: те же величины, по которым отбирался maps/platoon_pool.json.

Главное здесь — что видимость считается НАСТОЯЩИМ terrain.blocked() на TerrainMap, собранной
как в reset(): клетка в ИГРОВЫХ единицах (cell_m / m_per_unit). Пороги прозрачности
(see_through_m) сравниваются именно в единицах, и если построить карту в метрах, лес станет
прозрачным в пятнадцать раз дальше, чем в бою, — а замер этого не покажет, просто соврёт.

ОГОВОРКА про калибровку. Это РЕАЛИЗАЦИЯ критерия, описанного в docs/JOURNAL.md, а не тот самый
код (в репозитории его нет). Проверено на текущем пуле (editor/check_measure.py): все 17 годных
попадают в окно 25-85% (27-81%, медиана 61%), обе карты сплошного леса отсеиваются (7%).
Третья отбракованная, platoon_crop_8, даёт здесь 29% против 22% в исходном отборе — мерка
читает на несколько пунктов выше и у границы окна с ней расходится. Для «годна/вырождена»
хватает, для сравнения с числами журнала — нет.
"""
import numpy as np

import project as P
import terrain

VIS_MIN, VIS_MAX = 0.25, 0.85
BAND_M = (300.0, 900.0)


def measure(grid, cell_m, n_pairs=600, seed=12345, fields=None):
    """Доли типов, видимость на боевой дистанции, строения и вердикт годности.

    fields — поля из vectormap.rasterize. Их стоит передавать всегда, когда они есть: там
    лежит высота, а без неё видимость меряется по плоской карте и завышается — гряда, которая
    в бою закроет полполя, для мерки не существует."""
    Gx, Gy = grid.shape
    if fields is not None:
        f = dict(fields)
        f["height"] = np.asarray(f.get("height", 0.0), dtype=np.float32) / P.M_PER_UNIT
        tm = terrain.from_fields(grid, f, cell_m / P.M_PER_UNIT)
    else:
        tm = terrain.from_grid(grid, cell_m / P.M_PER_UNIT)
    rng = np.random.default_rng(seed)
    lo_u, hi_u = BAND_M[0] / P.M_PER_UNIT, BAND_M[1] / P.M_PER_UNIT

    # Пары берутся случайно по ВСЕЙ карте, а не там, где красиво: «видно с опушки на опушку»
    # ничего не значит без контроля по случайным точкам (docs/JOURNAL.md, п. 2 и 3.3).
    open_n = tried = guard = 0
    while tried < n_pairs and guard < n_pairs * 40:
        guard += 1
        ax, ay, bx, by = rng.integers(0, [Gx, Gy, Gx, Gy])
        p0 = np.array([(ax + 0.5) * tm.cell, (ay + 0.5) * tm.cell], dtype=np.float32)
        p1 = np.array([(bx + 0.5) * tm.cell, (by + 0.5) * tm.cell], dtype=np.float32)
        d = float(np.linalg.norm(p1 - p0))
        if not (lo_u <= d <= hi_u):
            continue
        tried += 1
        if not tm.blocked(p0, p1):
            open_n += 1
    vis = open_n / tried if tried else 0.0

    frac = {k: float((grid == k).sum()) / grid.size for k in P.TILE_NAMES}
    comps = int(tm.building_comp.max())
    span_m = 0.0
    for c in range(1, comps + 1):
        xs, ys = np.where(tm.building_comp == c)
        if len(xs):
            span_m = max(span_m, (max(np.ptp(xs), np.ptp(ys)) + 1) * cell_m)

    # Вырожденная карта — не «плохая»: в сплошном лесу бой сводится к натыканию в упор,
    # в голой степи — к перестрелке без манёвра. Ни то ни другое не учит командовать.
    bad = []
    if frac[0] < 0.15:
        bad.append("мало открытого (<15%)")
    if frac[1] + frac[2] < 0.10:
        bad.append("нет укрытий (лес+здания <10%)")
    if frac[3] > 0.25:
        bad.append("много воды (>25%)")
    if max(frac.values()) > 0.85:
        bad.append("однородная (>85% одного типа)")
    if vis < VIS_MIN:
        bad.append(f"глухая: видимость {vis * 100:.0f}% (<{VIS_MIN * 100:.0f}%)")
    elif vis > VIS_MAX:
        bad.append(f"голая: видимость {vis * 100:.0f}% (>{VIS_MAX * 100:.0f}%)")

    return {"frac": frac, "vis": vis, "pairs": tried, "comps": comps, "span_m": span_m, "bad": bad}
