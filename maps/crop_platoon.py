"""Нарезка участков ВЗВОДНОГО масштаба (1.5x1.5 км по 30 м на клетку = 50x50).

Отличие от crop_random.py (ротный масштаб, 7 км): на полутора километрах кусок карты часто
оказывается ОДНОРОДНЫМ — сплошной лес или голое поле без единого ориентира. На семи километрах
такого почти не бывает. Поэтому фильтр строже: нужно и открытое пространство (есть где двигаться
под огнём), и укрытия (есть за что цепляться), и не слишком много воды.

    py -3.12 maps/crop_platoon.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CROP_CELL_M", "30")
os.environ.setdefault("CROP_M_PER_UNIT", "15")
os.environ.setdefault("CROP_ARENA", "170")
sys.path.insert(0, HERE)
from crop_tactical import crop_at, MAP_SPAN_M, CROP_SPAN_M  # noqa: E402

N_CROPS = 20
MARGIN = CROP_SPAN_M / 2 + 200
NAMES = {0: "поле", 1: "лес", 2: "здания", 3: "вода", 4: "дороги"}


def usable(grid):
    """Годная вырезка: есть где идти, есть где укрыться, не болото и не монотонна."""
    f = {k: float((grid == k).sum()) / grid.size for k in NAMES}
    if f[0] < 0.15:
        return False, f, "мало открытого"
    if f[1] + f[2] < 0.10:
        return False, f, "нет укрытий"
    if f[3] > 0.25:
        return False, f, "много воды"
    if max(f.values()) > 0.85:
        return False, f, "однородная"
    return True, f, "годная"


def main():
    rng = np.random.default_rng(20260814)
    made = tries = 0
    print(f"{'вырезка':<20}" + "".join(f"{n:>8}" for n in NAMES.values()))
    while made < N_CROPS and tries < 400:
        tries += 1
        cx = rng.uniform(MARGIN, MAP_SPAN_M - MARGIN)
        cy = rng.uniform(MARGIN, MAP_SPAN_M - MARGIN)
        name = f"platoon_crop_{made}"
        grid, _ = crop_at((cx, cy), out_name=name)
        ok, f, why = usable(grid)
        if not ok:
            continue
        print(f"{name:<20}" + "".join(f"{f[k] * 100:>7.0f}%" for k in NAMES))
        made += 1
    # подчистить хвост от прошлых, более длинных прогонов
    for k in range(made, 60):
        for ext in (".npy", ".json"):
            p = os.path.join(HERE, f"platoon_crop_{k}{ext}")
            if os.path.exists(p):
                os.remove(p)
    print(f"\nготово: {made} годных вырезок из {tries} попыток")


if __name__ == "__main__":
    main()
