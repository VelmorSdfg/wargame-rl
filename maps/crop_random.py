"""
Нарезает НЕСКОЛЬКО случайных тактических участков 7x7км с карты reference_map.png —
для разнообразия при обучении (одна вырезка = риск заучивания конкретной карты агентом).
Отсеивает участки, где вода > MAX_WATER_FRAC (слишком много воды — неинтересно тактически).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_tactical import crop_at, MAP_SPAN_M, CROP_SPAN_M

N_CROPS = 20
MAX_WATER_FRAC = 0.30
MARGIN = CROP_SPAN_M / 2 + 200  # держим центр подальше от края, чтобы кроп не клэмпился


def main():
    rng = np.random.default_rng(123)
    made = 0
    tries = 0
    while made < N_CROPS and tries < 60:
        tries += 1
        cx = rng.uniform(MARGIN, MAP_SPAN_M - MARGIN)
        cy = rng.uniform(MARGIN, MAP_SPAN_M - MARGIN)
        name = f"tactical_crop_{made}"
        grid, capacity = crop_at((cx, cy), out_name=name)
        water_frac = float((grid == 3).sum()) / grid.size
        if water_frac > MAX_WATER_FRAC:
            print(f"  отклонено (вода {water_frac:.0%}) центр=({cx:.0f},{cy:.0f}) -> пробуем ещё")
            continue
        print(f"[{made}] принято: центр=({cx:.0f},{cy:.0f}) вода={water_frac:.0%} -> {name}")
        made += 1
    print(f"\nготово: {made} вырезок из {tries} попыток")


if __name__ == "__main__":
    main()
