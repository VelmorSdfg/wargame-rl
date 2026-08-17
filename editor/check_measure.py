"""Проверка мерки редактора на самом пуле: годные карты должны попадать в окно 25-85%,
вырожденные — вылетать.

Это контроль, без которого панель замера ничего не значит: «моя карта даёт 60%» — не довод,
пока не показано, что та же мерка ставит 60% и картам, которые отбирались руками, а сплошному
лесу ставит 7%. Расхождение на границе окна тоже вскрывается здесь (см. platoon_crop_8).

    py -3.12 editor/check_measure.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.join(os.path.dirname(HERE), "maps")
sys.path.insert(0, HERE)
from measure import measure  # noqa: E402

with open(os.path.join(MAPS, "platoon_pool.json"), "r", encoding="utf-8") as f:
    pool = json.load(f)

names = pool["good"] + list(pool["rejected"])
t0 = time.time()
vis_good, wrong = [], 0
for n in names:
    grid = np.load(os.path.join(MAPS, n + ".npy"))
    with open(os.path.join(MAPS, n + ".json"), "r", encoding="utf-8") as f:
        cell_m = json.load(f)["cell_m"]
    m = measure(grid, cell_m)
    in_pool = n in pool["good"]
    if in_pool:
        vis_good.append(m["vis"])
    if in_pool == bool(m["bad"]):
        wrong += 1
    print(f"{'пул ' if in_pool else 'БРАК'} {n:<18} лес {m['frac'][1] * 100:4.0f}%  "
          f"вид {m['vis'] * 100:4.0f}%  строений {m['comps']:3d}  габарит {m['span_m']:3.0f} м  "
          f"{'ГОДНА' if not m['bad'] else '; '.join(m['bad'])}")

print(f"\nвидимость по пулу: {min(vis_good) * 100:.0f}-{max(vis_good) * 100:.0f}%, "
      f"медиана {np.median(vis_good) * 100:.0f}%")
print(f"расходится с отбором: {wrong} карт из {len(names)}")
print(f"замер одной карты: {(time.time() - t0) / len(names) * 1000:.0f} мс")
