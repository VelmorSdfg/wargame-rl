"""Контроль переделки карты на поля + проверка векторного источника.

Первая половина — страховка, и она важнее второй. terrain.py переведён с «типа клетки» на
поля свойств; пока карта строится из типов, ВСЕ ответы обязаны совпасть со старой версией
до последнего бита. Старая версия берётся не из памяти, а из git — сравнение идёт с тем, что
реально работало.

Вторая половина проверяет вектор: что фигуры растеризуются в ожидаемую местность, что переправа
пробивает реку, что граф дорог связен, и что пересборка под другой размер клетки даёт ту же
карту, а не другую.

    py -3.12 check_map_format.py
"""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import terrain  # noqa: E402
import vectormap  # noqa: E402

M_PER_UNIT = 15.0
FAILED = []

# Эталон прибит к КОНКРЕТНОМУ коммиту — последнему до перевода карты на поля. С HEAD сравнение
# держалось ровно до первого коммита переделки, после чего сверяло новое с новым и всегда было
# зелёным. Здесь это не мелочь: вся проверка на том и стоит, что старая версия настоящая.
BASELINE = "ccad67a"


def ok(cond, what):
    print(("  ok  " if cond else "ПРОВАЛ ") + what)
    if not cond:
        FAILED.append(what)


def load_legacy():
    """terrain.py из последнего коммита, импортированный как отдельный модуль.

    Кладём в корень проекта, а не во временную папку: модуль читает terrain.json рядом с собой
    по __file__, и из /tmp он подхватил бы не тот конфиг (или упал бы)."""
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{BASELINE}:terrain.py"],
                             capture_output=True, check=True).stdout
    except Exception as exc:                                     # noqa: BLE001
        print(f"  git недоступен ({exc}) — сравнение со старой версией пропущено")
        return None
    path = os.path.join(ROOT, "_legacy_terrain_tmp.py")
    with open(path, "wb") as f:
        f.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("terrain_legacy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def compare_maps(old, new, rng, n_pts=400, n_rays=300):
    """Все наружные запросы карты на одинаковых точках. Расхождение = поломка."""
    diffs = []
    pts = rng.uniform(0.5, old.width_m - 0.5, size=(n_pts, 2)).astype(np.float32)

    for p in pts:
        if abs(old.cover_at(p) - new.cover_at(p)) > 1e-9:
            diffs.append("cover_at")
            break
    for p in pts:
        if (abs(old.speed_at(p, False) - new.speed_at(p, False)) > 1e-9
                or abs(old.speed_at(p, True) - new.speed_at(p, True)) > 1e-9):
            diffs.append("speed_at")
            break
    for p in pts:
        if (old.passable(p, False) != new.passable(p, False)
                or old.passable(p, True) != new.passable(p, True)):
            diffs.append("passable")
            break
    for p in pts[:60]:
        if not np.allclose(old.sense_obstacles(p, False), new.sense_obstacles(p, False)):
            diffs.append("sense_obstacles")
            break

    for _ in range(n_rays):
        a = rng.uniform(0, old.width_m, 2).astype(np.float32)
        b = rng.uniform(0, old.width_m, 2).astype(np.float32)
        comp = int(old.component_at(a))
        tr = (comp,) if comp else ()
        if (old.blocked(a, b) != new.blocked(a, b)
                or old.blocked(a, b, tr) != new.blocked(a, b, tr)
                or old.blocked(a, b, demolish=True) != new.blocked(a, b, demolish=True)):
            diffs.append("blocked")
            break

    for p in pts[:40]:
        if old.nearest_road_point(p) != new.nearest_road_point(p):
            diffs.append("nearest_road_point")
            break
        if old.nearest_cover_point(p) != new.nearest_cover_point(p):
            diffs.append("nearest_cover_point")
            break
    for p in pts[:20]:
        if old.nearest_edge_point(p, 1, 35.0, keep_out=3.0) != new.nearest_edge_point(p, 1, 35.0, keep_out=3.0):
            diffs.append("nearest_edge_point")
            break
    for p, q in zip(pts[:12], pts[12:24]):
        if old.firing_position(p, q) != new.firing_position(p, q):
            diffs.append("firing_position")
            break

    if not np.array_equal(old.building_comp, new.building_comp):
        diffs.append("building_comp")
    return diffs


def part_legacy():
    print("\n1. Поля против типов — старая версия из git")
    got = load_legacy()
    if got is None:
        return
    legacy, tmp_path = got
    try:
        import json
        maps_dir = os.path.join(ROOT, "maps")
        with open(os.path.join(maps_dir, "platoon_pool.json"), "r", encoding="utf-8") as f:
            pool = json.load(f)
        names = pool["good"][:6] + list(pool["rejected"])[:1]
        rng = np.random.default_rng(1234)
        bad = []
        for n in names:
            grid = np.load(os.path.join(maps_dir, n + ".npy"))
            with open(os.path.join(maps_dir, n + ".json"), "r", encoding="utf-8") as f:
                cell = json.load(f)["cell_m"] / M_PER_UNIT
            old = legacy.from_grid(grid, cell)
            new = terrain.from_grid(grid, cell)
            d = compare_maps(old, new, np.random.default_rng(rng.integers(1 << 30)))
            if d:
                bad.append(f"{n}: {', '.join(sorted(set(d)))}")
        ok(not bad, "все запросы карты совпадают со старой версией на "
                    f"{len(names)} картах" + (" — " + "; ".join(bad) if bad else ""))

        # процедурная карта: тот же seed должен дать ту же местность
        old = legacy.make_map(np.random.default_rng(7), 170.0)
        new = terrain.make_map(np.random.default_rng(7), 170.0)
        ok(np.array_equal(old.grid, new.grid), "процедурная генерация не изменилась")
        ok(not compare_maps(old, new, np.random.default_rng(99)),
           "на процедурной карте запросы тоже совпадают")
    finally:
        os.remove(tmp_path)


def demo_doc():
    """Учебная карта 2550 м: река наискось, две дороги, мост, село у перекрёстка, лес по берегу.

    Ровно тот порядок, которым карта и рисуется: сначала вода, потом дороги и переправа, потом
    село в узле, потом лес на неудобьях."""
    S = 2550.0
    shapes = [
        {"kind": "line", "type": "water", "width_m": 40,
         "points": [[200, 2400], [900, 1500], [1400, 900], [2100, 150]]},
        {"kind": "line", "type": "road", "width_m": 9,
         "points": [[100, 700], [800, 950], [1500, 1150], [2450, 1350]]},
        {"kind": "line", "type": "road", "width_m": 7,
         "points": [[1150, 100], [1250, 800], [1230, 1600], [1400, 2450]]},
        {"kind": "crossing", "point": [1235, 1035], "width_m": 60},
        {"kind": "polygon", "type": "forest",
         "points": [[300, 2300], [750, 1750], [1000, 1500], [1150, 1650], [800, 2000], [520, 2450]]},
        {"kind": "polygon", "type": "forest",
         "points": [[1700, 500], [2100, 200], [2400, 400], [2050, 800], [1750, 800]]},
        {"kind": "polygon", "type": "forest",
         "points": [[1500, 1700], [2000, 1550], [2400, 1750], [2350, 2200], [1900, 2350], [1550, 2100]]},
        {"kind": "polygon", "type": "forest",
         "points": [[150, 150], [700, 100], [850, 500], [500, 700], [200, 550]]},
        # межи: узкие лесополосы по границам полей — укрытие без сплошной стены
        {"kind": "line", "type": "forest", "width_m": 30,
         "points": [[300, 1200], [900, 1250], [1100, 1400]]},
        {"kind": "line", "type": "forest", "width_m": 25,
         "points": [[1450, 1250], [2000, 1300], [2450, 1200]]},
        {"kind": "line", "type": "forest", "width_m": 25,
         "points": [[900, 300], [950, 800], [1100, 1000]]},
    ]
    for i in range(6):                     # село лентой вдоль поперечной дороги
        shapes.append({"kind": "building",
                       "rect_m": [1300 + (i % 2) * 90 - 45, 900 + i * 70, 40, 28, 8.0],
                       "capacity": 1})
    return vectormap.new_doc((S, S), shapes)


def part_vector():
    print("\n2. Векторная карта")
    doc = demo_doc()
    path = vectormap.save(doc, os.path.join(ROOT, "maps", "demo.vector.json"))
    prefix, surface, meta = vectormap.build(path, cell_m=15.0)

    tid = vectormap._types()
    frac = {k: float((surface == v).mean()) for k, v in tid.items()}
    print("      доли:", {k: round(v, 3) for k, v in frac.items()})
    ok(all(frac[k] > 0 for k in ("forest", "water", "road", "building")),
       "все нарисованные типы попали в сетку")
    ok(surface.shape == (170, 170), f"сетка {surface.shape} при клетке 15 м на поле 2550 м")

    tm, meta = vectormap.load_map(prefix, M_PER_UNIT)
    ok(abs(tm.width_m - 170.0) < 1e-6, f"поле {tm.width_m:.0f} игр.ед — как ARENA")
    ok(len(meta["building_capacity"]) >= 6, f"дома пришли объектами: {len(meta['building_capacity'])}")

    # дом из вектора — это ОДИН компонент, а не слипшееся пятно
    sizes = [int((tm.building_comp == c).sum()) for c in range(1, tm.building_comp.max() + 1)]
    ok(max(sizes) <= 9 if sizes else False, f"компоненты домов мелкие, максимум {max(sizes)} клеток")

    # переправа: поперёк реки должна быть проходимая полоса
    water_row = None
    for gy in range(tm.Gy):
        if (tm.grid[:, gy] == tid["water"]).sum() > 3:
            water_row = gy
            break
    passable_cells = [gx for gx in range(tm.Gx)
                      if tm.grid[gx, water_row] not in (tid["water"],)]
    ok(water_row is not None and len(passable_cells) > 0, "река есть, и она не перекрывает карту целиком")
    near_bridge = [gx for gx in range(tm.Gx)
                   if tm.grid[gx, int(1035 / 15)] == tid["road"]]
    ok(len(near_bridge) > 0, "мост пробил воду и положил дорогу")

    g = meta["graph"]
    ok(len(g["nodes"]) >= 3 and len(g["edges"]) >= 2,
       f"граф дорог собран: {len(g['nodes'])} узлов, {len(g['edges'])} участков")
    # связность графа
    adj = {}
    for e in g["edges"]:
        adj.setdefault(e["a"], []).append(e["b"])
        adj.setdefault(e["b"], []).append(e["a"])
    seen, stack = set(), [g["edges"][0]["a"]] if g["edges"] else []
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack += adj.get(v, [])
    ok(len(seen) == len(g["nodes"]), f"граф связен: {len(seen)} из {len(g['nodes'])} узлов")
    # участок хранит СВОЮ ФОРМУ: длина по ломаной должна быть заметно больше прямой между
    # узлами на извилистой дороге, иначе маршрут пойдёт не по дороге, а напрямик
    curvy = max(g["edges"], key=lambda e: len(e.get("path", [])))
    straight = math.hypot(g["nodes"][curvy["a"]][0] - g["nodes"][curvy["b"]][0],
                          g["nodes"][curvy["a"]][1] - g["nodes"][curvy["b"]][1])
    ok(len(curvy["path"]) >= 2 and curvy["len_m"] >= straight - 1.0,
       f"участок хранит форму дороги: {len(curvy['path'])} точек, "
       f"{curvy['len_m']:.0f} м против {straight:.0f} м по прямой")

    # пересборка под другую клетку — та же местность, а не другая
    _, surface30, _ = vectormap.build(path, cell_m=30.0, out_prefix=prefix + "_c30")
    frac30 = {k: float((surface30 == v).mean()) for k, v in tid.items()}
    worst = max(abs(frac[k] - frac30[k]) for k in tid)
    ok(surface30.shape == (85, 85) and worst < 0.06,
       f"пересборка под 30 м даёт ту же карту (макс. расхождение долей {worst * 100:.1f} п.п.)")

    # ДОРОГА ПО ЛЕСУ. Тип у клетки один и на пересечении побеждает лес, но поля разводят
    # свойства: ход по просеке дорожный, укрытие и непрозрачность — лесные.
    thru = vectormap.new_doc((600.0, 600.0), [
        {"kind": "polygon", "type": "forest",
         "points": [[100, 100], [500, 100], [500, 500], [100, 500]]},
        {"kind": "line", "type": "road", "width_m": 8, "points": [[0, 300], [600, 300]]},
    ])
    s_t, f_t, c_t, cap_t = vectormap.rasterize(thru, 15.0)
    tm_t = terrain.from_fields(s_t, f_t, 1.0, cap_t, c_t)
    gy_road = int(300 / 15)
    gx_mid = int(300 / 15)
    p_road = np.array([(gx_mid + 0.5) * tm_t.cell, (gy_road + 0.5) * tm_t.cell], dtype=np.float32)
    p_wood = np.array([(gx_mid + 0.5) * tm_t.cell, (gy_road + 2.5) * tm_t.cell], dtype=np.float32)
    ok(tm_t.speed_at(p_road) > tm_t.speed_at(p_wood),
       f"дорога в лесу даёт ход дорожный: {tm_t.speed_at(p_road):.2f} против "
       f"{tm_t.speed_at(p_wood):.2f} в чаще")
    ok(tm_t.cover_at(p_road) == tm_t.cover_at(p_wood) and bool(tm_t.f_blocks[gx_mid, gy_road]),
       "и при этом остаётся лесом: укрытие и перекрытие обзора те же")

    # ДОМ НА ДОМЕ. В редакторе легко поставить один дом поверх другого; перекрытые клетки
    # достаются верхнему, и у нижнего их может не остаться вовсе. Раньше такая карта роняла
    # сборку (argmin пустой последовательности при поиске центра строения).
    over = vectormap.new_doc((600.0, 600.0), [
        {"kind": "polygon", "type": "forest",
         "points": [[50, 50], [550, 50], [550, 550], [50, 550]]},
        {"kind": "building", "rect_m": [300, 300, 60, 40, 0.0], "capacity": 1},
        {"kind": "building", "rect_m": [300, 300, 120, 100, 0.0], "capacity": 2},
    ])
    s_over, f_over, comp_over, cap_over = vectormap.rasterize(over, 15.0)
    try:
        tm_over = terrain.from_fields(s_over, f_over, 1.0, cap_over, comp_over)
        built = True
    except Exception as exc:                                   # noqa: BLE001
        built = False
        print(f"      {type(exc).__name__}: {exc}")
    ok(built, "дом, целиком накрытый другим, не ломает сборку карты")
    ok(built and all(int((tm_over.building_comp == c).sum()) > 0 for c in cap_over),
       "вместимости остались только у существующих строений")

    # мерка редактора работает на векторной карте
    sys.path.insert(0, os.path.join(ROOT, "editor"))
    from measure import measure  # noqa: E402
    m = measure(surface, 15.0, n_pairs=300)
    print(f"      замер: видимость {m['vis'] * 100:.0f}%, строений {m['comps']}, "
          f"вердикт {'ГОДНА' if not m['bad'] else '; '.join(m['bad'])}")
    ok(0.05 < m["vis"] < 0.99, "мерка работает на векторной карте")

    for suffix in (".fields.npz", ".map.json"):
        p = prefix + "_c30" + suffix
        if os.path.exists(p):
            os.remove(p)


def main():
    part_legacy()
    part_vector()
    print("\nПРОВАЛОВ: " + (str(len(FAILED)) + " — " + "; ".join(FAILED) if FAILED else "нет"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
