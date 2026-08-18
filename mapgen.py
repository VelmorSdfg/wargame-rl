"""Генератор местности в векторе и нарезка из него боевых карт.

Порядок рисования тут не произвольный, а тот, которым местность и складывается в природе:
сначала вода, потом дороги по удобному с переправами через воду, потом сёла в узлах дорог,
и только потом лес — на том, что осталось неудобным. Карта, начатая с леса, выходит кашей:
пятна не объясняются ничем, дороги их обходят как попало, и на глаз это сразу видно.

Второй смысл файла — нарезка. Один нарисованный театр даёт сколько угодно боевых карт:
случайное место, случайный угол, зеркало. Вектор режется под любым углом без лесенки, а каждый
кусок проверяется той же меркой, что отбирала пул (docs/JOURNAL.md, п. 3.3): сплошной лес и
голая степь выбрасываются, потому что ни то ни другое не учит командовать.

    py -3.12 mapgen.py theatre --size 10000 --seed 5      # нарисовать театр
    py -3.12 mapgen.py crops maps/theatre_5.vector.json --count 20 --size 2550 --cell 15
    py -3.12 mapgen.py battle --seed 3                     # сразу одна боевая карта
"""
import argparse
import math
import os
import sys

import numpy as np

import vectormap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor"))
from measure import measure  # noqa: E402

MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")


# ---------------------------------------------------------------- вспомогательное


def _wander(rng, a, b, n, sway):
    """Ломаная из a в b, гуляющая вбок: реки и дороги не бывают прямыми, а прямая линия на
    карте сразу читается как нарисованная."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ln = float(np.linalg.norm(d))
    perp = np.array([-d[1], d[0]]) / max(ln, 1e-9)
    pts = []
    off = 0.0
    for i in range(n + 1):
        t = i / n
        off = off * 0.6 + rng.normal(0, sway) * 0.4          # плавный дрейф, а не дрожь
        edge = math.sin(math.pi * t)                          # у концов не уводим
        p = a + d * t + perp * off * edge
        pts.append([float(p[0]), float(p[1])])
    return pts


def _blob(rng, cx, cy, r, irregular=0.45, n=11):
    """Неправильное пятно: радиус гуляет по углу. Круги и прямоугольники выдают генератор
    с первого взгляда, а лес кругами вдобавок перекрывает любой задевший его луч."""
    pts = []
    phase = rng.uniform(0, 2 * math.pi)
    k1, k2 = rng.uniform(1.5, 3.0), rng.uniform(3.0, 5.0)
    for i in range(n):
        t = 2 * math.pi * i / n
        rr = r * (1 + irregular * (0.6 * math.sin(k1 * t + phase) + 0.4 * math.sin(k2 * t)))
        pts.append([cx + rr * math.cos(t), cy + rr * math.sin(t)])
    return pts


def _seg_hits(poly_a, poly_b):
    """Точки пересечения двух ломаных — там ставятся переправы и вырастают сёла."""
    out = []
    for a0, a1 in zip(poly_a[:-1], poly_a[1:]):
        for b0, b1 in zip(poly_b[:-1], poly_b[1:]):
            hit = vectormap._seg_intersect(a0, a1, b0, b1)
            if hit:
                out.append(hit[0])                   # (точка, доля, доля) -> точка
    return out


def _clamp_pts(pts, size):
    return [[float(np.clip(x, 0, size)), float(np.clip(y, 0, size))] for x, y in pts]


# ---------------------------------------------------------------- генерация


def generate(size_m=2550.0, seed=0, battle_edges=True):
    """Вектор местности. battle_edges: держать полосы развёртывания (300 м от нижнего и
    верхнего края) чистыми — иначе тот, кто там начинает, стартует в готовой крепости."""
    rng = np.random.default_rng(seed)
    S = float(size_m)
    shapes = []

    # ПЛОТНОСТИ, А НЕ ЧИСЛА. Первый вариант задавал «пять дорог и два села» — на боевой карте
    # выходило похоже, а на десятикилометровом театре всё растворялось: вырезки ловили 0-2%
    # дорог и ни одного дома при норме пула 7% и 3%. Размеры пятен по той же причине абсолютные:
    # лес долей от карты давал на театре массивы по километру, и кусок целиком попадал внутрь.
    area_km2 = (S / 1000.0) ** 2
    n_long = max(1, int(round(S / 1000.0 / 2.2)))          # продольная дорога каждые ~2.2 км
    n_cross = max(2, int(round(S / 1000.0 / 1.8)))         # поперечная каждые ~1.8 км
    n_villages = max(1, int(round(area_km2 / 7.0)))        # село на ~7 км²
    n_forest = max(3, int(round(area_km2 * rng.uniform(0.8, 1.4))))
    n_hedge = max(2, int(round(area_km2 * 0.7)))
    FOREST_R = (120.0, 340.0)                              # радиус массива, МЕТРЫ
    ROAD_KEEPOUT = 150.0                                   # лес не лезет вплотную к дороге

    # 1. ВОДА — одна ось наискось. Она диктует всё остальное: где трудно, где переправы.
    side = rng.integers(0, 2)
    if side == 0:
        a, b = (rng.uniform(0, S * 0.35), S), (rng.uniform(S * 0.65, S), 0)
    else:
        a, b = (rng.uniform(0, S * 0.35), 0), (rng.uniform(S * 0.65, S), S)
    river = _clamp_pts(_wander(rng, a, b, 7, S * 0.05), S)
    river_w = float(rng.uniform(0.010, 0.022) * S)
    shapes.append({"kind": "line", "type": "water", "width_m": round(river_w, 1),
                   "points": river})

    # 2. ДОРОГИ. Одна вдоль долины, две-три поперёк — они и пересекут реку.
    roads = []
    rd = np.array(river[-1]) - np.array(river[0])
    ang = math.atan2(rd[1], rd[0])
    for k in range(n_long):                                   # продольные
        # разносим по ширине, а не бросаем случайно: две дороги в одном месте — это одна дорога
        frac = (k + 0.5) / n_long - 0.5
        off = frac * 1.6 * S + rng.uniform(-0.05, 0.05) * S
        px, py = -math.sin(ang) * off, math.cos(ang) * off
        p0 = [float(np.clip(river[0][0] + px, 0, S)), float(np.clip(river[0][1] + py, 0, S))]
        p1 = [float(np.clip(river[-1][0] + px, 0, S)), float(np.clip(river[-1][1] + py, 0, S))]
        roads.append(_clamp_pts(_wander(rng, p0, p1, 6, S * 0.03), S))
    for k in range(n_cross):                                  # поперечные
        t = (k + 0.5 + rng.uniform(-0.15, 0.15)) / n_cross
        c = np.array(river[0]) + (np.array(river[-1]) - np.array(river[0])) * t
        n = np.array([-math.sin(ang), math.cos(ang)])
        p0 = _clamp_pts([c + n * S * 0.8], S)[0]
        p1 = _clamp_pts([c - n * S * 0.8], S)[0]
        roads.append(_clamp_pts(_wander(rng, p0, p1, 6, S * 0.025), S))
    for r in roads:
        shapes.append({"kind": "line", "type": "road",
                       "width_m": round(float(rng.uniform(7, 12)), 1), "points": r})

    # 3. ПЕРЕПРАВЫ — на каждом пересечении дороги с рекой. Без них река делит карту пополам.
    crossings = []
    for r in roads:
        for hit in _seg_hits(r, river):
            crossings.append(hit)
            # ширину проезда и длину моста не задаём: они выводятся из ширины дороги и реки
            # в этом месте (vectormap.crossing_geom) — мост должен быть узким местом
            shapes.append({"kind": "crossing", "point": [round(hit[0], 1), round(hit[1], 1)]})

    # 4. СЁЛА — в узлах дорог, лентой ВДОЛЬ улицы, а не пятном.
    knots = []
    for i in range(len(roads)):
        for j in range(i + 1, len(roads)):
            knots += _seg_hits(roads[i], roads[j])
    rng.shuffle(knots)
    spots = list(knots)
    while len(spots) < n_villages:                            # узлов не хватило — сажаем вдоль дороги
        r = roads[int(rng.integers(0, len(roads)))]
        spots.append(tuple(r[int(rng.integers(1, len(r) - 1))]))
    n_vil = n_villages
    for v in range(n_vil):
        kx, ky = spots[v]
        host = min(roads, key=lambda r: min((p[0] - kx) ** 2 + (p[1] - ky) ** 2 for p in r))
        idx = int(np.argmin([(p[0] - kx) ** 2 + (p[1] - ky) ** 2 for p in host]))
        nxt = host[min(idx + 1, len(host) - 1)]
        dirv = np.array(nxt) - np.array(host[idx])
        ln = float(np.linalg.norm(dirv)) or 1.0
        dirv = dirv / ln
        perp = np.array([-dirv[1], dirv[0]])
        houses = int(rng.integers(6, 14)) if v == 0 else int(rng.integers(3, 6))
        for hidx in range(houses):
            along = (hidx - houses / 2) * rng.uniform(45, 70)
            sidep = (1 if hidx % 2 else -1) * rng.uniform(22, 40)
            p = np.array([kx, ky]) + dirv * along + perp * sidep
            if not (0 < p[0] < S and 0 < p[1] < S):
                continue
            shapes.append({"kind": "building",
                           "rect_m": [round(float(p[0]), 1), round(float(p[1]), 1),
                                      round(float(rng.uniform(18, 32)), 1),
                                      round(float(rng.uniform(12, 22)), 1),
                                      round(float(math.degrees(math.atan2(dirv[1], dirv[0]))
                                                  + rng.uniform(-12, 12)), 1)],
                           "capacity": 1})

    # 5. ЛЕС — на неудобьях: по берегу и вдали от дорог. Форма вытянутая и рваная.
    def far_from_roads(p, d):
        for r in roads:
            for q in r:
                if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < d * d:
                    return False
        return True

    band = 300.0 if battle_edges else 0.0
    n_patch = n_forest
    placed = 0
    for _ in range(n_patch * 8):
        if placed >= n_patch:
            break
        if rng.random() < 0.45:                               # прижатый к реке массив
            t = rng.uniform(0.1, 0.9)
            i = min(int(t * (len(river) - 1)), len(river) - 2)
            base = np.array(river[i])
            n = np.array([-(river[i + 1][1] - river[i][1]), river[i + 1][0] - river[i][0]])
            n = n / (np.linalg.norm(n) or 1.0)
            c = base + n * rng.uniform(0.02, 0.09) * S * (1 if rng.random() < 0.5 else -1)
        else:
            c = rng.uniform(0.08, 0.92, size=2) * S
        if battle_edges and not (band < c[1] < S - band):
            continue
        if not far_from_roads(c, ROAD_KEEPOUT):
            continue
        r = rng.uniform(*FOREST_R)
        shapes.append({"kind": "polygon", "type": "forest",
                       "points": [[round(x, 1), round(y, 1)]
                                  for x, y in _clamp_pts(_blob(rng, c[0], c[1], r), S)]})
        placed += 1

    # 6. МЕЖИ — узкие лесополосы по границам полей. Дают укрытие, не превращая поле в стену.
    for _ in range(n_hedge):
        p0 = rng.uniform(0.05, 0.95, size=2) * S
        d = rng.uniform(-1, 1, size=2)
        d = d / (np.linalg.norm(d) or 1.0)
        p1 = p0 + d * rng.uniform(300.0, 900.0)              # межа — сотни метров, не доля карты
        shapes.append({"kind": "line", "type": "forest",
                       "width_m": round(float(rng.uniform(18, 32)), 1),
                       "points": _clamp_pts(_wander(rng, p0, p1, 3, S * 0.012), S)})

    return vectormap.new_doc((S, S), shapes), {"crossings": len(crossings), "villages": n_vil}


def generate_good(size_m=2550.0, seed=0, cell_m=30.0, tries=8, battle_edges=True, any_map=False):
    """Генерировать, пока карта не окажется годной по мерке. Замер тут не украшение: на глаз
    «лес есть, дороги есть» ничего не значит — вырожденной карта становится по числам.

    any_map=True отключает отбор: карта отдаётся как есть, с замером в отчёте. Нужно, когда
    важен сам факт карты (проверить редактор, посмотреть форму), а не её пригодность к бою."""
    if any_map:
        doc, _info = generate(size_m, seed, battle_edges)
        surface, *_ = vectormap.rasterize(doc, cell_m)
        return doc, measure(surface, cell_m, n_pairs=300), 1
    for k in range(tries):
        doc, info = generate(size_m, seed * 1000 + k, battle_edges)
        surface, *_ = vectormap.rasterize(doc, cell_m)
        m = measure(surface, cell_m, n_pairs=300)
        if not m["bad"]:
            return doc, m, k + 1
    return doc, m, tries


# ---------------------------------------------------------------- нарезка


def cut_crops(theatre_path, count, size_m, cell_m, out_dir, seed=0, keep_bad=False):
    """Случайные куски театра: место, угол, зеркало. Вырожденные выбрасываются меркой."""
    doc = vectormap.load(theatre_path)
    W, H = doc["size_m"]
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    half = size_m * 0.75                      # запас, чтобы кусок под углом не вылез за театр
    made, tried, rows = 0, 0, []
    while made < count and tried < count * 40:
        tried += 1
        cx = rng.uniform(half, W - half)
        cy = rng.uniform(half, H - half)
        ang = rng.uniform(0, 360)
        piece = vectormap.crop(doc, (cx, cy), (size_m, size_m), ang)
        if len(piece["shapes"]) < 3:
            continue
        surface, *_ = vectormap.rasterize(piece, cell_m)
        m = measure(surface, cell_m, n_pairs=300)
        if m["bad"] and not keep_bad:
            continue
        name = f"{os.path.splitext(os.path.basename(theatre_path))[0].replace('.vector', '')}_c{made}"
        vp = vectormap.save(piece, os.path.join(out_dir, name + ".vector.json"))
        vectormap.build(vp, cell_m)
        rows.append((name, cx, cy, ang, m))
        made += 1
    return rows, tried


# ---------------------------------------------------------------- CLI


def _report(name, m):
    print(f"{name:<22} лес {m['frac'][1] * 100:4.0f}%  застр {m['frac'][2] * 100:3.0f}%  "
          f"дор {m['frac'][4] * 100:3.0f}%  вид {m['vis'] * 100:4.0f}%  "
          f"стр {m['comps']:3d}   {'ГОДНА' if not m['bad'] else '; '.join(m['bad'])}")


def main():
    ap = argparse.ArgumentParser(description="генератор карт и нарезка боевых кусков")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("theatre", help="нарисовать большой театр")
    g.add_argument("--size", type=float, default=10000.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--cell", type=float, default=30.0)
    g.add_argument("--out", default=None)

    b = sub.add_parser("battle", help="нарисовать одну боевую карту")
    b.add_argument("--size", type=float, default=2550.0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--cell", type=float, default=15.0)
    b.add_argument("--out", default=None)

    c = sub.add_parser("crops", help="нарезать боевые карты из театра")
    c.add_argument("theatre")
    c.add_argument("--count", type=int, default=10)
    c.add_argument("--size", type=float, default=2550.0)
    c.add_argument("--cell", type=float, default=15.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out-dir", default=os.path.join(MAPS, "crops"))
    c.add_argument("--keep-bad", action="store_true", help="не отбраковывать вырожденные куски")
    for p in (g, b):
        p.add_argument("--any", action="store_true", dest="any_map",
                       help="не переигрывать ради годности — отдать карту как есть")

    args = ap.parse_args()

    if args.cmd in ("theatre", "battle"):
        battle = args.cmd == "battle"
        doc, m, tries = generate_good(args.size, args.seed, cell_m=args.cell if battle else 30.0,
                                      battle_edges=battle, any_map=args.any_map)
        name = args.out or os.path.join(MAPS, f"{'battle' if battle else 'theatre'}_{args.seed}.vector.json")
        vectormap.save(doc, name)
        prefix, surface, meta = vectormap.build(name, args.cell)
        _report(os.path.basename(name), m)
        print(f"попыток до годной: {tries}; фигур: {len(doc['shapes'])}; "
              f"узлов дорог {len(meta['graph']['nodes'])}, участков {len(meta['graph']['edges'])}")
        print(f"собрано: {prefix}.fields.npz + {prefix}.map.json")
    else:
        rows, tried = cut_crops(args.theatre, args.count, args.size, args.cell,
                                args.out_dir, args.seed, keep_bad=args.keep_bad)
        for name, cx, cy, ang, m in rows:
            _report(f"{name} ({ang:.0f}°)", m)
        print(f"\nгодных {len(rows)} из {tried} попыток, лежат в {args.out_dir}")


if __name__ == "__main__":
    main()
