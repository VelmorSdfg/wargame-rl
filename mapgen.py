"""Генератор местности в векторе и нарезка из него боевых карт.

Порядок рисования тут не произвольный, а тот, которым местность и складывается в природе:
сначала РЕЛЬЕФ, потом вода — по спуску, а не наискось по линейке, потом дороги с переправами,
потом сёла в узлах дорог, и только потом лес — на том, что осталось неудобным. Карта, начатая
с леса, выходит кашей: пятна не объясняются ничем, дороги их обходят как попало.

ИЕРАРХИЯ МАСШТАБОВ. Прежний генератор раскладывал всё одного размера, и большая карта выходила
маленькой, повторённой шестнадцать раз: на театре 100 км² было 0.6 здания на км² четырнадцатью
одинаковыми хуторами, лес одного радиуса и размах высот 97 м — тот же, что на карте вчетверо
меньшей. Теперь у каждого слоя свои размеры (замер на 10 км против прежнего):

    фигур на км²   2.6 -> 5.0        здания на км²   0.6 -> 3.1
    ширины дорог   7-12 м -> 5-16    радиус леса     98-344 м -> 126-1200 м
    размах высот   104 м -> 155 м    видимость       45% -> 39% (окно пула 25-85%)

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


# Размеры строений — НАСТОЯЩИЕ, а не подогнанные под клетку. Раньше здесь стояло
# uniform(16, 30) x uniform(11, 20), в городе ещё x1.35: ни одной избы, ни одного сарая, всё
# размером с сельский клуб. Числа были не небрежностью, а вынужденной подгонкой — пока местность
# считалась клетками по 30 м, строение мельче примерно 7x7 м до боя не доходило вовсе.
#
# VectorTerrain этот порог снял: помеха и материал считаются по фигуре, а не по доле покрытия
# клетки. Село наконец может состоять из изб.
#
# Оговорка: у мелкого дома по-прежнему нет СВОЕЙ КЛЕТКИ, а значит нет и компоненты застройки —
# занять его отделением нельзя, и в поиске укрытия по сетке он не участвует. Он укрывает и
# перекрывает обзор, но гарнизона не держит.
HOUSE_MIX = {
    #            доля   длина м    ширина м     что это
    "hamlet":  ((0.45, (8, 12), (6, 9)),      # изба
                (0.35, (5, 9), (4, 6)),       # сарай, баня, хлев
                (0.20, (14, 22), (8, 12))),   # дом с пристройками
    "village": ((0.38, (9, 14), (7, 10)),     # изба
                (0.24, (5, 9), (4, 7)),       # сарай
                (0.26, (16, 24), (9, 13)),    # дом побольше, лавка, школа
                (0.12, (30, 60), (10, 16))),  # коровник, склад, мастерская
    "town":    ((0.28, (12, 20), (8, 12)),    # частный сектор
                (0.34, (24, 40), (10, 14)),   # многоквартирный
                (0.20, (45, 75), (11, 16)),   # длинный дом
                (0.18, (6, 10), (4, 8))),     # гаражи и сараи во дворах
}


def _house_size(rng, kind):
    """Длина и ширина строения по жребию из смеси своего посёлка."""
    mix = HOUSE_MIX.get(kind, HOUSE_MIX["village"])
    r = float(rng.random()) * sum(m[0] for m in mix)
    acc = 0.0
    for share, (lo_l, hi_l), (lo_w, hi_w) in mix:
        acc += share
        if r <= acc:
            return float(rng.uniform(lo_l, hi_l)), float(rng.uniform(lo_w, hi_w))
    share, (lo_l, hi_l), (lo_w, hi_w) = mix[-1]
    return float(rng.uniform(lo_l, hi_l)), float(rng.uniform(lo_w, hi_w))


def _clamp_pts(pts, size):
    return [[float(np.clip(x, 0, size)), float(np.clip(y, 0, size))] for x, y in pts]


# ---------------------------------------------------------------- генерация


def _fractal(rng, n, octaves=6, persist=0.52, ridged=False):
    """Фрактальное поле 0..1: сумма октав шума, каждая вдвое мельче и вдвое слабее.

    Так устроен настоящий рельеф — крупные массивы, на них складки, на складках мелочь. Одна
    октава даёт гладкий бугор, и карта любого размера выглядит одинаково пустой."""
    from PIL import Image as _I
    out = np.zeros((n, n), dtype=np.float32)
    amp, total, size = 1.0, 0.0, 2
    for _ in range(octaves):
        g = rng.random((size + 1, size + 1)).astype(np.float32)
        up = np.asarray(_I.fromarray(g).resize((n, n), _I.BICUBIC), dtype=np.float32)
        if ridged:
            up = 1.0 - np.abs(2.0 * up - 1.0)     # гребни вместо холмов: острее и «горнее»
        out += up * amp
        total += amp
        amp *= persist
        size *= 2
    out /= max(total, 1e-6)
    lo, hi = float(out.min()), float(out.max())
    return (out - lo) / max(hi - lo, 1e-6)


def _relief_field(rng, S, cell):
    """Поле высот карты. Размах растёт с размером: на 10 км это сотни метров, на боевой карте
    десятки. Раньше он был один и тот же (97 м и там, и там), и театр читался столом."""
    n = max(32, int(round(S / cell)))
    # Число октав считается от РАЗМЕРА КАРТЫ так, чтобы самая мелкая складка была около двухсот
    # метров на любой карте. С постоянным числом октав на боевой карте появлялась рябь с шагом
    # в сорок метров — это не рельеф, а шум, и он рубил видимость до 13% при норме пула 25-85%.
    oct_base = int(np.clip(round(math.log(max(S / 200.0, 2.0), 2)), 3, 7))
    base = _fractal(rng, n, octaves=oct_base, persist=0.55)
    ridge = _fractal(rng, n, octaves=max(2, oct_base - 2), persist=0.5, ridged=True)
    field = 0.75 * base + 0.25 * ridge
    field = field - float(field.mean())
    # Задаём РАЗМАХ, а не множитель: у фрактала он от затравки пляшет вдвое, и карта то горная,
    # то плоская. Двадцать метров на километр плюс сорок — обычная холмистая местность:
    # 2.5 км -> ~56 м, 10 км -> ~165 м.
    # Подобрано замером видимости, а не на глаз: при размахе 96 м на боевой карте она падала до
    # 13% при норме пула 25-85% — рельеф закрывал всё. Двенадцать метров на километр плюс
    # двадцать пять дают 33% на боевой карте и 34% на театре.
    target = 0.012 * S + 25.0
    span = float(field.max() - field.min()) or 1.0
    return (field * (target / span)).astype(np.float32)


def _fill_pits(h, eps=0.02, rounds=300):
    """Залить замкнутые понижения — то, что вода сделала бы сама.

    Без этого жадный спуск застревает в котловине и часами бродит по ней кругами: на театре
    выходило русло в пятьдесят километров при диагонали в четырнадцать. На залитой поверхности
    спуск монотонен, и река идёт от истока к краю без петель.

    Способ обычный для гидрологии: начинаем с «воды по горлышко» всюду, кроме краёв, и
    итерациями опускаем уровень до максимума из собственной высоты и минимума соседей."""
    w = np.full_like(h, float(h.max()) + 1000.0)
    w[0, :] = h[0, :]
    w[-1, :] = h[-1, :]
    w[:, 0] = h[:, 0]
    w[:, -1] = h[:, -1]
    for _ in range(rounds):
        nb = np.full_like(w, np.inf)
        nb[1:, :] = np.minimum(nb[1:, :], w[:-1, :])
        nb[:-1, :] = np.minimum(nb[:-1, :], w[1:, :])
        nb[:, 1:] = np.minimum(nb[:, 1:], w[:, :-1])
        nb[:, :-1] = np.minimum(nb[:, :-1], w[:, 1:])
        new = np.maximum(h, np.minimum(w, nb + eps))
        if np.max(np.abs(new - w)) < 1e-3:
            w = new
            break
        w = new
    return w


def _river_by_descent(height, cell, rng, max_steps=4000):
    """Русло по СПУСКУ, а не наискось через карту.

    Река, проведённая по линейке, лезет через гребни, и рельеф перестаёт что-либо объяснять.
    Здесь она идёт из высокого места вниз по склону, а из ям выбирается подъёмом уровня — так
    же, как настоящая вода заполняет котловину и переливается через край."""
    n, m = height.shape
    h = _fill_pits(height)                         # по залитой поверхности спуск монотонен
    # Исток — в верхней четверти высот и В СЕРЕДИНЕ карты: взятый у края, он через три шага
    # утыкается в границу, и реки не получается вовсе (замер: русло в 100 метров).
    lo_i, hi_i = int(n * 0.2), int(n * 0.8)
    lo_j, hi_j = int(m * 0.2), int(m * 0.8)
    inner = h[lo_i:hi_i, lo_j:hi_j]
    thr = float(np.percentile(inner, 88))
    cand = np.argwhere(inner >= thr) + (lo_i, lo_j)
    i, j = cand[int(rng.integers(0, len(cand)))]
    path = [(i, j)]
    pdi = pdj = 0                                  # прошлый шаг: по нему идёт инерция
    for _ in range(max_steps):
        score, bh, bi, bj = None, 0.0, i, j
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if not (0 <= ni < n and 0 <= nj < m):
                    return path                    # дошли до края карты — устье
                # Инерция: продолжать в ту же сторону дешевле. Без неё жадный спуск петляет,
                # и на карте 10 км выходит русло в сорок километров — вчетверо длиннее диагонали.
                keep = 0.30 if (di == pdi and dj == pdj) else 0.0
                v = h[ni, nj] + rng.uniform(0.0, 0.10) - keep
                if score is None or v < score:
                    score, bh, bi, bj = v, h[ni, nj], ni, nj
        pdi, pdj = bi - i, bj - j
        if bh >= h[i, j]:
            # Яма: поднимаем уровень заметным шагом. Сравниваем ЧИСТЫЕ высоты, а не оценку с
            # инерцией, — иначе «застрял» срабатывает на каждом шагу, вода заливает всю округу
            # и русло начинает бродить кругами.
            h[i, j] = bh + 0.5
        i, j = bi, bj
        path.append((i, j))
        if i <= 0 or j <= 0 or i >= n - 1 or j >= m - 1:
            return path
    # ходы кончились — выводим к ближайшему краю по прямой, чтобы река не обрывалась в поле
    ei = 0 if i < n - 1 - i else n - 1
    ej = 0 if j < m - 1 - j else m - 1
    if abs(i - ei) < abs(j - ej):
        path += [(int(i + (ei - i) * t), j) for t in np.linspace(0.1, 1.0, 12)]
    else:
        path += [(i, int(j + (ej - j) * t)) for t in np.linspace(0.1, 1.0, 12)]
    return path


def _thin_path(path, cell, min_step_m=90.0, smooth=2):
    """Проредить ломаную и сгладить углы.

    Спуск идёт по клеткам восемью направлениями, поэтому русло получается с изломами по 45
    градусов — на карте это читается как ломаная, а не как река. Пара проходов скользящего
    среднего по узлам убирает излом, оставляя форму."""
    out = [path[0]]
    for p in path[1:]:
        if math.hypot((p[0] - out[-1][0]) * cell, (p[1] - out[-1][1]) * cell) >= min_step_m:
            out.append(p)
    if out[-1] != path[-1]:
        out.append(path[-1])
    pts = [[float(i * cell), float(j * cell)] for i, j in out]
    for _ in range(smooth):
        if len(pts) < 3:
            break
        pts = ([pts[0]]
               + [[(a[0] + 2 * b[0] + c[0]) / 4.0, (a[1] + 2 * b[1] + c[1]) / 4.0]
                  for a, b, c in zip(pts, pts[1:], pts[2:])]
               + [pts[-1]])
    return [[round(x, 1), round(y, 1)] for x, y in pts]


def generate(size_m=2550.0, seed=0, battle_edges=True):
    """Вектор местности плюс карта высот. battle_edges: держать полосы развёртывания (300 м от
    нижнего и верхнего края) чистыми — иначе тот, кто там начинает, стартует в готовой крепости.

    ИЕРАРХИЯ МАСШТАБОВ — главное здесь. Прежний генератор задавал плотности на км² и раскладывал
    всё одного размера: четырнадцать сёл по четыре дома, сотня одинаковых лесков, три холма и
    размах высот 97 метров что на 2.5 км, что на 10. Из-за этого большая карта была маленькой,
    повторённой шестнадцать раз, и десять километров ощущались как два с половиной. Теперь у
    каждого слоя свои размеры: город против хутора, магистраль против просёлка, лесной массив
    против рощи, а размах высот растёт с размером карты."""
    rng = np.random.default_rng(seed)
    S = float(size_m)
    shapes = []
    area_km2 = (S / 1000.0) ** 2
    band = 300.0 if battle_edges else 0.0
    cell = vectormap.default_height_cell((S, S))

    # 1. РЕЛЬЕФ ПЕРВЫМ. Раньше он шёл последним и подгонялся под уже нарисованную реку; теперь
    # наоборот — вода ищет низину в готовом рельефе, как и положено.
    height = _relief_field(rng, S, cell)

    # 2. ВОДА ПО СПУСКУ. Река, проведённая наискось по линейке, лезет через гребни, и рельеф
    # перестаёт что-либо объяснять. Здесь она течёт вниз по склону от истока к краю карты.
    river = _clamp_pts(_thin_path(_river_by_descent(height, cell, rng), cell), S)
    river_w = float(np.clip(0.006 * S, 12.0, 90.0))       # на театре река шире, чем на поле боя
    shapes.append({"kind": "line", "type": "water", "width_m": round(river_w, 1),
                   "points": river})

    # 3. ДОРОГИ ИЕРАРХИЕЙ: магистраль, районные, улицы города. Раньше все были одной ширины, и
    # десять километров дорожной сети выглядели как одна дорога, размноженная копиями.
    roads = []

    def add_road(pts, width):
        roads.append(pts)
        shapes.append({"kind": "line", "type": "road", "width_m": round(float(width), 1),
                       "points": pts})

    n_main = 1 if S < 6000 else 2
    for k in range(n_main):
        t = (k + 0.5) / n_main
        if rng.random() < 0.5:
            p0, p1 = [0.0, t * S], [S, float(np.clip(t * S + rng.uniform(-0.2, 0.2) * S, 0, S))]
        else:
            p0, p1 = [t * S, 0.0], [float(np.clip(t * S + rng.uniform(-0.2, 0.2) * S, 0, S)), S]
        add_road(_clamp_pts(_wander(rng, p0, p1, 5, S * 0.02), S), rng.uniform(14, 18))

    n_long = max(1, int(round(S / 1000.0 / 2.6)))
    n_cross = max(2, int(round(S / 1000.0 / 2.0)))
    rd = np.array(river[-1]) - np.array(river[0])
    ang = math.atan2(rd[1], rd[0])
    for k in range(n_long):
        frac = (k + 0.5) / n_long - 0.5
        off = frac * 1.6 * S + rng.uniform(-0.05, 0.05) * S
        px, py = -math.sin(ang) * off, math.cos(ang) * off
        p0 = [float(np.clip(river[0][0] + px, 0, S)), float(np.clip(river[0][1] + py, 0, S))]
        p1 = [float(np.clip(river[-1][0] + px, 0, S)), float(np.clip(river[-1][1] + py, 0, S))]
        add_road(_clamp_pts(_wander(rng, p0, p1, 6, S * 0.03), S), rng.uniform(8, 11))
    for k in range(n_cross):
        t = (k + 0.5 + rng.uniform(-0.15, 0.15)) / n_cross
        c = np.array(river[0]) + (np.array(river[-1]) - np.array(river[0])) * t
        nv = np.array([-math.sin(ang), math.cos(ang)])
        p0 = _clamp_pts([c + nv * S * 0.8], S)[0]
        p1 = _clamp_pts([c - nv * S * 0.8], S)[0]
        add_road(_clamp_pts(_wander(rng, p0, p1, 6, S * 0.025), S), rng.uniform(7, 10))

    # 4. ПЕРЕПРАВЫ — на каждом пересечении дороги с рекой. Без них река делит карту пополам.
    crossings = []
    for r in roads:
        for hit in _seg_hits(r, river):
            crossings.append(hit)
            # ширину проезда и длину моста не задаём: они выводятся из ширины дороги и реки
            # в этом месте (vectormap.crossing_geom) — мост должен быть узким местом
            shapes.append({"kind": "crossing", "point": [round(hit[0], 1), round(hit[1], 1)]})

    # 5. НАСЕЛЁННЫЕ ПУНКТЫ ТРЁХ РАЗМЕРОВ. Раньше их было два, и на театре выходило четырнадцать
    # одинаковых хуторов по четыре дома на сто квадратных километров — то есть пусто. Город даёт
    # то, чего на маленькой карте быть не может.
    knots = []
    for i in range(len(roads)):
        for j in range(i + 1, len(roads)):
            knots += _seg_hits(roads[i], roads[j])
    # Мест под посёлки берём с запасом: одни перекрёстки — это два десятка точек на театр, и
    # хутора ставить оказывается некуда. Добавляем точки вдоль дорог через каждые полкилометра.
    rng.shuffle(knots)
    spots = list(knots)
    for r in roads:
        acc = 0.0
        for k in range(1, len(r)):
            acc += math.hypot(r[k][0] - r[k - 1][0], r[k][1] - r[k - 1][1])
            if acc > 500.0:
                acc = 0.0
                spots.append((float(r[k][0]), float(r[k][1])))
    rng.shuffle(spots)

    n_town = int(round(area_km2 / 45.0))
    n_village = max(1, int(round(area_km2 / 9.0)))
    n_hamlet = max(1, int(round(area_km2 / 3.0)))
    plan = ([("town", int(rng.integers(26, 60))) for _ in range(n_town)]
            + [("village", int(rng.integers(9, 17))) for _ in range(n_village)]
            + [("hamlet", int(rng.integers(2, 5))) for _ in range(n_hamlet)])
    used = []
    settlements = {"town": 0, "village": 0, "hamlet": 0}
    for kind, houses in plan:
        keep = 900.0 if kind == "town" else (450.0 if kind == "village" else 220.0)
        spot = None
        for c in spots:
            if all((c[0] - u[0]) ** 2 + (c[1] - u[1]) ** 2 > keep * keep for u in used):
                spot = c
                break
        if spot is None:
            continue
        used.append(spot)
        spots.remove(spot)
        kx, ky = spot
        if battle_edges and not (band < ky < S - band):
            continue
        settlements[kind] += 1
        host = min(roads, key=lambda r: min((p[0] - kx) ** 2 + (p[1] - ky) ** 2 for p in r))
        idx = int(np.argmin([(p[0] - kx) ** 2 + (p[1] - ky) ** 2 for p in host]))
        nxt = host[min(idx + 1, len(host) - 1)]
        dirv = np.array(nxt) - np.array(host[idx])
        ln = float(np.linalg.norm(dirv)) or 1.0
        dirv = dirv / ln
        perp = np.array([-dirv[1], dirv[0]])
        # город растёт кварталами вдоль нескольких улиц, село — лентой вдоль одной, хутор — кучкой
        streets = 3 if kind == "town" else 1
        per_street = max(1, houses // streets)
        for st in range(streets):
            base = np.array([kx, ky]) + perp * ((st - (streets - 1) / 2) * rng.uniform(70, 110))
            if kind == "town" and st:
                # улица города — тоже дорога, иначе кварталы висят в чистом поле
                a = base - dirv * per_street * 30.0
                b = base + dirv * per_street * 30.0
                add_road(_clamp_pts([list(a), list(b)], S), rng.uniform(5, 7))
            for hidx in range(per_street):
                along = (hidx - per_street / 2) * rng.uniform(38, 62)
                sidep = (1 if hidx % 2 else -1) * rng.uniform(20, 38)
                pnt = base + dirv * along + perp * sidep
                if not (0 < pnt[0] < S and 0 < pnt[1] < S):
                    continue
                w_m, h_m = _house_size(rng, kind)
                shapes.append({"kind": "building",
                               "rect_m": [round(float(pnt[0]), 1), round(float(pnt[1]), 1),
                                          round(w_m, 1), round(h_m, 1),
                                          round(float(math.degrees(math.atan2(dirv[1], dirv[0]))
                                                      + rng.uniform(-10, 10)), 1)],
                               "capacity": 1})

    # 6. ЛЕС ДВУХ РАЗМЕРОВ ПЛЮС МЕЖИ. Прежде все пятна были одного радиуса, и лес на театре
    # читался как рассыпанный горох.
    def far_from_roads(p, d):
        for r in roads:
            for q in r:
                if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < d * d:
                    return False
        return True

    n_massif = int(round(area_km2 / 22.0))
    n_grove = max(3, int(round(area_km2 * rng.uniform(0.8, 1.3))))
    n_hedge = max(2, int(round(area_km2 * 0.7)))
    woods = ([("massif", float(rng.uniform(600, 1200))) for _ in range(n_massif)]
             + [("grove", float(rng.uniform(120, 340))) for _ in range(n_grove)])
    placed = 0
    for kind, radius in woods:
        keepout = 400.0 if kind == "massif" else 150.0
        for _ in range(24):
            if rng.random() < 0.45 and len(river) > 2:     # прижатый к реке массив
                t = rng.uniform(0.1, 0.9)
                i = min(int(t * (len(river) - 1)), len(river) - 2)
                base = np.array(river[i])
                nv = np.array([-(river[i + 1][1] - river[i][1]), river[i + 1][0] - river[i][0]])
                nv = nv / (np.linalg.norm(nv) or 1.0)
                c = base + nv * rng.uniform(0.02, 0.09) * S * (1 if rng.random() < 0.5 else -1)
            else:
                c = rng.uniform(0.08, 0.92, size=2) * S
            if battle_edges and not (band < c[1] < S - band):
                continue
            if not far_from_roads(c, keepout):
                continue
            shapes.append({"kind": "polygon", "type": "forest",
                           "points": [[round(x, 1), round(y, 1)]
                                      for x, y in _clamp_pts(_blob(rng, c[0], c[1], radius), S)]})
            placed += 1
            break

    for _ in range(n_hedge):
        p0 = rng.uniform(0.05, 0.95, size=2) * S
        d = rng.uniform(-1, 1, size=2)
        d = d / (np.linalg.norm(d) or 1.0)
        p1 = p0 + d * rng.uniform(300.0, 900.0)
        shapes.append({"kind": "line", "type": "forest",
                       "width_m": round(float(rng.uniform(18, 32)), 1),
                       "points": _clamp_pts(_wander(rng, p0, p1, 3, S * 0.012), S)})

    doc = vectormap.new_doc((S, S), shapes)
    doc["height"] = {"cell_m": cell, "h": height}
    # ДОЛИНА врезается в готовое поле высот: русло уже лежит в низине, но берега надо оформить,
    # иначе река идёт по плоскому и не читается как река.
    vectormap.stamp(doc, [{"kind": "line", "type": "relief", "points": river,
                           "h_m": -float(np.clip(0.0022 * S, 6.0, 30.0)),
                           "width_m": river_w * 3.0}],
                    cell_m=cell, slope_m=float(np.clip(0.03 * S, 120.0, 400.0)))
    return doc, {"crossings": len(crossings), "villages": settlements["village"],
                 "towns": settlements["town"], "hamlets": settlements["hamlet"],
                 "forests": placed, "relief_m": float(height.max() - height.min())}


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
        surface, fields, *_ = vectormap.rasterize(doc, cell_m)
        m = measure(surface, cell_m, n_pairs=300, fields=fields)
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
        surface, fields, *_ = vectormap.rasterize(piece, cell_m)
        m = measure(surface, cell_m, n_pairs=300, fields=fields)
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
