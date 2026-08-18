"""Векторная карта — ИСТОЧНИК местности. Из неё считаются поля свойств и граф дорог.

Зачем вектор, если бой всё равно читает сетку. Сетка отвечает на вопрос «сколько здесь
укрытия», и для этого она хороша. Но описывать местность ею плохо: дорога шириной восемь
метров в тридцатиметровой клетке превращается в тридцатиметровую полосу, тонкая лесополоса
либо исчезает, либо становится стеной, а село слипается в одно пятно. Всё это уже кусало
проект (docs/JOURNAL.md, п. 3.5). Поэтому:

    вектор (что где)  ->  поля (сколько чего)  ->  бой
                      ->  граф дорог           ->  дальние маршруты

Вектор — единственное, что нельзя потерять. Поля и граф пересобираются одной командой, в том
числе под ДРУГОЙ размер клетки: разрешение перестаёт быть решением, зашитым в данные.

Формат (json, координаты в МЕТРАХ, y растёт вверх — как в бою):

    {
      "version": 1,
      "size_m": [2550, 2550],
      "shapes": [
        {"kind": "polygon",  "type": "forest", "points": [[x, y], ...]},
        {"kind": "line",     "type": "road",   "width_m": 8,  "points": [[x, y], ...]},
        {"kind": "line",     "type": "water",  "width_m": 25, "points": [[x, y], ...]},
        {"kind": "building", "rect_m": [cx, cy, w, h, angle_deg], "capacity": 1},
        {"kind": "crossing", "point": [x, y], "width_m": 30}
      ]
    }

Растеризация идёт с ЧЕТЫРЁХКРАТНОЙ подвыборкой и порогами по доле покрытия, а не «попал ли
центр клетки». Иначе тонкие объекты пропадают через раз, а дорога, чиркнувшая по лесу, стирает
лес в целой клетке — ровно та ошибка, что чинилась в maps/segment_reference.py.

    py -3.12 vectormap.py maps/demo.vector.json --cell 15     # собрать поля и граф
"""
import argparse
import json
import math
import os

import numpy as np

import terrain

SUB = 4                      # подвыборка при растеризации: клетка делится на SUB×SUB
# Пороги доли покрытия клетки и приоритет. Дорога — САМАЯ слабая (кроме открытого): она тонкая,
# порог у неё низкий, и пустить её вперёд леса значит стереть лес любой тропинкой.
#
# «open» — не фон, а ЛАСТИК: поляна в лесу, просека, выгон у села. Фон и так открытый, но
# вырезать дырку в уже нарисованном массиве иначе нечем, кроме правки узлов. Порог у него
# высокий (половина клетки): поляна должна съесть клетку целиком, а не обкусывать края.
THRESHOLDS = {"road": 0.05, "forest": 0.15, "open": 0.50, "water": 0.30, "building": 0.03}
PRIORITY = ("road", "forest", "open", "water", "building")   # последний главнее


def _types():
    types, *_ = terrain._type_tables()
    return {name: t["id"] for name, t in types.items()}


# ---------------------------------------------------------------- формат


def new_doc(size_m, shapes=None):
    return {"version": 1, "size_m": [float(size_m[0]), float(size_m[1])], "shapes": shapes or []}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version") != 1:
        raise ValueError(f"{path}: версия формата {doc.get('version')}, поддерживается 1")
    return doc


def save(doc, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------- растеризация


def _blank(gx, gy):
    return np.zeros((gy * SUB, gx * SUB), dtype=bool)      # [row, col], row = y сверху вниз


def _fill_polygon(mask, pts_px):
    """Заливка многоугольника чётно-нечётным правилом. Без PIL: растеризация нужна и там, где
    картинок нет вовсе (обучение, сборка на сервере), тащить ради неё зависимость незачем."""
    if len(pts_px) < 3:
        return
    ys = [p[1] for p in pts_px]
    y0 = max(0, int(math.floor(min(ys))))
    y1 = min(mask.shape[0] - 1, int(math.ceil(max(ys))))
    n = len(pts_px)
    for row in range(y0, y1 + 1):
        yc = row + 0.5
        xs = []
        for i in range(n):
            ax, ay = pts_px[i]
            bx, by = pts_px[(i + 1) % n]
            if (ay > yc) != (by > yc):
                xs.append(ax + (yc - ay) * (bx - ax) / (by - ay))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            c0 = max(0, int(math.ceil(xs[k] - 0.5)))
            c1 = min(mask.shape[1] - 1, int(math.floor(xs[k + 1] - 0.5)))
            if c1 >= c0:
                mask[row, c0:c1 + 1] = True


def _stamp_disc(mask, cx, cy, r):
    if r <= 0:
        return
    x0, x1 = max(0, int(cx - r - 1)), min(mask.shape[1] - 1, int(cx + r + 1))
    y0, y1 = max(0, int(cy - r - 1)), min(mask.shape[0] - 1, int(cy + r + 1))
    if x1 < x0 or y1 < y0:
        return
    xs = np.arange(x0, x1 + 1) + 0.5
    ys = np.arange(y0, y1 + 1) + 0.5
    dx = xs[None, :] - cx
    dy = ys[:, None] - cy
    mask[y0:y1 + 1, x0:x1 + 1] |= (dx * dx + dy * dy) <= r * r


def _fill_thick_line(mask, pts_px, width_px):
    """Полоса заданной ширины вдоль ломаной. Круги в узлах — чтобы на изломах не было щелей:
    именно щели в изгородях у Eugen жалуются как «поддельное укрытие»."""
    r = max(width_px / 2.0, 0.5)
    for (ax, ay), (bx, by) in zip(pts_px[:-1], pts_px[1:]):
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy)
        if ln < 1e-9:
            continue
        nx, ny = -dy / ln * r, dx / ln * r
        _fill_polygon(mask, [(ax + nx, ay + ny), (bx + nx, by + ny),
                             (bx - nx, by - ny), (ax - nx, ay - ny)])
    for (px, py) in pts_px:
        _stamp_disc(mask, px, py, r)


def _dist_point_seg(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln2 = dx * dx + dy * dy
    t = 0.0 if ln2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ln2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def crossing_geom(doc, sh):
    """Геометрия переправы: (центр, длина, ширина, угол) — ОТРЕЗОК поперёк воды, а не круг.

    Мост — узкое место, за которое дерутся; круглая дырка в реке превращает его в брод во всю
    ширину и стирает единственный настоящий чокпойнт на карте. Поэтому ширина берётся от
    дороги, а длина — от ширины реки в этом месте, с небольшим запасом на берега.

    Направление и размеры выводятся из соседних фигур, если явно не заданы: так исправляются
    и старые карты, где у переправы был только радиус."""
    p = sh["point"]
    angle = sh.get("angle_deg")
    width = sh.get("width_m_road")
    length = sh.get("length_m")

    best_road, best_water = None, None
    for s in doc["shapes"]:
        if s["kind"] != "line":
            continue
        for a, b in zip(s["points"][:-1], s["points"][1:]):
            d = _dist_point_seg(p, a, b)
            if s["type"] == "road" and (best_road is None or d < best_road[0]):
                best_road = (d, a, b, float(s.get("width_m", 8.0)))
            elif s["type"] == "water" and (best_water is None or d < best_water[0]):
                best_water = (d, a, b, float(s.get("width_m", 30.0)))

    if angle is None:
        if best_road:
            _d, a, b, _w = best_road
            angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        else:
            angle = 0.0
    if width is None:
        width = best_road[3] if best_road else 8.0
    if length is None:
        river_w = best_water[3] if best_water else float(sh.get("width_m", 40.0))
        length = river_w * 1.8 + 20.0          # запас, чтобы гарантированно пробить оба берега
    return p, float(length), float(width), float(angle)


def dash_polyline(pts, on_m, off_m):
    """Разбить ломаную на штрихи: [[точки штриха], ...].

    Нужно для редкой лесополосы и живой изгороди с прорехами. Сплошная полоса леса шириной в
    90 м перекрывает обзор наглухо, а настоящая посадка дырявая: местами простреливается,
    местами нет. Прорехи — это не украшение, а тактика: через них проходят и стреляют."""
    if on_m <= 0:
        return [pts]
    out, cur = [], [tuple(pts[0])]
    drawing, left = True, float(on_m)
    for a, b in zip(pts[:-1], pts[1:]):
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        seg = math.hypot(bx - ax, by - ay)
        t = 0.0
        while seg - t > 1e-9:
            step = min(left, seg - t)
            t += step
            left -= step
            p = (ax + (bx - ax) * t / seg, ay + (by - ay) * t / seg)
            if drawing:
                cur.append(p)
            if left <= 1e-9:                       # штрих или промежуток кончился — меняем фазу
                if drawing and len(cur) > 1:
                    out.append(cur)
                drawing = not drawing
                left = float(off_m if not drawing else on_m)
                cur = [p] if drawing else []
    if drawing and len(cur) > 1:
        out.append(cur)
    return out


def _rect_points(cx, cy, w, h, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for sx, sy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        dx, dy = sx * w, sy * h
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def rasterize(doc, cell_m):
    """Вектор -> (сетка типов, поля, компоненты зданий, вместимости).

    Оси: вектор в метрах с y ВВЕРХ, сетка индексируется [gx, gy] с тем же y вверх. Внутри
    растеризуем в картинку с y вниз и переворачиваем один раз в самом конце — переворачивать
    в нескольких местах уже пробовали, это стоило зеркальной карты."""
    tid = _types()
    W, H = float(doc["size_m"][0]), float(doc["size_m"][1])
    gx, gy = int(round(W / cell_m)), int(round(H / cell_m))
    px_m = cell_m / SUB                                   # метров в подпикселе

    def to_px(p):
        return (p[0] / px_m, (H - p[1]) / px_m)           # y вниз для растеризации

    masks = {k: _blank(gx, gy) for k in THRESHOLDS}
    buildings = []
    for sh in doc["shapes"]:
        kind = sh["kind"]
        if kind == "polygon":
            _fill_polygon(masks[sh["type"]], [to_px(p) for p in sh["points"]])
        elif kind == "line":
            width_px = float(sh.get("width_m", cell_m)) / px_m
            dash = sh.get("dash_m")               # [штрих, промежуток] в метрах
            runs = (dash_polyline(sh["points"], dash[0], dash[1]) if dash else [sh["points"]])
            for run in runs:
                _fill_thick_line(masks[sh["type"]], [to_px(p) for p in run], width_px)
        elif kind == "building":
            cx, cy, w, h, ang = sh["rect_m"]
            _fill_polygon(masks["building"], [to_px(p) for p in _rect_points(cx, cy, w, h, ang)])
            buildings.append({"rect_m": list(sh["rect_m"]), "capacity": int(sh.get("capacity", 1))})
        elif kind == "crossing":
            pass                                          # обрабатывается ниже, поверх воды
        else:
            raise ValueError(f"неизвестная фигура: {kind}")

    # переправы: мост или брод пробивает воду и кладёт поверх дорогу. Без них река делит карту
    # пополам, и половина местности недостижима — самая частая ошибка при рисовании.
    for sh in doc["shapes"]:
        if sh["kind"] != "crossing":
            continue
        (px_, py_), length, width, angle = crossing_geom(doc, sh)
        hole = _blank(gx, gy)
        _fill_polygon(hole, [to_px(q) for q in _rect_points(px_, py_, length, width, angle)])
        masks["water"] &= ~hole
        masks["road"] |= hole

    # доля покрытия клетки: подпиксели SUB×SUB сворачиваются усреднением
    def frac(mask):
        return mask.reshape(gy, SUB, gx, SUB).mean(axis=(1, 3))

    surf_img = np.full((gy, gx), tid["open"], dtype=np.int8)
    for name in PRIORITY:
        surf_img[frac(masks[name]) > THRESHOLDS[name]] = tid[name]
    surface = np.ascontiguousarray(surf_img[::-1].T)      # -> [gx, gy], y вверх

    # дома — ОБЪЕКТЫ: компоненты известны из вектора, искать связные пятна клеток не нужно
    comp = np.zeros((gx, gy), dtype=np.int32)
    capacity = {}
    for i, b in enumerate(buildings, start=1):
        m = _blank(gx, gy)
        cx, cy, w, h, ang = b["rect_m"]
        _fill_polygon(m, [to_px(p) for p in _rect_points(cx, cy, w, h, ang)])
        cells = np.ascontiguousarray((frac(m) > THRESHOLDS["building"])[::-1].T)
        cells &= surface == tid["building"]
        comp[cells] = i
        capacity[i] = b["capacity"]
    # дом, целиком накрытый другим (в редакторе это легко сделать случайно), теряет все свои
    # клетки: перекрытие достаётся верхнему. Вместимость такого компонента — мусор, убираем.
    alive = set(int(c) for c in np.unique(comp) if c > 0)
    capacity = {c: v for c, v in capacity.items() if c in alive}
    # клетки застройки, не попавшие ни в один дом (например от полигона «квартал»), нумеруются
    # как раньше — связностью внутри квартала, чтобы село не слиплось в одно строение
    rest = (surface == tid["building"]) & (comp == 0)
    if rest.any():
        tmp = np.where(rest, tid["building"], tid["open"]).astype(np.int8)
        extra, n_extra = terrain._building_components(tmp, tid["building"])
        base = comp.max()
        comp[rest] = extra[rest] + base
        for k in range(1, n_extra + 1):
            capacity[int(base + k)] = 1

    fields = terrain.fields_from_surface(surface)

    # ДОРОГА ПОД ЛЕСОМ. У клетки один тип, и на пересечении лес побеждает дорогу — иначе любая
    # тропинка стирала бы массив. Но в жизни просека это и укрытие, и быстрый ход одновременно,
    # а тип у неё может быть только один. Разводим по полям: скорость берём дорожную, укрытие и
    # непрозрачность оставляем лесные. Ровно ради этого клетка и перестала быть меткой.
    types, *_ = terrain._type_tables()
    road_mask = (frac(masks["road"]) > THRESHOLDS["road"])
    road_mask = np.ascontiguousarray(road_mask[::-1].T)
    road_mask &= (surface == tid["forest"])          # по воде и застройке скорость не меняем
    fields["speed_foot"][road_mask] = types["road"]["speed"]
    fields["speed_veh"][road_mask] = types["road"].get("speed_vehicle", types["road"]["speed"])

    return surface, fields, comp, capacity


# ---------------------------------------------------------------- вырезка


def _clip_polygon(pts, w, h):
    """Отсечение многоугольника прямоугольником 0..w × 0..h (Сазерленд-Ходжман).
    Прямоугольник выпуклый, поэтому алгоритм корректен без оговорок."""
    edges = ((lambda p: p[0] >= 0, 0, 0), (lambda p: p[0] <= w, 0, w),
             (lambda p: p[1] >= 0, 1, 0), (lambda p: p[1] <= h, 1, h))
    poly = list(pts)
    for inside, axis, val in edges:
        if not poly:
            return []
        out = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            ia, ib = inside(a), inside(b)
            if ia != ib:
                d = b[axis] - a[axis]
                t = 0.0 if abs(d) < 1e-12 else (val - a[axis]) / d
                cut = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if ia:
                out.append(a)
                if not ib:
                    out.append(cut)
            elif ib:
                out.append(cut)
        poly = out
    return poly


def _clip_polyline(pts, w, h):
    """Ломаная, отсечённая прямоугольником, — СПИСОК кусков: линия может входить и выходить
    несколько раз, и склеивать её обратно нельзя, иначе появится дорога через угол карты."""
    def inside(p):
        return -1e-9 <= p[0] <= w + 1e-9 and -1e-9 <= p[1] <= h + 1e-9

    def clip_seg(a, b):
        """Часть отрезка внутри прямоугольника (Лианг-Барски) или None."""
        t0, t1 = 0.0, 1.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        for p, q in ((-dx, a[0]), (dx, w - a[0]), (-dy, a[1]), (dy, h - a[1])):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)
        if t0 > t1:
            return None
        return ((a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy))

    runs, cur = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = clip_seg(a, b)
        if seg is None:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        s, e = seg
        if not cur:
            cur = [s, e]
        else:
            if (cur[-1][0] - s[0]) ** 2 + (cur[-1][1] - s[1]) ** 2 > 1e-6:
                if len(cur) > 1:
                    runs.append(cur)
                cur = [s, e]
            else:
                cur.append(e)
        if not inside(b):
            runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return [r for r in runs if len(r) > 1]


def crop(doc, center_m, size_m, angle_deg=0.0):
    """Кусок карты в СВОЕЙ системе координат: повернули, сдвинули, обрезали.

    Поворот здесь бесплатный и в этом главный смысл вектора: растровую карту под углом резать
    нельзя без пересэмплинга и лесенки, а фигуры просто поворачиваются. Один нарисованный
    театр даёт сколько угодно непохожих боевых карт — место, угол, зеркало."""
    w, h = float(size_m[0]), float(size_m[1])
    a = math.radians(-angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = center_m

    def T(p):
        dx, dy = p[0] - cx, p[1] - cy
        return (dx * ca - dy * sa + w / 2, dx * sa + dy * ca + h / 2)

    shapes = []
    for sh in doc["shapes"]:
        if sh["kind"] == "polygon":
            poly = _clip_polygon([T(p) for p in sh["points"]], w, h)
            if len(poly) >= 3:
                shapes.append({"kind": "polygon", "type": sh["type"],
                               "points": [[round(x, 1), round(y, 1)] for x, y in poly]})
        elif sh["kind"] == "line":
            for run in _clip_polyline([T(p) for p in sh["points"]], w, h):
                shapes.append({"kind": "line", "type": sh["type"],
                               "width_m": sh.get("width_m", 8.0),
                               "points": [[round(x, 1), round(y, 1)] for x, y in run]})
        elif sh["kind"] == "building":
            bx, by, bw, bh, ang = sh["rect_m"]
            nx, ny = T((bx, by))
            if -bw <= nx <= w + bw and -bh <= ny <= h + bh:
                shapes.append({"kind": "building",
                               "rect_m": [round(nx, 1), round(ny, 1), bw, bh,
                                          round(ang - angle_deg, 1)],
                               "capacity": sh.get("capacity", 1)})
        elif sh["kind"] == "crossing":
            nx, ny = T(sh["point"])
            if 0 <= nx <= w and 0 <= ny <= h:
                shapes.append({"kind": "crossing", "point": [round(nx, 1), round(ny, 1)],
                               "width_m": sh.get("width_m", 30.0)})
    return new_doc((w, h), shapes)


# ---------------------------------------------------------------- граф дорог


def _seg_intersect(a, b, c, d):
    """Пересечение отрезков -> (точка, доля вдоль первого, доля вдоль второго) или None.

    Доли обязательны: по ним перекрёсток встаёт в СВОЮ позицию на ломаной. Раньше он писался
    в середину сегмента, и участок дороги получал неверную длину — на извилистой дороге путь
    выходил короче прямой между узлами, что и вскрыла проверка."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = a, b, c, d
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1)), max(0.0, min(1.0, t)), max(0.0, min(1.0, u))
    return None


def road_graph(doc, snap_m=20.0):
    """Дорожная сеть графом: узлы — концы и перекрёстки, рёбра — участки между ними.

    Нужен для ДАЛЬНЕЙ навигации: марш через полкарты и подход резервов считаются по десяткам
    узлов за микросекунды, тогда как по сетке это сотни тысяч клеток. Ближний манёвр остаётся
    на сетке — граф говорит «через какие перекрёстки», сетка «как именно идти»."""
    lines = [(sh["points"], float(sh.get("width_m", 8.0))) for sh in doc["shapes"]
             if sh["kind"] == "line" and sh["type"] == "road"]
    nodes = []

    def node_id(p):
        for i, q in enumerate(nodes):
            if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 <= snap_m * snap_m:
                return i
        nodes.append([round(p[0], 1), round(p[1], 1)])
        return len(nodes) - 1

    # точки разбиения на каждой ломаной: концы, перекрёстки с другими дорогами, переправы
    splits = []
    for pts, _w in lines:
        marks = {0: pts[0], len(pts) - 1: pts[-1]}
        splits.append(marks)
    for i, (pts_i, _wi) in enumerate(lines):
        for j, (pts_j, _wj) in enumerate(lines):
            if j <= i:
                continue
            for si in range(len(pts_i) - 1):
                for sj in range(len(pts_j) - 1):
                    hit = _seg_intersect(pts_i[si], pts_i[si + 1], pts_j[sj], pts_j[sj + 1])
                    if hit:
                        pt, t, u = hit
                        splits[i][si + t] = pt
                        splits[j][sj + u] = pt
    for sh in doc["shapes"]:
        if sh["kind"] != "crossing":
            continue
        for i, (pts, _w) in enumerate(lines):
            for si in range(len(pts) - 1):
                ax, ay = pts[si]
                bx, by = pts[si + 1]
                px, py = sh["point"]
                ln2 = (bx - ax) ** 2 + (by - ay) ** 2
                if ln2 < 1e-9:
                    continue
                t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ln2))
                qx, qy = ax + t * (bx - ax), ay + t * (by - ay)
                if (qx - px) ** 2 + (qy - py) ** 2 <= (snap_m * 2) ** 2:
                    splits[i][si + t] = (qx, qy)

    edges = []
    for i, (pts, w) in enumerate(lines):
        keys = sorted(splits[i])
        ids = [node_id(splits[i][k]) for k in keys]
        for k in range(len(keys) - 1):
            a, b = keys[k], keys[k + 1]
            length = _polyline_length(pts, a, b)
            if ids[k] != ids[k + 1] and length > 1e-6:
                edges.append({"a": ids[k], "b": ids[k + 1], "len_m": round(length, 1),
                              "width_m": round(w, 1), "path": _polyline_slice(pts, a, b)})
    return {"nodes": nodes, "edges": edges}


def _point_at(pts, t):
    """Точка на ломаной по дробной позиции: индекс сегмента + доля вдоль него."""
    i = int(math.floor(t))
    i = max(0, min(i, len(pts) - 2))
    f = t - i
    return (pts[i][0] + f * (pts[i + 1][0] - pts[i][0]),
            pts[i][1] + f * (pts[i + 1][1] - pts[i][1]))


def _polyline_slice(pts, a, b):
    """Кусок ломаной между дробными позициями — С ПРОМЕЖУТОЧНЫМИ УЗЛАМИ.

    Нужен, чтобы участок дороги хранил свою настоящую форму, а не только концы: и рисуется он
    тогда по дороге, а не хордой через полкарты, и маршрут по нему пойдёт там, где дорога есть."""
    lo, hi = min(a, b), max(a, b)
    out = [_point_at(pts, lo)]
    k = int(math.floor(lo)) + 1
    while k < hi - 1e-9:
        out.append(tuple(pts[k]))
        k += 1
    out.append(_point_at(pts, hi))
    return [[round(x, 1), round(y, 1)] for x, y in out]


def _polyline_length(pts, a, b):
    total = 0.0
    piece = _polyline_slice(pts, a, b)
    for p, q in zip(piece[:-1], piece[1:]):
        total += math.hypot(q[0] - p[0], q[1] - p[1])
    return total


# ---------------------------------------------------------------- сборка


def build(vector_path, cell_m, out_prefix=None):
    """Вектор -> <имя>.fields.npz + <имя>.map.json. Всё, кроме вектора, — производное:
    можно удалить и пересобрать, в том числе под другой размер клетки."""
    doc = load(vector_path)
    out_prefix = out_prefix or vector_path[:-len(".vector.json")]
    surface, fields, comp, capacity = rasterize(doc, cell_m)
    graph = road_graph(doc)
    np.savez_compressed(out_prefix + ".fields.npz", surface=surface, building_comp=comp,
                        **{k: v for k, v in fields.items()})
    meta = {"cell_m": cell_m, "size_m": doc["size_m"], "source": os.path.basename(vector_path),
            "building_capacity": {str(k): v for k, v in capacity.items()}, "graph": graph}
    with open(out_prefix + ".map.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_prefix, surface, meta


def load_map(prefix, m_per_unit):
    """Собранная карта -> TerrainMap. cell_m делится на m_per_unit: вся арифметика боя идёт в
    ИГРОВЫХ единицах, и карта в метрах схлопнула бы арену в угол сетки (была такая ошибка)."""
    with open(prefix + ".map.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(prefix + ".fields.npz")
    fields = {k: z[k] for k in terrain.FIELDS}
    cap = {int(k): v for k, v in meta.get("building_capacity", {}).items()}
    tm = terrain.from_fields(z["surface"], fields, meta["cell_m"] / m_per_unit, cap, z["building_comp"])
    return tm, meta


def main():
    ap = argparse.ArgumentParser(description="сборка векторной карты в поля и граф дорог")
    ap.add_argument("vector", help="путь к <имя>.vector.json")
    ap.add_argument("--cell", type=float, default=15.0, help="размер клетки полей, метров")
    args = ap.parse_args()

    prefix, surface, meta = build(args.vector, args.cell)
    types = _types()
    names = {v: k for k, v in types.items()}
    u, c = np.unique(surface, return_counts=True)
    print(f"сетка {surface.shape[0]}x{surface.shape[1]} по {args.cell:.0f} м "
          f"= {surface.shape[0] * args.cell:.0f} м")
    print("доли:", {names[int(k)]: round(float(v) / surface.size, 3) for k, v in zip(u, c)})
    print(f"строений: {len(meta['building_capacity'])}, "
          f"узлов дорог: {len(meta['graph']['nodes'])}, участков: {len(meta['graph']['edges'])}")
    print(f"собрано: {prefix}.fields.npz + {prefix}.map.json")


if __name__ == "__main__":
    main()
