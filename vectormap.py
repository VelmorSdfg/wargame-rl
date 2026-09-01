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
RELIEF_SMOOTH = 6            # проходов сглаживания поля высот: холм должен иметь склон, а не борт
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
    """Вектор плюс карта высот, если она есть рядом.

    Старый формат, где рельеф лежал фигурами, переносится здесь же: фигуры один раз вдавливаются
    в карту высот тем же кодом, что считал их раньше, и исчезают. Поэтому уже нарисованные карты
    не меняются, а править их дальше можно кистями."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version") != 1:
        raise ValueError(f"{path}: версия формата {doc.get('version')}, поддерживается 1")
    hp = height_path(path)
    if os.path.exists(hp):
        z = np.load(hp)
        doc["height"] = {"cell_m": float(z["cell_m"]), "h": np.asarray(z["h"], dtype=np.float32)}
    elif any(sh.get("type") == "relief" for sh in doc["shapes"]):
        # Клетку берём ТУ ЖЕ, в которой карта собиралась для боя. Тогда перенос точен до нуля:
        # пересчёт из растра в ту же клетку — тождество. Возьми мельче — и рельеф чуть поедет
        # (радиус сглаживания в старом коде зависел от размера сетки), а вместе с ним поедут
        # линии огня: замер показал расхождение 4% пар видимости.
        cell = _built_cell(path) or default_height_cell(doc["size_m"])
        doc["height"] = {"cell_m": cell, "h": bake_relief(doc, cell)}
        doc["shapes"] = [sh for sh in doc["shapes"] if sh.get("type") != "relief"]
    return doc


def save(doc, path):
    """Вектор в json, карту высот — в npz рядом. В json растру не место: это мегабайт чисел,
    который человеку не читать, а diff по нему бесполезен."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hm = doc.get("height")
    plain = {k: v for k, v in doc.items() if k != "height"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plain, f, ensure_ascii=False, indent=2)
    hp = height_path(path)
    if hm is not None and np.any(hm["h"]):
        np.savez_compressed(hp, h=np.asarray(hm["h"], dtype=np.float32),
                            cell_m=float(hm["cell_m"]))
    elif os.path.exists(hp):
        os.remove(hp)                      # рельеф стёрли — файл не должен остаться призраком
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


def _box_blur(a):
    """Три-на-три среднее без scipy: зависимость ради шести проходов тянуть незачем."""
    out = a.copy()
    out[1:] += a[:-1]; out[:-1] += a[1:]
    out[:, 1:] += a[:, :-1]; out[:, :-1] += a[:, 1:]
    out[1:, 1:] += a[:-1, :-1]; out[:-1, :-1] += a[1:, 1:]
    out[1:, :-1] += a[:-1, 1:]; out[:-1, 1:] += a[1:, :-1]
    return out / 9.0


def _rect_points(cx, cy, w, h, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for sx, sy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        dx, dy = sx * w, sy * h
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def _fill_masks(doc, gx, gy, cell_m, to_px):
    """Заливка фигур в маски по типам. Общая часть полной растеризации и оконной: разойдись они —
    вблизи камеры карта была бы не та, что читает бой."""
    px_m = cell_m / SUB
    masks = {k: _blank(gx, gy) for k in THRESHOLDS}
    buildings = []
    for sh in doc["shapes"]:
        kind = sh["kind"]
        if sh.get("type") == "relief":
            continue                      # рельеф — не материал, он собирается отдельно
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
            pass                                          # обрабатывается отдельно, поверх воды
        else:
            raise ValueError(f"неизвестная фигура: {kind}")
    return masks, buildings


def _apply_crossings(doc, masks, gx, gy, to_px):
    """Переправы пробивают воду. Без них река делит карту пополам, и половина местности
    недостижима — самая частая ошибка при рисовании.

    МОСТ кладёт поверх дорогу, БРОД — нет. Разница не косметическая: дорога у нас быстрее поля
    (1.1 против 1.0, техника 1.5), а брод медленнее всего (0.45, техника 0.3). Пока брод считался
    дорогой, переход реки вброд выходил быстрее, чем обход посуху, — то есть местность врала в
    сторону, обратную здравому смыслу."""
    for sh in doc["shapes"]:
        if sh["kind"] != "crossing":
            continue
        (px_, py_), length, width, angle = crossing_geom(doc, sh)
        hole = _blank(gx, gy)
        _fill_polygon(hole, [to_px(q) for q in _rect_points(px_, py_, length, width, angle)])
        masks["water"] &= ~hole
        if not sh.get("ford"):
            masks["road"] |= hole


def _crossing_cells(doc, gx, gy, cell_m, y_top, want_ford=False, x_left=0.0):
    """Клетки, которые занимает переправа, — их тип назначается ПОВЕРХ долевого спора.

    Почему не долей покрытия, как у всего остального. Клетка имеет ОДИН тип, и тип берётся
    по доле: вода занимает 30% клетки — клетка вода. Мост шириной восемь метров в клетке
    тридцать метров занимает четверть и проигрывает всегда, сколько его ни рисуй. На театре
    из-за этого 10 переправ из 12 выходили непроходимыми, а сеть дорог разрывалась пополам.
    Спор тут неуместен: переправа для того и существует, чтобы объявить клетку проезжей.

    И не растеризацией прямоугольника, а ШАГОМ ПО ОСИ моста. Подпиксель при клетке 30 м —
    это 7.5 м, то есть мост шириной 8.4 м ложится в один подпиксель и местами промахивается
    мимо их центров: одна переправа из двенадцати так и оставалась водой. Ход по оси даёт
    непрерывную полосу клеток при любой ширине — а непрерывность здесь и есть весь смысл.

    x_left/y_top — левый и верхний край СЕТКИ в мире. Верхний край был здесь с самого начала, а
    левый подразумевался нулём — то есть считалось, что сетка всегда начинается от края карты.
    Для полной растеризации это верно, для ОКНА нет: кусок начинается с произвольного x, и
    переправа в нём съезжала на x0/клетку столбцов, а чаще просто выпадала за границы и
    пропадала. В стиле «клетки боя» это значило, что мост есть в бою, но не показан на всех
    кусках, кроме самого левого."""
    out = None
    for sh in doc["shapes"]:
        if sh["kind"] != "crossing" or bool(sh.get("ford")) != want_ford:
            continue
        (cx, cy), length, width, angle = crossing_geom(doc, sh)
        if out is None:
            out = np.zeros((gy, gx), dtype=bool)
        a = math.radians(angle)
        ux, uy = math.cos(a), math.sin(a)             # вдоль моста, поперёк воды
        step = cell_m * 0.4
        n = max(2, int(length / step) + 1)
        m = max(1, int(width / step) + 1)             # широкая дорога занимает не одну полосу
        for i in range(n + 1):
            t = -length / 2.0 + length * i / n
            for k in range(m + 1):
                q = -width / 2.0 + width * k / m
                x = cx + ux * t - uy * q
                y = cy + uy * t + ux * q
                col = int((x - x_left) / cell_m)
                row = int((y_top - y) / cell_m)
                if 0 <= col < gx and 0 <= row < gy:
                    out[row, col] = True
    return out

def _shape_bounds(sh):
    """Габарит фигуры в метрах с запасом на толщину. Нужен, чтобы оконная растеризация не гоняла
    все фигуры карты ради куска в полкилометра: на театре 10x10 км их сотни."""
    if sh["kind"] == "polygon":
        xs = [p[0] for p in sh["points"]]
        ys = [p[1] for p in sh["points"]]
        return min(xs), min(ys), max(xs), max(ys)
    if sh["kind"] == "line":
        r = float(sh.get("width_m", 8.0)) / 2 + 1.0
        xs = [p[0] for p in sh["points"]]
        ys = [p[1] for p in sh["points"]]
        return min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r
    if sh["kind"] == "building":
        cx, cy, w, h, _ = sh["rect_m"]
        r = math.hypot(w, h) / 2 + 1.0
        return cx - r, cy - r, cx + r, cy + r
    if sh["kind"] == "crossing":
        r = float(sh.get("length_m") or 0.0) / 2 + float(sh.get("width_m", 30.0)) + 60.0
        x, y = sh["point"]
        return x - r, y - r, x + r, y + r
    return -1e9, -1e9, 1e9, 1e9


def surface_window(doc, cell_m, x0_m, y0_m, w_m, h_m):
    """Сетка ТИПОВ для куска карты в мелкой клетке — то, что показывается вблизи камеры.

    Высоту окно не считает нарочно. Сглаживание рельефа работает на всю карту сразу (радиус —
    сотни метров), и посчитанное по куску не совпало бы с посчитанным по целой: на стыке кусков
    земля бы ломалась. Высота берётся из общего поля с интерполяцией — она и так плавная. А
    вблизи важна не она, а точность МЕСТНОСТИ: кромка леса, изгиб дороги, берег реки.

    Переправы участвуют: без них мост в окне исчез бы, и река резала бы карту надвое."""
    tid = _types()
    gx, gy = max(1, int(round(w_m / cell_m))), max(1, int(round(h_m / cell_m)))
    px_m = cell_m / SUB
    top = y0_m + gy * cell_m
    x1, y1 = x0_m + gx * cell_m, top

    def to_px(p):
        return ((p[0] - x0_m) / px_m, (top - p[1]) / px_m)

    near = {"shapes": [sh for sh in doc["shapes"]
                       if not (lambda b: b[2] < x0_m or b[0] > x1 or b[3] < y0_m or b[1] > y1)(
                           _shape_bounds(sh))]}
    masks, _ = _fill_masks(near, gx, gy, cell_m, to_px)
    _apply_crossings(near, masks, gx, gy, to_px)

    surf_img = np.full((gy, gx), tid["open"], dtype=np.int8)
    for name in PRIORITY:
        frac = masks[name].reshape(gy, SUB, gx, SUB).mean(axis=(1, 3))
        surf_img[frac > THRESHOLDS[name]] = tid[name]
    cross = _crossing_cells(near, gx, gy, cell_m, top, x_left=x0_m)
    if cross is not None:
        surf_img[cross] = tid["road"]        # то же, что и в полной растеризации: мост главнее
    fords = _crossing_cells(near, gx, gy, cell_m, top, want_ford=True, x_left=x0_m)
    if fords is not None:
        surf_img[fords] = tid["ford"]
    return np.ascontiguousarray(surf_img[::-1].T)


# ---------------------------------------------------------------- карта высот
#
# Высота — РАСТРОВЫЙ слой карты, а не фигуры. Причина простая: лес и дорога это области с краем,
# и край обязан быть точным на любом приближении — потому вектор. Высота же определена всюду и
# гладкая, её «край» (обрыв) — редкий случай, и он выражается крутым уклоном. Фигурами высоту
# рисовать можно (и удобно), но они ШТАМПЫ: вдавливаются в карту высот и перестают существовать
# отдельно. Иначе получается два источника правды, а рельеф из складывающихся фигур не умеет
# ни абсолютных отметок, ни обрыва, ни правки задним числом.
#
# Файл: <имя>.height.npz рядом с вектором, метры, [gx, gy], y вверх.

HEIGHT_CELLS = (5.0, 10.0, 20.0, 40.0)   # кратны друг другу вдвое: на этом стоит сшивка уровней
#                                          подробности в объёмном виде


def default_height_cell(size_m):
    """Клетка карты высот под размер карты: около пятисот точек на сторону, но из ряда кратных."""
    want = max(float(size_m[0]), float(size_m[1])) / 512.0
    return min(HEIGHT_CELLS, key=lambda c: abs(math.log(c / max(want, 1e-6))))


def height_path(vector_path):
    return vector_path[:-len(".vector.json")] + ".height.npz"


def _built_cell(vector_path):
    """Клетка, в которой карта уже собиралась для боя, если сборка рядом лежит."""
    meta = vector_path[:-len(".vector.json")] + ".map.json"
    if not os.path.exists(meta):
        return None
    try:
        with open(meta, "r", encoding="utf-8") as f:
            return float(json.load(f)["cell_m"])
    except Exception:
        return None


def _resample(h, src_cell, gx, gy, dst_cell):
    """Карта высот в другую клетку, билинейно. Отсчёт по центрам клеток: значение относится к
    клетке целиком (так его читает бой), а не к узлу."""
    sx, sy = h.shape
    u = ((np.arange(gx) + 0.5) * dst_cell) / src_cell - 0.5
    v = ((np.arange(gy) + 0.5) * dst_cell) / src_cell - 0.5
    u = np.clip(u, 0, sx - 1)
    v = np.clip(v, 0, sy - 1)
    i0 = np.floor(u).astype(np.int32); i1 = np.minimum(i0 + 1, sx - 1); fu = (u - i0)[:, None]
    j0 = np.floor(v).astype(np.int32); j1 = np.minimum(j0 + 1, sy - 1); fv = (v - j0)[None, :]
    a = h[np.ix_(i0, j0)] * (1 - fu) + h[np.ix_(i1, j0)] * fu
    b = h[np.ix_(i0, j1)] * (1 - fu) + h[np.ix_(i1, j1)] * fu
    return np.ascontiguousarray(a * (1 - fv) + b * fv).astype(np.float32)


def bake_relief(doc, cell_m):
    """Фигуры рельефа -> поле высот в метрах, [gx, gy], y вверх.

    Это ПЕРЕНОС старого формата, где рельеф жил фигурами: код тот же, чтобы уже нарисованные
    карты не поехали. Новым картам эта дорога не нужна — они правятся кистями по растру."""
    W, H = float(doc["size_m"][0]), float(doc["size_m"][1])
    gx, gy = int(round(W / cell_m)), int(round(H / cell_m))
    px_m = cell_m / SUB

    def to_px(p):
        return (p[0] / px_m, (H - p[1]) / px_m)

    img = np.zeros((gy * SUB, gx * SUB), dtype=np.float32)
    for sh in doc["shapes"]:
        if sh.get("type") != "relief":
            continue
        m = _blank(gx, gy)
        if sh["kind"] == "polygon":
            _fill_polygon(m, [to_px(p) for p in sh["points"]])
        elif sh["kind"] == "line":
            _fill_thick_line(m, [to_px(p) for p in sh["points"]],
                             float(sh.get("width_m", 200.0)) / px_m)
        else:
            continue
        img += m.astype(np.float32) * float(sh.get("h_m", 20.0))
    cells = img.reshape(gy, SUB, gx, SUB).mean(axis=(1, 3)).astype(np.float32)
    if not cells.any():
        return np.zeros((gx, gy), dtype=np.float32)
    # Склон должен быть длиной с сам холм. Коротким размытием этого не добиться: выходит плита с
    # отвесными бортами. Идём пирамидой — ужимаем поле в восемь раз, там сглаживаем и растягиваем
    # обратно бикубикой. Радиус сглаживания становится сотнями метров при той же цене.
    from PIL import Image as _I
    small = _I.fromarray(cells, mode="F").resize((max(3, gx // 8), max(3, gy // 8)), _I.BILINEAR)
    wide = np.asarray(small.resize((gx, gy), _I.BICUBIC), dtype=np.float32)
    near = cells.copy()
    for _ in range(RELIEF_SMOOTH):
        near = _box_blur(near)
    return np.ascontiguousarray((0.75 * wide + 0.25 * near)[::-1].T).astype(np.float32)


def sample_height(hm, X, Y):
    """Высота карты в произвольных точках мира, билинейно. X, Y — массивы метров."""
    h = np.asarray(hm["h"], dtype=np.float32)
    c = float(hm["cell_m"])
    sx, sy = h.shape
    u = np.clip(np.asarray(X, dtype=np.float32) / c - 0.5, 0, sx - 1)
    v = np.clip(np.asarray(Y, dtype=np.float32) / c - 0.5, 0, sy - 1)
    i0 = np.floor(u).astype(np.int32); i1 = np.minimum(i0 + 1, sx - 1); fu = u - i0
    j0 = np.floor(v).astype(np.int32); j1 = np.minimum(j0 + 1, sy - 1); fv = v - j0
    a = h[i0, j0] * (1 - fu) + h[i1, j0] * fu
    b = h[i0, j1] * (1 - fu) + h[i1, j1] * fu
    return (a * (1 - fv) + b * fv).astype(np.float32)


def absorb_relief(doc, cell_m=None):
    """Вдавить фигуры рельефа в карту высот и убрать их из вектора.

    Это и есть «штамп» из гибридной схемы: рисовать фигурами удобно, но жить они не должны —
    два источника правды на одну высоту означают, что правка одного молча расходится с другим."""
    if not any(sh.get("type") == "relief" for sh in doc["shapes"]):
        return doc
    cell = float(cell_m or default_height_cell(doc["size_m"]))
    add = bake_relief(doc, cell)
    hm = doc.get("height")
    if hm is None or float(hm["cell_m"]) != cell:
        base = (np.zeros_like(add) if hm is None
                else _resample(np.asarray(hm["h"], dtype=np.float32), float(hm["cell_m"]),
                               add.shape[0], add.shape[1], cell))
        doc["height"] = {"cell_m": cell, "h": (base + add).astype(np.float32)}
    else:
        doc["height"] = {"cell_m": cell, "h": (np.asarray(hm["h"], np.float32) + add)}
    doc["shapes"] = [sh for sh in doc["shapes"] if sh.get("type") != "relief"]
    return doc


def _wide_blur(a, r):
    """Размытие радиусом r клеток тремя проходами скользящего среднего.

    Своё, а не из PIL: тамошнее гауссово не берёт вещественные картинки (mode F), а гонять
    высоту через восемь бит значило бы округлять метры до четверти."""
    r = int(max(1, round(r * 0.55)))
    k = 2 * r + 1
    for _ in range(3):
        pad = np.pad(a, ((r + 1, r), (0, 0)), mode="edge")
        c = np.cumsum(pad, axis=0)
        a = (c[k:] - c[:-k]) / k
        pad = np.pad(a, ((0, 0), (r + 1, r)), mode="edge")
        c = np.cumsum(pad, axis=1)
        a = (c[:, k:] - c[:, :-k]) / k
    return a.astype(np.float32)


def stamp(doc, shapes, cell_m=None, slope_m=150.0, absolute=False):
    """Вдавить фигуры в карту высот: полигон — холм или котловина, линия — гряда или лощина.

    Склон СВОЙ у каждого штампа и задаётся в метрах, а не выводится из размера карты. Старый
    рельеф сглаживался пирамидой с радиусом в восьмую карты, из-за чего одна и та же нарисованная
    гряда давала разный холм на карте 2.5 км и на 10 км, а обрыв нарисовать было нечем вовсе.

    absolute — не прибавить, а ВЫРОВНЯТЬ до отметки: так делаются плато, дно карьера, терраса."""
    hm = doc.get("height")
    cell = float(cell_m or (hm and hm["cell_m"]) or default_height_cell(doc["size_m"]))
    W, H = float(doc["size_m"][0]), float(doc["size_m"][1])
    gx, gy = max(1, int(round(W / cell))), max(1, int(round(H / cell)))
    if hm is None:
        cur = np.zeros((gx, gy), dtype=np.float32)
    else:
        cur = _resample(np.asarray(hm["h"], np.float32), float(hm["cell_m"]), gx, gy, cell)
    px_m = cell / SUB

    def to_px(p):
        return (p[0] / px_m, (H - p[1]) / px_m)

    for sh in shapes:
        m = _blank(gx, gy)
        if sh["kind"] == "polygon":
            _fill_polygon(m, [to_px(p) for p in sh["points"]])
        elif sh["kind"] == "line":
            _fill_thick_line(m, [to_px(p) for p in sh["points"]],
                             float(sh.get("width_m", 200.0)) / px_m)
        else:
            continue
        cells = m.reshape(gy, SUB, gx, SUB).mean(axis=(1, 3)).astype(np.float32)
        blur = _wide_blur(cells, max(0.5, float(slope_m) / cell))
        # Растяжка после размытия возвращает штампу плоскую вершину: без неё узкая гряда шириной
        # с радиус склона проседает вдвое и высота в панели перестаёт значить хоть что-то.
        w = np.clip((blur - 0.25) / 0.5, 0.0, 1.0)
        w = np.ascontiguousarray(w[::-1].T)
        h_m = float(sh.get("h_m", 20.0))
        cur = cur * (1 - w) + h_m * w if absolute else cur + h_m * w
    doc["height"] = {"cell_m": cell, "h": cur.astype(np.float32)}
    return doc


def height_field(doc, gx, gy, cell_m):
    """Поле высот карты под запрошенную сетку, метры, [gx, gy], y вверх."""
    hm = doc.get("height")
    if hm is None:
        return bake_relief(doc, cell_m)
    return _resample(np.asarray(hm["h"], dtype=np.float32), float(hm["cell_m"]), gx, gy, cell_m)


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

    masks, buildings = _fill_masks(doc, gx, gy, cell_m, to_px)

    # РЕЛЬЕФ берётся из карты высот — отдельного растрового слоя карты (см. height_field).
    # Отдельным полем, а не типом клетки: высота — величина, а не материал, и на одной и той же
    # высоте бывает и лес, и поле.
    height_cell = height_field(doc, gx, gy, cell_m)

    _apply_crossings(doc, masks, gx, gy, to_px)

    # доля покрытия клетки: подпиксели SUB×SUB сворачиваются усреднением
    def frac(mask):
        return mask.reshape(gy, SUB, gx, SUB).mean(axis=(1, 3))

    surf_img = np.full((gy, gx), tid["open"], dtype=np.int8)
    for name in PRIORITY:
        surf_img[frac(masks[name]) > THRESHOLDS[name]] = tid[name]
    cross = _crossing_cells(doc, gx, gy, cell_m, H)
    if cross is not None:
        surf_img[cross] = tid["road"]                     # мост главнее всех, даже застройки
    fords = _crossing_cells(doc, gx, gy, cell_m, H, want_ford=True)
    if fords is not None:
        surf_img[fords] = tid["ford"]                     # брод так же главнее спора, но он не дорога
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
    if height_cell is not None and height_cell.any():
        fields["height"] = np.ascontiguousarray(height_cell).astype(np.float32)

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
    out = new_doc((w, h), shapes)

    # Взорванные мосты крой уносит вместе с фигурами. Иначе вырезка из театра молча ставила бы
    # снесённый мост обратно: точки сноса остались бы в источнике, а боевая карта собралась бы
    # с чистого листа — и замысел «переправа взорвана» пропадал бы ровно там, где по нему играют.
    blown = []
    for q in doc.get("blown", ()):
        nx, ny = T(q)
        if 0 <= nx <= w and 0 <= ny <= h:
            blown.append([round(nx, 1), round(ny, 1)])
    if blown:
        out["blown"] = blown

    # Высоту крой тоже уносит: без неё вырезанная боевая карта оказывалась плоской, хотя её
    # рисовали в горах. Растр пересэмплируется с поворотом — тем же преобразованием, что и фигуры.
    hm = doc.get("height")
    if hm is not None and np.any(hm["h"]):
        c = float(hm["cell_m"])
        gx, gy = max(1, int(round(w / c))), max(1, int(round(h / c)))
        ux = (np.arange(gx) + 0.5) * c - w / 2
        vy = (np.arange(gy) + 0.5) * c - h / 2
        U, V = np.meshgrid(ux, vy, indexing="ij")
        X = cx + U * ca + V * sa                    # обратный поворот: матрица ортогональна
        Y = cy - U * sa + V * ca
        out["height"] = {"cell_m": c, "h": sample_height(hm, X, Y)}
    return out


def line_hits(a_pts, b_pts):
    """Точки пересечения двух ломаных."""
    out = []
    for a0, a1 in zip(a_pts[:-1], a_pts[1:]):
        for b0, b1 in zip(b_pts[:-1], b_pts[1:]):
            hit = _seg_intersect(a0, a1, b0, b1)
            if hit:
                out.append(hit[0])
    return out


def crossing_gaps(doc, tol_m=60.0, only=None):
    """Пересечения дороги с водой, где переправы НЕТ.

    Дорога через реку без переправы не работает: у клетки один тип, вода перекрывает дорогу, и
    берег остаётся берегом. На вид при этом всё нарисовано — дорога входит в реку и выходит с
    той стороны, — поэтому промах и не замечается, пока карта не окажется разрезанной пополам.

    only — список фигур, которыми интересуемся (только что нарисованная линия): тогда ищем
    пересечения лишь с ней, а не по всей карте.

    ВЗОРВАННЫЙ МОСТ не считается пробелом. Удалить переправу — это осмысленное действие («мост
    взорван»), и точка сноса помнится в doc["blown"]. Без этого пересчёт после правки воскрешал
    бы снесённый мост, стоило подвинуть рядом дорогу, — то есть отменял бы решение молча.

    Видит только ЛИНИИ: озеро полигоном через дорогу здесь не найдётся. Сейчас воды-полигонов
    в картах нет ни одной, поэтому и не усложняем."""
    roads = [sh for sh in doc["shapes"] if sh["kind"] == "line" and sh.get("type") == "road"]
    waters = [sh for sh in doc["shapes"] if sh["kind"] == "line" and sh.get("type") == "water"]
    have = [sh["point"] for sh in doc["shapes"] if sh["kind"] == "crossing"]
    have += [list(p) for p in doc.get("blown", ())]
    out = []
    for r in roads:
        for w in waters:
            if only is not None and r not in only and w not in only:
                continue
            for p in line_hits(r["points"], w["points"]):
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) < tol_m for q in have + out):
                    continue
                out.append([round(float(p[0]), 1), round(float(p[1]), 1)])
    return out


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
    fields = {k: z[k] for k in terrain.FIELDS if k in z}
    # высота лежит в МЕТРАХ (как всё в векторе), а бой считает в игровых единицах — иначе
    # холм в 30 м оказался бы выше карты в пятнадцать раз
    fields["height"] = np.asarray(fields.get("height", 0.0), dtype=np.float32) / m_per_unit
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
