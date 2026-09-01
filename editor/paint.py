"""Картинка местности ВЕКТОРОМ: фигуры рисуются как есть, а не клетками.

Разница видна на первом же приближении. Растровая картинка тайла показывает тип клетки, и дорога
в восемь метров при клетке в пять идёт лесенкой; вектор рисует ту же дорогу линией нужной ширины,
и край у неё остаётся краем на любом увеличении. Стоит это столько же: тайл 640 м в 1.25 м на
пиксель — 12.5 мс против 11.4 мс у растра в 5 м.

Оговорка, ради которой всё и написано отдельно. Эта картинка — ПОКАЗ, а не то, что читает бой.
Бою достаётся сетка полей со своей клеткой, и край леса там всё равно ступенька. Поэтому галка
«сетка полей» переключает объёмный вид на растровую картинку: один щелчок — и видно ровно то,
что получит бой, со всеми ступеньками. Врать про местность редактор не должен.

Порядок рисования тот же, что порядок приоритетов в vectormap: дорога, лес, поле, вода,
застройка — последний главнее. Поэтому лес перекрывает дорогу и здесь, и в сетке боя, и просека
в лесу выглядит так же, как считается.
"""
import threading

import numpy as np
from PIL import Image, ImageDraw

import vectormap
from view3d import COLORS

SS = 2                       # во сколько раз рисуем крупнее ради сглаживания края
TYPE_ID = {"open": 0, "forest": 1, "building": 2, "water": 3, "road": 4, "ford": 5}

# Топографический стиль — по советской карте 1:25000. Он не украшение: на такой карте рельеф
# читается горизонталями, а не тенью, и по ним сразу видно седловину, обратный скат и крутизну —
# то, ради чего рельеф в игре и заведён. Отмывка этого не даёт: она показывает, где склон, но не
# отвечает, на сколько метров.
TOPO = {"open": (255, 255, 255),        # белое поле — как на бумаге
        "forest": (196, 230, 186),      # лес светло-зелёный
        "water": (168, 214, 235),       # вода голубая
        "building": (40, 40, 40),       # строения чёрные
        "road": (226, 132, 52),         # шоссе оранжевое
        "ford": (214, 196, 150)}       # брод — песчаная перемычка поперёк воды
TOPO_CONTOUR = (176, 118, 74)           # горизонтали коричневые
TOPO_INDEX = (150, 92, 50)              # утолщённая — каждая пятая
TOPO_WATER_EDGE = (60, 130, 175)
TOPO_EDGE = (58, 92, 52)                # контур растительности и знаки пород — тёмно-зелёные

WATER_ALPHA = 0.72                      # непрозрачность воды в ЖИВОМ стиле.
#                            Сплошная заливка читалась как синяя лента поверх карты, а не как
#                            река: дно не видно, и берег ничем не отличается от края фигуры.
#                            Прозрачность даёт и полезное, а не только красивое — под водой
#                            остаётся видна ДОРОГА, потому что она рисуется раньше воды. Там,
#                            где дорога уходит в реку и не выходит мостом, это сразу заметно.
#                            В топостиле вода СПЛОШНАЯ: голубая заливка там условный знак, и
#                            размывать его значит портить чтение карты.


def _color(name, style="live"):
    return TOPO[name] if style == "topo" else COLORS[TYPE_ID[name]]


# Крап местности. Лес и поле одним цветом читаются как заливка в редакторе, а не как местность:
# глазу не за что зацепиться, и масштаб пропадает — это та же беда, что с гладкой землёй в срезе.
# Шум берётся от МИРОВЫХ координат, а не от пикселя картинки: иначе при смене уровня подробности
# крап пересыпался бы, и лес мерцал бы на каждом движении камеры.
_NOISE = np.random.default_rng(4242).random((256, 256)).astype(np.float32)


def _noise_field(x0, y0, w_m, h_m, px_w, px_h, scale_m, coarse=4):
    """Шум на всю картинку. Считается на СЕТКЕ ВЧЕТВЕРО РЕЖЕ и растягивается: пятно крапа
    занимает десятки пикселей, и считать его в каждом — это 26 мс на выборку из таблицы против
    полутора. На глаз разницы нет, а картинка куска складывается из четырёх таких слоёв."""
    nw = max(8, px_w // coarse)
    nh = max(8, px_h // coarse)
    gx = (np.arange(nw, dtype=np.float32) + 0.5) * (w_m / nw) + x0
    gy = (np.arange(nh, dtype=np.float32) + 0.5) * (h_m / nh) + y0
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    # ascontiguousarray с явным float32 обязателен: PIL читает буфер как есть, и массив
    # двойной точности он принимает за одинарную — картинка выходит из бесконечностей и NaN
    small = np.ascontiguousarray(_vnoise(X, Y, scale_m), dtype=np.float32)
    if (nw, nh) == (px_w, px_h):
        return small
    return np.asarray(Image.fromarray(small).resize((px_w, px_h), Image.BILINEAR),
                      dtype=np.float32)


def _vnoise(X, Y, scale_m):
    """Плавный шум в точке мира: билинейная выборка из таблицы со сглаживанием."""
    u, v = X / scale_m, Y / scale_m
    i0 = np.floor(u).astype(np.int32)
    j0 = np.floor(v).astype(np.int32)
    fu, fv = u - i0, v - j0
    fu = fu * fu * (3.0 - 2.0 * fu)               # сглаживание, иначе шум идёт гранями
    fv = fv * fv * (3.0 - 2.0 * fv)
    i0 %= 256
    j0 %= 256
    i1, j1 = (i0 + 1) % 256, (j0 + 1) % 256
    a = _NOISE[i0, j0] * (1 - fu) + _NOISE[i1, j0] * fu
    b = _NOISE[i0, j1] * (1 - fu) + _NOISE[i1, j1] * fu
    return a * (1 - fv) + b * fv


def _type_mask(shapes, x0, y0, w_m, h_m, px_w, px_h):
    """Тип в каждой точке картинки — чтобы знать, где класть крап леса, а где поля.

    Рисуется БЕЗ подвыборки, в один проход: цвет берётся из сглаженной картинки, а маске
    сглаживание не нужно — она только говорит, какой крап накладывать."""
    img = Image.new("L", (max(1, px_w), max(1, px_h)), TYPE_ID["open"] + 1)
    d = ImageDraw.Draw(img)
    sx, sy = px_w / w_m, px_h / h_m

    def P(p):
        return ((p[0] - x0) * sx, (y0 + h_m - p[1]) * sy)

    for name in vectormap.PRIORITY:
        v = TYPE_ID[name] + 1
        for sh in shapes:
            if sh.get("type") != name or sh["kind"] not in ("polygon", "line"):
                continue
            if sh["kind"] == "polygon" and len(sh["points"]) >= 3:
                d.polygon([P(p) for p in sh["points"]], fill=v)
            elif sh["kind"] == "line":
                w = max(1, int(round(float(sh.get("width_m", 8.0)) * sx)))
                dash = sh.get("dash_m")
                runs = (vectormap.dash_polyline(sh["points"], dash[0], dash[1]) if dash
                        else [sh["points"]])
                for run in runs:
                    if len(run) >= 2:
                        d.line([P(p) for p in run], fill=v, width=w, joint="curve")
    for sh in shapes:
        if sh["kind"] == "building":
            cx, cy, bw, bh, ang = sh["rect_m"]
            d.polygon([P(p) for p in vectormap._rect_points(cx, cy, bw, bh, ang)],
                      fill=TYPE_ID["building"] + 1)
    return np.asarray(img)[::-1]                  # строка 0 -> y0


def forest_marks(d, shapes, x0, y0, w_m, h_m, px_w, px_h, mask):
    """Точечный контур леса и знаки пород — как на топокарте.

    По условным знакам контур растительности рисуется ТОЧЕЧНЫМ пунктиром (а не сплошной линией),
    хвойный лес обозначается ёлочкой, лиственный — кружком на ножке, смешанный — обоими рядом.
    Порода у нас в векторе не хранится, поэтому берётся устойчивым жребием от места: одно и то же
    место всегда даёт тот же знак, и при смене подробности лес не пересаживается."""
    sx = px_w / w_m
    if sx <= 0:
        return

    def P(p):
        return ((p[0] - x0) * sx, (y0 + h_m - p[1]) * (px_h / h_m))

    # 1. контур точками
    dot = max(1.0, 1.2 * sx * 2.0)
    for sh in shapes:
        if sh.get("type") != "forest":
            continue
        pts = None
        if sh["kind"] == "polygon" and len(sh["points"]) >= 3:
            pts = list(sh["points"]) + [sh["points"][0]]
        elif sh["kind"] == "line" and len(sh["points"]) >= 2:
            pts = list(sh["points"])
        if not pts:
            continue
        for run in vectormap.dash_polyline(pts, 3.0, 9.0):
            q = [P(p) for p in run]
            if len(q) >= 2:
                d.line(q, fill=TOPO_EDGE, width=max(1, int(dot)))

    # 2. знаки пород по жребию от места
    step_m = 70.0
    r = int(max(0, np.ceil(w_m / step_m)))
    size_px = 13.0 * sx
    if size_px < 4.0 or r == 0:                      # мельче — знак в кашу, на карте его не ставят
        return
    for gi in range(int(np.floor(x0 / step_m)), int(np.floor((x0 + w_m) / step_m)) + 1):
        for gj in range(int(np.floor(y0 / step_m)), int(np.floor((y0 + h_m) / step_m)) + 1):
            n1 = _NOISE[gi % 256, gj % 256]
            n2 = _NOISE[(gi * 7 + 13) % 256, (gj * 5 + 29) % 256]
            wx = (gi + 0.25 + 0.5 * n1) * step_m
            wy = (gj + 0.25 + 0.5 * n2) * step_m
            ix = int((wx - x0) * sx)
            iy = int((wy - y0) * (px_h / h_m))
            if not (0 <= ix < px_w and 0 <= iy < px_h) or mask[iy, ix] != TYPE_ID["forest"] + 1:
                continue
            px_, py_ = P((wx, wy))
            hgt = size_px
            if n1 > 0.45:                            # хвойный: ёлочка
                d.line([(px_, py_), (px_, py_ - hgt)], fill=TOPO_EDGE, width=1)
                for k, t in ((0.35, 0.45), (0.6, 0.3)):
                    yy = py_ - hgt * (1 - k)
                    d.line([(px_ - hgt * t * 0.5, yy + hgt * 0.12), (px_, yy)],
                           fill=TOPO_EDGE, width=1)
                    d.line([(px_ + hgt * t * 0.5, yy + hgt * 0.12), (px_, yy)],
                           fill=TOPO_EDGE, width=1)
            else:                                    # лиственный: кружок на ножке
                rr = hgt * 0.30
                d.line([(px_, py_), (px_, py_ - hgt * 0.55)], fill=TOPO_EDGE, width=1)
                d.ellipse([px_ - rr, py_ - hgt * 0.55 - rr, px_ + rr, py_ - hgt * 0.55 + rr],
                          outline=TOPO_EDGE, width=1)


def texture(rgb, mask, rect, style):
    """Крап поверх заливки: лес пятнами кроны, поле — полосами угодий.

    В топостиле поле остаётся белым (так и на бумаге), а лес получает редкие тёмные точки —
    те самые кружки, которыми лес обозначают на карте."""
    forest = mask == TYPE_ID["forest"] + 1
    open_ = mask == TYPE_ID["open"] + 1
    if not forest.any() and not open_.any():
        return rgb
    px_h, px_w = mask.shape

    def noise(scale, coarse=4):
        return _noise_field(rect[0], rect[1], rect[2], rect[3], px_w, px_h, scale, coarse)

    if style == "topo":
        if forest.any():
            dots = 0.6 * noise(9.0, 2) + 0.4 * noise(3.0, 2)
            rgb[forest & (dots > 0.74)] = (132, 178, 120)     # кружки леса, как на бумаге
        return rgb
    # Один множитель на всю картинку вместо выборки по маске: выборка с последующим умножением
    # копирует полкартинки туда и обратно и стоит вчетверо дороже самого шума.
    mul = np.ones(mask.shape, dtype=np.float32)
    if forest.any():
        n = 0.55 * noise(34.0) + 0.3 * noise(11.0) + 0.15 * noise(4.0, 2)
        np.copyto(mul, 0.70 + 0.58 * n, where=forest)          # кроны пятнами
    if open_.any():
        n = 0.65 * noise(90.0) + 0.35 * noise(26.0)
        np.copyto(mul, 0.86 + 0.28 * n, where=open_)           # угодья широкими полосами
    return np.clip(rgb * mul[..., None], 0, 255).astype(np.uint8)


def contour_step(h_range):
    """Сечение рельефа: чем ровнее местность, тем чаще горизонтали. Пять метров — то же, что на
    карте 1:25000, на которую этот стиль и похож."""
    for step in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        if h_range / step <= 40:
            return step
    return 100.0


def _blur_m(h, cell_m, radius_m):
    """Скользящее среднее радиусом В МЕТРАХ. Радиус в метрах, а не в клетках, потому что поле
    высот у карт разной клетки, а линия должна выходить одна и та же."""
    r = max(1, int(round(radius_m / max(cell_m, 1e-6) / 2)))
    k = 2 * r + 1
    for _ in range(2):
        pad = np.pad(h, ((r + 1, r), (0, 0)), mode="edge")
        c = np.cumsum(pad, axis=0)
        h = (c[k:] - c[:-k]) / k
        pad = np.pad(h, ((0, 0), (r + 1, r)), mode="edge")
        c = np.cumsum(pad, axis=1)
        h = (c[:, k:] - c[:, :-k]) / k
    return h.astype(np.float32)


# Кому какие рёбра клетки соединять — таблица «шагающих квадратов». Углы считаются против
# часовой: 1 — юго-запад, 2 — юго-восток, 4 — северо-восток, 8 — северо-запад; рёбра в том же
# порядке: 0 — низ, 1 — право, 2 — верх, 3 — лево.
_MS = {1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 6: ((0, 2),), 7: ((3, 2),),
       8: ((2, 3),), 9: ((2, 0),), 11: ((2, 1),), 12: ((1, 3),), 13: ((1, 0),), 14: ((0, 3),)}
_SADDLE = {5: (((0, 1), (2, 3)), ((3, 0), (1, 2))),      # (центр выше, центр ниже)
           10: (((3, 0), (1, 2)), ((0, 1), (2, 3)))}


def _trace_level(h, v, cell_m, chunk=64):
    """Одна горизонталь линиями: список массивов (N,2) в метрах мира.

    Точка горизонтали лежит на ребре между двумя узлами сетки высот и считается линейно. Каждое
    ребро получает НОМЕР, и звенья сшиваются по номерам, а не по координатам: у общего ребра
    двух клеток номер один и тот же, поэтому шов сходится точно, без сравнения чисел с допуском.
    """
    nx, ny = h.shape
    if nx < 2 or ny < 2:
        return []
    ny1 = ny - 1
    nH = (nx - 1) * ny

    dh = h[1:, :] - h[:-1, :]
    th = (v - h[:-1, :]) / np.where(np.abs(dh) < 1e-12, 1.0, dh)
    xh = (np.arange(nx - 1, dtype=np.float32)[:, None] + 0.5 + th) * cell_m
    yh = np.broadcast_to((np.arange(ny, dtype=np.float32)[None, :] + 0.5) * cell_m, xh.shape)

    dv = h[:, 1:] - h[:, :-1]
    tv = (v - h[:, :-1]) / np.where(np.abs(dv) < 1e-12, 1.0, dv)
    yv = (np.arange(ny - 1, dtype=np.float32)[None, :] + 0.5 + tv) * cell_m
    xv = np.broadcast_to((np.arange(nx, dtype=np.float32)[:, None] + 0.5) * cell_m, yv.shape)

    XY = np.stack([np.concatenate([xh.ravel(), xv.ravel()]),
                   np.concatenate([yh.ravel(), yv.ravel()])], axis=1).astype(np.float32)

    b = h >= v
    if not b.any() or b.all():
        return []
    c0, c1 = b[:-1, :-1], b[1:, :-1]
    c2, c3 = b[1:, 1:], b[:-1, 1:]
    case = (c0.astype(np.uint8) | (c1.astype(np.uint8) << 1)
            | (c2.astype(np.uint8) << 2) | (c3.astype(np.uint8) << 3))

    def ends(e, I, J):
        if e == 0:
            return I * ny + J
        if e == 1:
            return nH + (I + 1) * ny1 + J
        if e == 2:
            return I * ny + (J + 1)
        return nH + I * ny1 + J

    # Клетки с пересечением ищем ОДИН раз, а не по разу на каждый из шестнадцати случаев:
    # полный проход по сетке стоил шестую долю секунды на карту, а пересечений в ней
    # тысячи из четверти миллиона клеток.
    I, J = np.nonzero((case != 0) & (case != 15))
    if not I.size:
        return []
    cv = case[I, J]
    A, B = [], []
    for c, pairs in _MS.items():
        sel = cv == c
        if sel.any():
            Ic, Jc = I[sel], J[sel]
            for e1, e2 in pairs:
                A.append(ends(e1, Ic, Jc))
                B.append(ends(e2, Ic, Jc))
    for c, (up, down) in _SADDLE.items():
        sel = cv == c
        if not sel.any():
            continue
        Ic, Jc = I[sel], J[sel]
        mid = 0.25 * (h[Ic, Jc] + h[Ic + 1, Jc] + h[Ic + 1, Jc + 1] + h[Ic, Jc + 1])
        for hi, pairs in ((mid >= v, up), (mid < v, down)):
            if not np.any(hi):
                continue
            for e1, e2 in pairs:
                A.append(ends(e1, Ic[hi], Jc[hi]))
                B.append(ends(e2, Ic[hi], Jc[hi]))
    if not A:
        return []
    A = np.concatenate(A).astype(np.int64)
    B = np.concatenate(B).astype(np.int64)

    # Сшивка. Ребро принадлежит самое большее двум клеткам, значит у каждого конца звена не
    # больше одного соседа — обход выходит однозначным, без разбора развилок.
    nseg = A.size
    ends_all = np.concatenate([A, B])
    segs_all = np.concatenate([np.arange(nseg), np.arange(nseg)])
    order = np.argsort(ends_all, kind="stable")
    ends_s, segs_s = ends_all[order], segs_all[order]
    same = np.nonzero(np.diff(ends_s) == 0)[0]          # пары, стоящие рядом после сортировки
    link = {}
    for p in same.tolist():
        s, t = int(segs_s[p]), int(segs_s[p + 1])
        link.setdefault(s, []).append(t)
        link.setdefault(t, []).append(s)

    used = np.zeros(nseg, dtype=bool)
    lines = []
    # сначала незамкнутые: их начала — звенья с одним соседом или вовсе без
    starts = [s for s in range(nseg) if len(link.get(s, ())) < 2]
    for s0 in starts + list(range(nseg)):
        if used[s0]:
            continue
        chain = [s0]
        used[s0] = True
        for _ in range(2):                               # в обе стороны от начального звена
            cur, tail = s0, []
            while True:
                nxt = [t for t in link.get(cur, ()) if not used[t]]
                if not nxt:
                    break
                cur = nxt[0]
                used[cur] = True
                tail.append(cur)
            chain = list(reversed(tail)) + chain if _ else chain + tail
        ids = _chain_points(chain, A, B)
        if ids.size >= 2:
            lines.append(XY[ids])
    out = []
    for ln in lines:
        out += _chop(_chaikin(ln), chunk)
    return out


def _chain_points(chain, A, B):
    """Цепочка звеньев -> последовательность номеров рёбер. Звено хранит два конца без порядка,
    поэтому направление восстанавливается по общему концу с предыдущим."""
    if not chain:
        return np.zeros(0, dtype=np.int64)
    if len(chain) == 1:
        return np.array([A[chain[0]], B[chain[0]]], dtype=np.int64)
    a0, b0 = A[chain[0]], B[chain[0]]
    a1, b1 = A[chain[1]], B[chain[1]]
    out = [b0, a0] if (a0 == a1 or a0 == b1) else [a0, b0]
    for s in chain[1:]:
        out.append(B[s] if A[s] == out[-1] else A[s])
    return np.array(out, dtype=np.int64)


def _chaikin(pts, passes=2):
    """Скругление углов по Чайкину. Точки горизонтали сидят на рёбрах сетки высот, и без этого
    линия идёт видимыми гранями по клетке: на карте в 20 м это ломаная с шагом в двадцать
    метров, на приближении заметная."""
    p = np.asarray(pts, dtype=np.float32)
    closed = bool(np.all(p[0] == p[-1])) and len(p) > 3
    for _ in range(passes):
        if len(p) < 3:
            break
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * (len(p) - 1), 2), dtype=np.float32)
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([mid, mid[:1]]) if closed else np.vstack([p[:1], mid, p[-1:]])
    return p


def _chop(pts, chunk):
    """Длинную линию режем на куски: иначе горизонталь через всю карту попадает в габарит
    любого тайла и рисуется целиком ради полусантиметра внутри квадрата."""
    if chunk <= 0 or len(pts) <= chunk:
        return [pts]
    return [pts[i:i + chunk + 1] for i in range(0, len(pts) - 1, chunk)]


class Contours:
    """Горизонтали ВСЕЙ карты линиями — один раз и на все уровни подробности.

    Зачем так. Раньше каждый кусок карты считал горизонтали сам, по своей растровой выборке
    высот: у ближнего куска точка 1.25 м, у дальнего 20, и прореживание бралось по крутизне
    ЭТОГО куска. Выходило, что соседние куски рисовали разные линии — и по положению, и по
    набору, — а при отъезде камеры они перерисовывались заново и не сходились с прежними.

    Теперь линия одна: она вычислена по общему полю высот в метрах мира и просто рисуется в
    любой кусок под любым увеличением. Прореживание — только выбор, какие из этих линий
    показать, и берётся кратным двум, поэтому дальний вид показывает ПОДМНОЖЕСТВО ближнего, а
    не другие линии."""

    def __init__(self, height, cell_m, step, smooth_m=45.0):
        h = np.asarray(height, dtype=np.float32)
        self.step = float(step)
        self.cell_m = float(cell_m)
        self.pts, ks = [], []
        self.slope = 0.0
        if h.size == 0 or not np.isfinite(h).all():
            self.k = np.zeros(0, dtype=np.int32)
            self.bbox = np.zeros((0, 4), dtype=np.float32)
            return
        h = _blur_m(h, self.cell_m, smooth_m)
        g = np.hypot(np.gradient(h, self.cell_m, axis=0), np.gradient(h, self.cell_m, axis=1))
        pos = g[g > 1e-6]
        self.slope = float(np.percentile(pos, 70)) if pos.size else 0.0
        lo, hi = float(h.min()), float(h.max())
        for k in range(int(np.floor(lo / self.step)), int(np.ceil(hi / self.step)) + 1):
            for ln in _trace_level(h, k * self.step, self.cell_m):
                if len(ln) >= 2:
                    self.pts.append(ln)
                    ks.append(k)
        self.k = np.array(ks, dtype=np.int32)
        self.bbox = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                              for p in self.pts], dtype=np.float32).reshape(-1, 4)

    def mult_for(self, m_per_px, min_px=16.0):
        """Во сколько раз проредить: показываем каждую вторую, четвёртую, восьмую линию, пока
        соседние не разойдутся на min_px точек. Кратность — степень двойки и зависит ТОЛЬКО от
        масштаба, а не от местности под куском, иначе у соседних кусков наборы линий разные."""
        mult = 1
        if self.slope <= 0.0:
            return mult
        while self.step * mult / (self.slope * max(m_per_px, 1e-6)) < min_px and mult < 1024:
            mult *= 2
        return mult

    def select(self, x0, y0, x1, y1, mult=1):
        """Линии, задевающие окно. Отбор по габаритам: на театре линий тысячи, а в квадрат
        640 м попадает десяток."""
        if not self.pts:
            return []
        bb = self.bbox
        m = ((bb[:, 0] <= x1) & (bb[:, 2] >= x0) & (bb[:, 1] <= y1) & (bb[:, 3] >= y0))
        if mult > 1:
            m &= (self.k % mult) == 0
        return [(int(self.k[i]), self.pts[i]) for i in np.nonzero(m)[0]]


_CSET = {}                    # (id массива высот, форма, клетка, сечение) -> Contours
_CSET_LOCK = threading.Lock()


def contour_set(height, cell_m, step, smooth_m=45.0):
    """Горизонтали карты с запоминанием. Ключ — САМ МАССИВ высот: штамп рельефа заменяет его
    целиком, поэтому смена массива и есть признак того, что линии устарели. Ссылку на массив
    держим в ключе, иначе id освободившейся памяти достанется другому массиву."""
    key = (id(height), height.shape, float(cell_m), float(step), float(smooth_m))
    with _CSET_LOCK:
        got = _CSET.get(key)
        if got is not None:
            return got[1]
    cs = Contours(height, cell_m, step, smooth_m)
    with _CSET_LOCK:
        if len(_CSET) > 3:
            _CSET.clear()
        _CSET[key] = (height, cs)
    return cs


def draw_contours(rgb, cset, x0, y0, w_m, h_m, px_w, px_h, mult=1, ss=2, index_every=5):
    """Горизонтали в готовую картинку куска. rgb: (py, px, 3), строка 0 — низ.

    Рисуем в маску вдвое крупнее и ужимаем: коричневая линия в один пиксель без сглаживания
    на белом поле рябит и рвётся на пологих местах. Смешиваем ТОЛЬКО по точкам самой линии,
    а не по всей картинке: линии занимают пару процентов квадрата, и полный проход по трём
    цветовым каналам стоил тридцать миллисекунд на тайл вместо трёх."""
    if not cset.pts:
        return rgb
    lines = cset.select(x0, y0, x0 + w_m, y0 + h_m, mult)
    if not lines:
        return rgb
    W, H = max(1, int(px_w * ss)), max(1, int(px_h * ss))
    sx, sy = W / w_m, H / h_m
    masks = [None, None]
    width = (max(1, ss), max(2, 2 * ss))
    for k, p in lines:
        i = 1 if (k % index_every == 0) else 0
        if masks[i] is None:
            m = Image.new("L", (W, H), 0)
            masks[i] = (m, ImageDraw.Draw(m))
        xy = np.empty((len(p), 2), dtype=np.float32)
        xy[:, 0] = (p[:, 0] - x0) * sx
        xy[:, 1] = (y0 + h_m - p[:, 1]) * sy            # y вниз для рисования
        masks[i][1].line(xy.ravel().tolist(), fill=255, width=width[i], joint="curve")
    alpha = [None, None]
    for i in (0, 1):
        if masks[i] is None:
            continue
        img = masks[i][0]
        if ss > 1:
            img = img.resize((px_w, px_h), Image.BOX)
        alpha[i] = np.asarray(img, dtype=np.float32)[::-1]     # строка 0 -> y0
    if alpha[0] is not None and alpha[1] is not None:
        alpha[0] = alpha[0] * (1.0 - alpha[1] / 255.0)         # утолщённая главнее тонкой
    for a, col in ((alpha[0], TOPO_CONTOUR), (alpha[1], TOPO_INDEX)):
        if a is None:
            continue
        ys, xs = np.nonzero(a > 0.5)
        if not ys.size:
            continue
        w = (a[ys, xs] / 255.0)[:, None]
        rgb[ys, xs] = (rgb[ys, xs] * (1.0 - w)
                       + np.array(col, dtype=np.float32) * w).astype(np.uint8)
    return rgb


def paint(shapes, x0, y0, w_m, h_m, px_w, px_h, ss=SS, style="live", lines=None,
          mult=1):
    """Кусок карты картинкой. Возвращает (py, px, 3) uint8, строка 0 — это y0 (низ куска).

    Строка 0 внизу нарочно: так картинка ложится в текстуру видеокарты без переворота, а
    переворачивать её в двух местах уже пробовали — это стоило зеркальной карты."""
    nw, nh = max(1, int(px_w * ss)), max(1, int(px_h * ss))
    sx, sy = nw / w_m, nh / h_m
    img = Image.new("RGB", (nw, nh), _color("open", style))
    d = ImageDraw.Draw(img)

    def P(p):
        return ((p[0] - x0) * sx, (y0 + h_m - p[1]) * sy)          # y вниз для рисования

    by_type = {}
    crossings = []
    buildings = []
    for sh in shapes:
        if sh["kind"] == "building":
            buildings.append(sh)
        elif sh["kind"] == "crossing":
            crossings.append(sh)
        elif sh.get("type") in TYPE_ID:
            by_type.setdefault(sh["type"], []).append(sh)

    def draw_shapes(dd, name, col):
        """Фигуры одного типа. Вынесено, потому что воду теперь рисуют дважды: сперва в маску,
        потом ею же прозрачно поверх картинки."""
        for sh in by_type.get(name, ()):
            if sh["kind"] == "polygon":
                if len(sh["points"]) >= 3:
                    dd.polygon([P(p) for p in sh["points"]], fill=col)
            elif sh["kind"] == "line":
                w = max(1, int(round(float(sh.get("width_m", 8.0)) * sx)))
                dash = sh.get("dash_m")
                runs = (vectormap.dash_polyline(sh["points"], dash[0], dash[1]) if dash
                        else [sh["points"]])
                for run in runs:
                    if len(run) >= 2:
                        pts = [P(p) for p in run]
                        dd.line(pts, fill=col, width=w, joint="curve")
                        if w > 2:                    # круглые торцы: без них штрих-пунктир рубленый
                            r = w / 2
                            for q in (pts[0], pts[-1]):
                                dd.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=col)

    see_through_water = style != "topo"
    for name in vectormap.PRIORITY:
        if name == "water" and see_through_water:
            continue                      # воду кладём ниже, прозрачной, поверх уже нарисованного
        draw_shapes(d, name, _color(name, style))

    if see_through_water and by_type.get("water"):
        # маска воды той же геометрией, что и заливка, — рисуем её белым в отдельный слой
        wm = Image.new("L", img.size, 0)
        draw_shapes(ImageDraw.Draw(wm), "water", 255)
        tint = Image.new("RGB", img.size, _color("water", style))
        img = Image.composite(Image.blend(img, tint, WATER_ALPHA), img, wm)
        d = ImageDraw.Draw(img)           # композит вернул НОВУЮ картинку — рисуем дальше в неё

    # переправа пробивает воду — как и в сетке боя. Мост кладёт поверх дорогу, брод остаётся
    # бродом: он проходим, но медленный, и красить его дорогой значило бы врать про местность.
    for sh in crossings:
        (px_, py_), length, width, angle = crossing_geom_safe(shapes, sh)
        d.polygon([P(q) for q in vectormap._rect_points(px_, py_, length, width, angle)],
                  fill=_color("ford" if sh.get("ford") else "road", style))

    for sh in buildings:
        cx, cy, w, h, ang = sh["rect_m"]
        d.polygon([P(p) for p in vectormap._rect_points(cx, cy, w, h, ang)],
                  fill=_color("building", style))

    out = img.resize((max(1, px_w), max(1, px_h)), Image.BOX)      # сглаживание края
    rgb = np.ascontiguousarray(np.asarray(out)[::-1])              # строка 0 -> y0

    mask = _type_mask(shapes, x0, y0, w_m, h_m, px_w, px_h)
    rgb = texture(rgb, mask, (x0, y0, w_m, h_m), style)
    if style == "topo":
        # знаки поверх крапа: рисуем в картинку окончательного размера, чтобы линия знака была
        # в один пиксель, а не размылась при уменьшении
        marks = Image.fromarray(rgb[::-1])
        forest_marks(ImageDraw.Draw(marks), shapes, x0, y0, w_m, h_m, px_w, px_h, mask)
        rgb = np.ascontiguousarray(np.asarray(marks)[::-1])

    if style == "topo" and lines is not None:
        # Горизонтали кладём ПОСЛЕ уменьшения картинки: они уже посчитаны по всей карте в метрах
        # мира, и рисовать их вместе с местностью в двойном размере значило бы размыть линию.
        rgb = draw_contours(rgb, lines, x0, y0, w_m, h_m, px_w, px_h, mult)
    return rgb


def crossing_geom_safe(shapes, sh):
    """Геометрия переправы. Функция vectormap ждёт документ целиком — здесь у нас только фигуры
    куска, и этого ей достаточно."""
    return vectormap.crossing_geom({"shapes": shapes}, sh)
