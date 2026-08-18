"""Редактор карт WarGame: рисование ВЕКТОРОМ, сборка полей, точки сценария.

Карта здесь — не клетки, а фигуры: контур леса, линия дороги с настоящей шириной, дом
прямоугольником, переправа точкой. Из них собираются поля свойств, которые читает бой, и граф
дорог для дальних маршрутов (vectormap.py). Клетки никуда не делись — просто стали
производным: их можно пересобрать под любой размер, и разрешение перестало быть решением,
зашитым в данные.

Что даёт вектор на практике:
  * дорога шириной 8 м остаётся дорогой шириной 8 м, а не полосой в клетку;
  * дома — объекты со своей вместимостью, а не связные пятна клеток;
  * карта режется под любым углом без лесенки (mapgen.py crops);
  * из тех же фигур потом вырастет картинка и 3D — сетка для этого не годится.

Точки сценария (объекты захвата и позиции сторон) пишутся в scenarios/*.json форматом
scenario.py: координаты в игровых единицах 0..ARENA, число позиций сверяется с составом из
units.json — иначе сценарий молча не загрузится в модель.

Оси: в векторе МЕТРЫ и y ВВЕРХ (свои внизу поля, враги вверху), как в бою. На экране это
переворот, и делается он ровно в одном месте — в матрице вида.

Запуск из корня проекта (нужны numpy, Pillow, tkinter):
    py -3.12 editor/map_editor.py
    py -3.12 editor/map_editor.py battle_3
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk
except ImportError as e:                                    # noqa: BLE001
    print(f"нужны tkinter и Pillow: {e}")
    raise SystemExit(1)


def _font(size=12):
    """Шрифт для подписей НА ХОЛСТЕ. Встроенный шрифт Pillow кириллицу не знает и рисует
    квадраты — «об.1» и «С» превращались в мусор."""
    for path in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P                                          # noqa: E402
from measure import measure                                  # noqa: E402
import terrain                                               # noqa: E402
import vectormap                                             # noqa: E402


def viewshed(tm, center_m, radius_m, m_per_unit, max_rays=1440):
    """Что видно из точки: поле 0..1 — сколько ЗАПАСА ПРОЗРАЧНОСТИ осталось у луча в этой клетке.

    Не «видно / не видно», а плавная величина, и это не украшение. В нашей модели лес не
    перекрывает обзор сразу: луч копит толщину и гаснет, набрав порог (90 м леса). Значит у
    опушки видно вглубь, а дальше сходит на нет — двоичная картинка это врала, показывая либо
    сплошной свет, либо сплошную тень, и именно поэтому «за лесами показывало плохо».

    Считается numpy'ем, все лучи разом: за мышью пером на питоне не угнаться, а так выходит
    десяток миллисекунд, и картинку можно тащить.

    Оговорка о приближении: бой копит толщину ОТДЕЛЬНО по каждому материалу, здесь она копится
    общей нормированной суммой. Различие проявляется только на луче, прошедшем и лес, и здание;
    у здания порог нулевой, оно гасит луч сразу, поэтому на практике картинка совпадает.
    """
    cx, cy = center_m[0] / m_per_unit, center_m[1] / m_per_unit
    R = radius_m / m_per_unit
    step = tm.cell * 0.5
    n_steps = max(2, int(R / step))
    n_rays = int(np.clip(2.0 * math.pi * R / step, 360, max_rays))

    ang = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False, dtype=np.float32)
    dist = (np.arange(1, n_steps + 1, dtype=np.float32) * step)[None, :]
    xs = cx + np.cos(ang)[:, None] * dist
    ys = cy + np.sin(ang)[:, None] * dist
    inside = (xs >= 0) & (ys >= 0) & (xs < tm.width_m) & (ys < tm.height_m)
    gx = np.clip((xs / tm.cell).astype(np.int32), 0, tm.Gx - 1)
    gy = np.clip((ys / tm.cell).astype(np.int32), 0, tm.Gy - 1)

    blocks = tm.f_blocks[gx, gy]
    lim = tm.f_see[gx, gy]
    # стоимость шага в долях «терпения» луча: непрозрачный материал с нулевым порогом (здание)
    # гасит сразу, прозрачный не стоит ничего
    # непрозрачное с нулевым порогом (здание) гасит луч сразу — берём большое КОНЕЧНОЕ число:
    # с бесконечностью следующая же строка даёт inf - inf = nan, и поле портится целиком
    cost = np.where(blocks, np.where(lim > 1e-6, step / np.maximum(lim, 1e-6), 1e6), 0.0)
    spent = np.cumsum(cost, axis=1) - cost          # накоплено ДО входа в клетку
    # Яркость = видно или нет, а НЕ «сколько терпения осталось». За тонкой лесополосой луч
    # тратит треть запаса, но цель за ней видно полностью — гасить там картинку значит врать.
    # Угасание показываем только ВНУТРИ растительности: туда и правда видно всё хуже с каждым
    # десятком метров, и опушка от чащи отличается именно этим.
    alive = (spent < 1.0)
    val = np.where(blocks, np.clip(1.0 - spent, 0.0, 1.0), alive.astype(np.float32))
    val = (val * alive).astype(np.float32)
    val[~inside] = 0.0

    field = np.zeros(tm.grid.shape, dtype=np.float32)
    np.maximum.at(field, (gx.ravel(), gy.ravel()), val.ravel())
    field[int(cx // tm.cell), int(cy // tm.cell)] = 1.0
    return field

BG = (16, 17, 20)
MAX_LAYER_PX = 3000

# Тёмная палитра под цвет холста: светлые панели рядом с тёмной картой сбивают глаз, и
# местность на них читается неверно — тот же лес кажется другим по яркости.
UI = {"bg": "#15161a", "panel": "#1c1e24", "line": "#2b2e37", "text": "#d7dae1",
      "dim": "#8a8f9b", "accent": "#e0b45c", "accent_dim": "#7a6636", "field": "#22252c"}


def apply_style(root):
    """Тема clam — единственная встроенная, которая позволяет красить всё. vista/xpnative
    игнорируют половину настроек цвета, и панели остаются светлыми."""
    st = ttk.Style(root)
    st.theme_use("clam")
    root.configure(background=UI["bg"])
    st.configure(".", background=UI["panel"], foreground=UI["text"],
                 fieldbackground=UI["field"], bordercolor=UI["line"],
                 lightcolor=UI["line"], darkcolor=UI["line"], focuscolor=UI["accent"])
    st.configure("TFrame", background=UI["panel"])
    st.configure("TLabel", background=UI["panel"], foreground=UI["text"])
    st.configure("Dim.TLabel", foreground=UI["dim"])
    st.configure("Head.TLabel", foreground=UI["accent"], font=("Segoe UI", 9, "bold"))
    st.configure("Status.TLabel", background=UI["bg"], foreground=UI["dim"],
                 font=("Consolas", 9))
    st.configure("TButton", background=UI["field"], foreground=UI["text"], borderwidth=1,
                 padding=(6, 4))
    st.map("TButton", background=[("active", UI["line"]), ("pressed", UI["accent_dim"])],
           foreground=[("active", UI["text"])])
    st.configure("Toolbutton", background=UI["field"], foreground=UI["text"], borderwidth=1,
                 padding=3)
    st.map("Toolbutton",
           background=[("selected", UI["accent_dim"]), ("active", UI["line"])],
           bordercolor=[("selected", UI["accent"])])
    st.configure("TCheckbutton", background=UI["panel"], foreground=UI["text"])
    st.map("TCheckbutton", background=[("active", UI["panel"])])
    st.configure("TRadiobutton", background=UI["panel"], foreground=UI["text"])
    st.map("TRadiobutton", background=[("active", UI["panel"])])
    st.configure("TEntry", fieldbackground=UI["field"], foreground=UI["text"],
                 insertcolor=UI["accent"], borderwidth=1)
    st.configure("TNotebook", background=UI["bg"], borderwidth=0)
    st.configure("TNotebook.Tab", background=UI["bg"], foreground=UI["dim"], padding=(12, 6))
    st.map("TNotebook.Tab", background=[("selected", UI["panel"])],
           foreground=[("selected", UI["accent"])])
    st.configure("TSeparator", background=UI["line"])
    st.configure("Horizontal.TScale", background=UI["panel"], troughcolor=UI["field"])
    return st
SHAPE_TYPES = ("forest", "water", "building", "road", "open")
TYPE_RU = {"forest": "лес", "water": "вода", "building": "застройка", "road": "дорога",
           "open": "поле (вырезает)"}
TYPE_COLOR = {"forest": P.TILE_COLORS[1], "water": P.TILE_COLORS[3],
              "building": P.TILE_COLORS[2], "road": P.TILE_COLORS[4], "open": P.TILE_COLORS[0]}
DEFAULT_WIDTH = {"road": 8.0, "water": 30.0, "forest": 25.0, "building": 20.0, "open": 40.0}

# Ширины линий по типам — те же грабли, что с домами: «лесополоса» и «опушка» отличаются
# ровно шириной, и подбирать её числом наугад неудобно. Кнопка просто заполняет поле.
LINE_PRESETS = {
    "road": [("тропа", 3), ("просёлок", 6), ("шоссе", 12)],
    "water": [("ручей", 8), ("речка", 20), ("река", 40)],
    # 3 клетки по 30 м — это 90 м, порог непрозрачности леса: тоньше не перекрывает обзор,
    # а только даёт укрытие (docs/JOURNAL.md, п. 3.3)
    "forest": [("изгородь", 8), ("лесополоса", 25), ("полоса леса", 90)],
    "open": [("просека", 30), ("выгон", 60)],
    "building": [("улица", 20)],
}
# Шаблоны строений в метрах — размеры настоящие, чтобы не прикидывать на глаз. Кнопка шаблона
# просто заполняет поля длины и ширины: размер дома задаётся числами, а не протяжкой мыши —
# протяжкой его не поставишь точно, а дома на карте отличаются именно размером.
HOUSE_PRESETS = [("сарай", 8, 6), ("изба", 12, 8), ("дом", 18, 10),
                 ("барак", 30, 10), ("ангар", 42, 18), ("склад", 60, 24)]
PREVIEW_MAX = 260            # клеток по стороне в предпросмотре: мельче глазу не нужно, а на
#                              театре 10 км честная сетка считалась бы секундами на каждый штрих


# ---------------------------------------------------------------- геометрия вида


def _m(a, b, c, d, e, f):
    return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]], dtype=np.float64)


def _translate(tx, ty):
    return _m(1, 0, tx, 0, 1, ty)


def _scale(sx, sy):
    return _m(sx, 0, 0, 0, sy, 0)


def _rotate(deg):
    t = math.radians(deg)
    return _m(math.cos(t), -math.sin(t), 0, math.sin(t), math.cos(t), 0)


def _glyph(kind, size=22, color=(224, 180, 92), fill=None):
    """Иконка инструмента, нарисованная кодом. Рисуем втрое крупнее и ужимаем — так у линий
    появляется сглаживание, а файлов с картинками в проекте не заводится."""
    S = size * 3
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = max(3, S // 11)
    c = (*color, 255)
    if kind == "select":
        d.polygon([(S * 0.28, S * 0.16), (S * 0.28, S * 0.80), (S * 0.45, S * 0.63),
                   (S * 0.56, S * 0.86), (S * 0.68, S * 0.79), (S * 0.57, S * 0.57),
                   (S * 0.78, S * 0.54)], fill=c)
    elif kind == "nodes":
        d.line([(S * 0.2, S * 0.72), (S * 0.45, S * 0.3), (S * 0.8, S * 0.6)], fill=c, width=w)
        for p in ((S * 0.2, S * 0.72), (S * 0.45, S * 0.3), (S * 0.8, S * 0.6)):
            d.rectangle([p[0] - S * 0.09, p[1] - S * 0.09, p[0] + S * 0.09, p[1] + S * 0.09],
                        fill=(255, 255, 255, 255), outline=c, width=w // 2)
    elif kind == "polygon":
        pts = [(S * 0.5, S * 0.14), (S * 0.86, S * 0.42), (S * 0.72, S * 0.85),
               (S * 0.28, S * 0.85), (S * 0.14, S * 0.42)]
        d.polygon(pts, fill=(*color, 70), outline=c)
        d.line(pts + [pts[0]], fill=c, width=w)
    elif kind == "line":
        d.line([(S * 0.16, S * 0.8), (S * 0.44, S * 0.36), (S * 0.84, S * 0.24)],
               fill=c, width=int(w * 1.8), joint="curve")
    elif kind == "building":
        d.polygon([(S * 0.5, S * 0.16), (S * 0.86, S * 0.44), (S * 0.14, S * 0.44)], fill=c)
        d.rectangle([S * 0.24, S * 0.44, S * 0.76, S * 0.84], fill=(*color, 90), outline=c, width=w)
    elif kind == "crossing":
        d.line([(S * 0.1, S * 0.66), (S * 0.9, S * 0.66)], fill=(70, 120, 190, 255),
               width=int(w * 2.4))
        d.arc([S * 0.16, S * 0.28, S * 0.84, S * 0.9], 200, 340, fill=c, width=int(w * 1.6))
        d.line([(S * 0.16, S * 0.6), (S * 0.16, S * 0.78)], fill=c, width=w)
        d.line([(S * 0.84, S * 0.6), (S * 0.84, S * 0.78)], fill=c, width=w)
    elif kind == "vision":
        d.ellipse([S * 0.12, S * 0.12, S * 0.88, S * 0.88], outline=c, width=w)
        d.pieslice([S * 0.12, S * 0.12, S * 0.88, S * 0.88], -35, 35, fill=(*color, 90))
        d.ellipse([S * 0.42, S * 0.42, S * 0.58, S * 0.58], fill=c)
    elif kind == "ruler":
        d.line([(S * 0.14, S * 0.72), (S * 0.86, S * 0.28)], fill=c, width=w)
        for t in (0.2, 0.4, 0.6, 0.8):
            x = S * (0.14 + 0.72 * t)
            y = S * (0.72 - 0.44 * t)
            d.line([(x, y), (x + S * 0.09, y + S * 0.14)], fill=c, width=max(2, w // 2))
    elif kind == "swatch":
        # тёмным заливкам (застройка) нужен светлый кант, иначе плашка сливается с панелью
        light = sum(fill) / 3 > 90
        d.rounded_rectangle([S * 0.08, S * 0.08, S * 0.92, S * 0.92], radius=S * 0.18,
                            fill=(*fill, 255),
                            outline=(*color, 160) if light else (150, 155, 165, 220),
                            width=max(2, w // 2))
    return ImageTk.PhotoImage(img.resize((size, size), Image.LANCZOS))


class View:
    """Мир (МЕТРЫ, y вверх) -> экран (пиксели, y вниз). Одной матрицей: поворот вида, поворот
    подложки и обратное преобразование под курсором — одна и та же арифметика, а руками
    расписанная она разъезжается (в проекте это уже стоило перепутанной оси Y)."""

    def __init__(self, zoom=0.3):
        self.cx = self.cy = 0.0
        self.zoom = zoom                                     # пикселей на метр
        self.angle = 0.0

    def matrix(self, w, h):
        return (_translate(w / 2, h / 2) @ _scale(self.zoom, -self.zoom)
                @ _rotate(self.angle) @ _translate(-self.cx, -self.cy))

    def inv(self, w, h):
        return np.linalg.inv(self.matrix(w, h))

    def to_screen(self, w, h, x, y):
        p = self.matrix(w, h) @ np.array([x, y, 1.0])
        return float(p[0]), float(p[1])

    def to_world(self, w, h, sx, sy):
        p = self.inv(w, h) @ np.array([sx, sy, 1.0])
        return float(p[0]), float(p[1])

    def fit(self, w, h, size_m):
        if w < 2 or h < 2:
            return
        self.angle = 0.0
        self.zoom = min(w / (size_m[0] * 1.06), h / (size_m[1] * 1.06))
        self.cx, self.cy = size_m[0] / 2, size_m[1] / 2


class Underlay:
    """Подложка-снимок под картой: своя прозрачность, поворот и масштаб (метров на пиксель)."""

    def __init__(self, image, name, center, m_per_px, angle=0.0, opacity=0.6, path=None):
        self.image = image
        self.name = name
        self.path = path
        self.cx, self.cy = center
        self.m_per_px = float(m_per_px)
        self.angle = float(angle)
        self.opacity = float(opacity)
        self.visible = True

    def world_to_image(self):
        w, h = self.image.size
        return (_translate(w / 2, h / 2) @ _scale(1 / self.m_per_px, -1 / self.m_per_px)
                @ _rotate(-self.angle) @ _translate(-self.cx, -self.cy))


# ---------------------------------------------------------------- документ


class Doc:
    """Карта в работе: вектор — источник, всё остальное производное."""

    def __init__(self, vec, cell_m, name):
        self.vec = vec
        self.cell_m = float(cell_m)
        self.name = name
        saved = vec.get("markers") or {}
        self.markers = {k: [list(map(float, p)) for p in saved.get(k, [])] for k in P.MARKER_KINDS}
        self.underlays = []
        self.version = 0
        self._surface = None
        self._surface_key = None
        self._graph = None
        self._graph_key = None
        self._tm = None
        self._tm_key = None

    @property
    def shapes(self):
        return self.vec["shapes"]

    @property
    def size_m(self):
        return self.vec["size_m"]

    def bump(self):
        self.version += 1

    def graph(self):
        """Граф дорог: узлы-перекрёстки и участки. Считается из вектора тем же кодом, что при
        сборке, и кэшируется по версии — иначе пересчитывался бы на каждый кадр."""
        if self._graph_key != self.version:
            self._graph = vectormap.road_graph(self.vec)
            self._graph_key = self.version
        return self._graph

    def terrain_map(self, cell_m=None):
        """TerrainMap из вектора — та же, что получит бой. Нужна для просмотра видимости:
        считать её надо НАСТОЯЩИМ движком, иначе инструмент покажет одно, а в бою будет другое
        (ровно на этом расхождении сидят жалобы игроков WARNO на опушки)."""
        cell = cell_m or self.cell_m
        key = (self.version, round(cell, 3), "tm")
        if self._tm_key != key:
            surface, fields, comp, cap = vectormap.rasterize(self.vec, cell)
            self._tm = terrain.from_fields(surface, fields, cell / P.M_PER_UNIT, cap, comp)
            self._tm_key = key
        return self._tm

    def surface(self, cell_m=None):
        """Растеризация в сетку. Для показа клетка берётся покрупнее (PREVIEW_MAX)."""
        cell = cell_m or max(self.cell_m, max(self.size_m) / PREVIEW_MAX)
        key = (self.version, round(cell, 3))
        if self._surface_key != key:
            self._surface = vectormap.rasterize(self.vec, cell)[0]
            self._surface_key = key
        return self._surface, cell

    def snapshot(self):
        return (copy.deepcopy(self.vec["shapes"]), copy.deepcopy(self.markers))

    def restore(self, snap):
        self.vec["shapes"] = copy.deepcopy(snap[0])
        self.markers = copy.deepcopy(snap[1])
        self.bump()


# ---------------------------------------------------------------- стартовый экран


class StartScreen(ttk.Frame):
    """Холста при запуске нет: карта либо создаётся с явным размером, либо открывается,
    либо генерируется — рисовать двадцать километров с нуля никто не станет."""

    def __init__(self, master, on_open):
        super().__init__(master, padding=24)
        self.on_open = on_open
        ttk.Label(self, text="WarGame — редактор карт (вектор)", font=("Segoe UI", 16)).pack(anchor="w")
        ttk.Label(self, text=P.describe(), foreground="#888").pack(anchor="w", pady=(2, 16))

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Button(row, text="Создать пустую…", command=self.new_map).pack(side="left")
        ttk.Button(row, text="Сгенерировать…", command=self.gen_map).pack(side="left", padx=6)
        ttk.Button(row, text="Открыть файл…", command=self.open_dialog).pack(side="left")

        ttk.Label(self, text="векторные карты в maps/", foreground="#888").pack(anchor="w", pady=(18, 2))
        box = ttk.Frame(self)
        box.pack(fill="both", expand=True)
        self.lst = tk.Listbox(box, height=14, width=74, activestyle="none")
        self.lst.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, command=self.lst.yview)
        sb.pack(side="left", fill="y")
        self.lst.config(yscrollcommand=sb.set)
        self.lst.bind("<Double-Button-1>", lambda e: self.open_selected())
        ttk.Button(self, text="Открыть выбранную", command=self.open_selected).pack(anchor="w", pady=8)

        self.files = []
        for path in sorted(_glob_vectors()):
            try:
                doc = vectormap.load(path)
                w, h = doc["size_m"]
                kinds = {}
                for s in doc["shapes"]:
                    kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
                self.lst.insert("end", f"{os.path.basename(path)[:-12]:<26} {w:.0f}x{h:.0f} м   "
                                       + "  ".join(f"{k}: {v}" for k, v in sorted(kinds.items())))
                self.files.append(path)
            except Exception as exc:                          # noqa: BLE001
                self.lst.insert("end", f"{os.path.basename(path)} — не читается: {exc}")
                self.files.append(None)

    def new_map(self):
        NewMapDialog(self, lambda w, h, cell: self.on_open(
            Doc(vectormap.new_doc((w, h)), cell, "new_map")))

    def gen_map(self):
        GenDialog(self, self.on_open)

    def open_dialog(self):
        path = filedialog.askopenfilename(initialdir=P.MAPS,
                                          filetypes=[("векторная карта", "*.vector.json")])
        if path:
            self.on_open(load_doc(path))

    def open_selected(self):
        sel = self.lst.curselection()
        if sel and self.files[sel[0]]:
            self.on_open(load_doc(self.files[sel[0]]))


class NewMapDialog(tk.Toplevel):
    def __init__(self, master, on_ok):
        super().__init__(master)
        self.title("новая карта")
        self.on_ok = on_ok
        self.resizable(False, False)
        f = ttk.Frame(self, padding=14)
        f.pack()
        self.w = tk.DoubleVar(value=P.ARENA_M)
        self.h = tk.DoubleVar(value=P.ARENA_M)
        self.cell = tk.DoubleVar(value=P.CELL_M)
        for i, (label, var) in enumerate((("ширина, м", self.w), ("высота, м", self.h),
                                          ("клетка полей, м", self.cell))):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=var, width=10).grid(row=i, column=1, sticky="w", padx=6)
            var.trace_add("write", lambda *_: self.refresh())
        self.info = ttk.Label(f, text="", justify="left")
        self.info.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 6))
        ttk.Button(f, text=f"под арену ({P.ARENA_M:.0f} м)",
                   command=self.preset).grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Button(f, text="создать", command=self.ok).grid(row=5, column=0, columnspan=2,
                                                            sticky="ew", pady=(8, 0))
        self.refresh()
        self.grab_set()

    def preset(self):
        self.w.set(P.ARENA_M)
        self.h.set(P.ARENA_M)
        self.cell.set(P.CELL_M)

    def _v(self):
        try:
            return float(self.w.get()), float(self.h.get()), float(self.cell.get())
        except (tk.TclError, ValueError):
            return None

    def refresh(self):
        v = self._v()
        if not v or min(v) <= 0:
            self.info.config(text="размеры не разобраны", foreground="#c66")
            return
        w, h, cell = v
        ok = abs(w - P.ARENA_M) < P.ARENA_M * 0.05 and abs(h - P.ARENA_M) < P.ARENA_M * 0.05
        self.info.config(text=f"сетка {w / cell:.0f} x {h / cell:.0f} клеток\n"
                              + ("совпадает с ареной — боевая карта" if ok else
                                 f"не арена ({P.ARENA_M:.0f} м) — годится как театр, "
                                 "из которого режут бои"),
                         foreground="#8b8" if ok else "#c96")

    def ok(self):
        v = self._v()
        if v:
            self.destroy()
            self.on_ok(*v)


class GenDialog(tk.Toplevel):
    """Сгенерировать местность и открыть на правку: рисовать с нуля обычно незачем, проще
    получить скелет и поправить то, что важно."""

    def __init__(self, master, on_open):
        super().__init__(master)
        self.title("сгенерировать карту")
        self.on_open = on_open
        self.resizable(False, False)
        f = ttk.Frame(self, padding=14)
        f.pack()
        self.size = tk.DoubleVar(value=P.ARENA_M)
        self.seed = tk.IntVar(value=0)
        self.cell = tk.DoubleVar(value=P.CELL_M)
        self.any = tk.BooleanVar(value=False)
        for i, (label, var) in enumerate((("размер, м", self.size), ("сид", self.seed),
                                          ("клетка полей, м", self.cell))):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=var, width=10).grid(row=i, column=1, sticky="w", padx=6)
        ttk.Checkbutton(f, text="не переигрывать ради годности", variable=self.any).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.busy = ttk.Label(f, text="")
        self.busy.grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(f, text="сгенерировать", command=self.ok).grid(row=5, column=0, columnspan=2,
                                                                  sticky="ew", pady=(10, 0))
        self.grab_set()

    def ok(self):
        import mapgen
        self.busy.config(text="генерирую…")
        self.update_idletasks()
        size, seed, cell = float(self.size.get()), int(self.seed.get()), float(self.cell.get())
        vec, m, tries = mapgen.generate_good(size, seed, cell_m=max(cell, size / 200),
                                             battle_edges=abs(size - P.ARENA_M) < P.ARENA_M * 0.3,
                                             any_map=bool(self.any.get()))
        self.destroy()
        self.on_open(Doc(vec, cell, f"gen_{seed}"))


# ---------------------------------------------------------------- редактор


class EditorFrame(ttk.Frame):
    # В интерфейсе только то, что работает на текущую задачу — рисование карт. Точки сценария,
    # панель замера, подложки, превью и пул убраны из окна, но НЕ из кода: методы на месте и
    # вызываются из проверок, вернуть их в интерфейс — одна строка.
    TOOLS = [("выбор / перенос", "select"), ("узлы", "nodes"), ("полигон", "polygon"),
             ("линия", "line"), ("дом", "building"), ("переправа", "crossing"),
             ("линейка", "ruler"), ("видимость", "vision")]

    def __init__(self, master, doc, on_close):
        super().__init__(master)
        self.doc = doc
        self.on_close = on_close
        self.view = View()
        self.undo, self.redo = [], []
        self.shape_type = tk.StringVar(value="forest")
        self.tool = tk.StringVar(value="select")
        self.width_m = tk.DoubleVar(value=DEFAULT_WIDTH["forest"])
        self.house_len = tk.DoubleVar(value=18.0)
        self.house_wid = tk.DoubleVar(value=10.0)
        self.bridge_auto = tk.BooleanVar(value=True)
        self.bridge_len = tk.DoubleVar(value=80.0)
        self.bridge_wid = tk.DoubleVar(value=8.0)
        self.ruler_circle = tk.BooleanVar(value=False)
        self.vision_r = tk.DoubleVar(value=1000.0)
        self.dash_on = tk.BooleanVar(value=False)
        self.dash_len = tk.DoubleVar(value=45.0)
        self.dash_gap = tk.DoubleVar(value=30.0)
        self.marker_kind = tk.StringVar(value="zones")
        self.show_grid = tk.BooleanVar(value=False)      # сетка — отладочный слой, не фон
        self.show_graph = tk.BooleanVar(value=True)
        self.grid_alpha = 0.6
        self.metrics = None

        # Подложки живут без интерфейса: сами объекты и вся арифметика на месте, нет только
        # панели. Понадобится обводить снимки — вернуть вкладку дешевле, чем писать заново.
        self.font = _font(12)
        self.font_small = _font(11)
        self.u_index = None
        self.u_move = tk.BooleanVar(value=False)
        self.u_visible = tk.BooleanVar(value=True)
        self.u_opacity = tk.DoubleVar(value=0.6)
        self.u_angle = tk.DoubleVar(value=0.0)
        self.u_scale = tk.DoubleVar(value=0.0)
        self.selected = None            # индекс фигуры
        self._prev_type = None          # чем рисовали до переключения на дом
        self._draft = None              # незакрытая фигура: список точек
        self._ruler = None              # два конца линейки, в метрах
        self._vision = None             # (точка, маска видимости, клетка)
        self._drag = None
        self._panning = False
        self._measure_job = None
        self._syncing = False
        self.W = self.H = 10

        self._build()
        self.after(60, self._first_open)

    # --- виджеты

    def _build(self):
        self.canvas = tk.Canvas(self, background="#101114", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double)
        self.canvas.bind("<ButtonPress-3>", self.on_right)
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        side = ttk.Frame(self, width=352)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        nb = ttk.Notebook(side)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._tab_draw(nb)
        self._tab_file(nb)
        self.status = ttk.Label(side, text="", anchor="w", wraplength=336, justify="left",
                                style="Status.TLabel")
        self.status.pack(fill="x", padx=8, pady=(0, 8))
        self.tool.trace_add("write", lambda *_: self._tool_changed())

        self._bound = []
        for seq, fn in (
                *[(k, lambda v=v: self._set_type(v)) for k, v in zip("12345", SHAPE_TYPES)],
                *[(k, lambda v=v: self.tool.set(v))
                  for k, v in zip("qwertyu", [t[1] for t in self.TOOLS])],
                ("<Control-z>", self.do_undo), ("<Control-y>", self.do_redo),
                ("<Control-s>", self.save_map), ("<F5>", self.remeasure),
                ("<Delete>", self.delete_selected), ("<Escape>", self.cancel_draft),
                ("<Return>", self.finish_draft),
                ("<plus>", lambda: self.zoom_by(1.25)), ("<minus>", lambda: self.zoom_by(0.8)),
                ("0", self.fit_view),
                ("<KeyPress-space>", lambda: setattr(self, "_panning", True)),
                ("<KeyRelease-space>", lambda: setattr(self, "_panning", False))):
            self._bind_hotkey(seq, fn)

    def _bind_hotkey(self, seq, fn):
        """Горячая клавиша не должна срабатывать, пока набирают имя карты."""
        def handler(ev, fn=fn):
            if isinstance(self.focus_get(), (tk.Entry, ttk.Entry, tk.Text, tk.Listbox)):
                return None
            fn()
            return "break"

        self.winfo_toplevel().bind(seq, handler)
        self._bound.append(seq)

    def destroy(self):
        top = self.winfo_toplevel()
        for seq in getattr(self, "_bound", []):
            top.unbind(seq)
        super().destroy()

    def _tab_draw(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="рисование")
        self._icons = {}                       # держим ссылки: Tk не считает их за владение
        self._type_btns = {}

        ttk.Label(f, text="ЗОНИРОВАНИЕ  [1-5]", style="Head.TLabel").pack(anchor="w")
        # Плашки С ПОДПИСЯМИ: одни цветные квадраты не читаются — «застройка» тёмно-серая и на
        # тёмной панели теряется, из-за чего кажется, что типов меньше, чем есть.
        chips = ttk.Frame(f)
        chips.pack(anchor="w", fill="x", pady=(4, 2))
        for i, t in enumerate(SHAPE_TYPES):
            self._icons[t] = _glyph("swatch", 22, fill=TYPE_COLOR[t])
            name = TYPE_RU[t].split(" ")[0]
            b = ttk.Radiobutton(chips, image=self._icons[t], text=f" {name}", compound="left",
                                value=t, variable=self.shape_type, style="Toolbutton", width=13,
                                command=lambda: self._set_type(self.shape_type.get()))
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
            self._type_btns[t] = b
        self.type_label = ttk.Label(f, text="", style="Dim.TLabel", wraplength=320)
        self.type_label.pack(anchor="w", pady=(4, 0))

        ttk.Separator(f).pack(fill="x", pady=9)
        ttk.Label(f, text="ИНСТРУМЕНТ", style="Head.TLabel").pack(anchor="w")
        grid = ttk.Frame(f)
        grid.pack(anchor="w", pady=(4, 2))
        for i, (label, val) in enumerate(self.TOOLS):
            self._icons[val] = _glyph(val, 26)
            ttk.Radiobutton(grid, image=self._icons[val], value=val, variable=self.tool,
                            style="Toolbutton", command=self._tool_changed).grid(
                row=i // 3, column=i % 3, padx=2, pady=2)
        self.tool_label = ttk.Label(f, text="", style="Dim.TLabel", wraplength=320,
                                    justify="left")
        self.tool_label.pack(anchor="w", pady=(2, 0))

        # Шаблоны строений — появляются только при выбранном инструменте «дом»: панель, которая
        # висит всегда, перестаёт читаться, а размеры домов нужны ровно в одном режиме.
        self.house_box = ttk.Frame(f)
        ttk.Label(self.house_box, text="РАЗМЕР СТРОЕНИЯ", style="Head.TLabel").pack(anchor="w",
                                                                                    pady=(6, 2))
        sizes = ttk.Frame(self.house_box)
        sizes.pack(anchor="w", pady=(0, 4))
        ttk.Label(sizes, text="длина").pack(side="left")
        ttk.Entry(sizes, textvariable=self.house_len, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(sizes, text="ширина").pack(side="left")
        ttk.Entry(sizes, textvariable=self.house_wid, width=6).pack(side="left", padx=4)
        grid = ttk.Frame(self.house_box)
        grid.pack(anchor="w")
        for i, (name, w, h) in enumerate(HOUSE_PRESETS):
            ttk.Button(grid, text=f"{name} {w}×{h}", width=13,
                       command=lambda w=w, h=h: self._set_house(w, h)).grid(
                row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")

        self.bridge_box = ttk.Frame(f)
        ttk.Label(self.bridge_box, text="ПЕРЕПРАВА", style="Head.TLabel").pack(anchor="w",
                                                                               pady=(6, 2))
        ttk.Checkbutton(self.bridge_box, text="размеры по дороге и реке",
                        variable=self.bridge_auto).pack(anchor="w")
        brow = ttk.Frame(self.bridge_box)
        brow.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Label(brow, text="длина").pack(side="left")
        ttk.Entry(brow, textvariable=self.bridge_len, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(brow, text="ширина").pack(side="left")
        ttk.Entry(brow, textvariable=self.bridge_wid, width=6).pack(side="left", padx=4)

        self.tool_box = ttk.Frame(f)          # панель линейки и видимости
        ttk.Checkbutton(self.tool_box, text="линейка кругом (радиус и кольца)",
                        variable=self.ruler_circle, command=self.draw).pack(anchor="w", pady=(6, 0))
        vrow = ttk.Frame(self.tool_box)
        vrow.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Label(vrow, text="радиус обзора, м").pack(side="left")
        ttk.Entry(vrow, textvariable=self.vision_r, width=7).pack(side="left", padx=6)

        self.line_box = ttk.Frame(f)
        ttk.Label(self.line_box, text="ШИРИНА ЛИНИИ", style="Head.TLabel").pack(anchor="w",
                                                                                pady=(6, 2))
        wrow = ttk.Frame(self.line_box)
        wrow.pack(anchor="w", fill="x")
        self.width_entry = ttk.Entry(wrow, textvariable=self.width_m, width=7)
        self.width_entry.pack(side="left")
        ttk.Label(wrow, text="м").pack(side="left", padx=4)
        self.line_btns = ttk.Frame(self.line_box)
        self.line_btns.pack(anchor="w", pady=(4, 0))
        drow = ttk.Frame(self.line_box)
        drow.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Checkbutton(drow, text="пунктиром", variable=self.dash_on).pack(side="left")
        ttk.Entry(drow, textvariable=self.dash_len, width=5).pack(side="left", padx=(8, 2))
        ttk.Label(drow, text="штрих").pack(side="left")
        ttk.Entry(drow, textvariable=self.dash_gap, width=5).pack(side="left", padx=(8, 2))
        ttk.Label(drow, text="прореха, м").pack(side="left")

        ttk.Separator(f).pack(fill="x", pady=9)
        row = ttk.Frame(f)
        row.pack(fill="x")
        for label, cmd, w in (("↶", self.do_undo, 4), ("↷", self.do_redo, 4),
                              ("скруглить", self.do_round, 11),
                              ("удалить", self.delete_selected, 10)):
            ttk.Button(row, text=label, width=w, command=cmd).pack(side="left", padx=(0, 3))
        ttk.Button(f, text="зеркалить карту по Y", command=self.do_mirror).pack(fill="x", pady=(3, 0))

        ttk.Separator(f).pack(fill="x", pady=9)
        ttk.Label(f, text="ВИД", style="Head.TLabel").pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        for label, cmd, w in (("−", lambda: self.zoom_by(0.8), 4),
                              ("+", lambda: self.zoom_by(1.25), 4),
                              ("вписать", self.fit_view, 9),
                              ("↺", lambda: self.rotate_view(90), 4),
                              ("↻", lambda: self.rotate_view(-90), 4)):
            ttk.Button(row, text=label, width=w, command=cmd).pack(side="left", padx=(0, 3))
        self.view_angle = tk.DoubleVar(value=0.0)
        ttk.Checkbutton(f, text="сетка полей — что читает бой", variable=self.show_grid,
                        command=self.draw).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(f, text="узлы дорог — перекрёстки и разрывы", variable=self.show_graph,
                        command=self._changed).pack(anchor="w")
        ttk.Label(f, text="колесо — зум · СКМ или пробел — тащить · F5 — замер",
                  style="Dim.TLabel", wraplength=320, justify="left").pack(anchor="w", pady=(2, 0))

        ttk.Separator(f).pack(fill="x", pady=9)
        self.shape_info = tk.Text(f, height=12, width=34, font=("Consolas", 9), relief="flat",
                                  background=UI["field"], foreground=UI["text"],
                                  highlightthickness=0, padx=8, pady=6)
        self.shape_info.pack(fill="x")
        self._tool_changed()

    HINTS = {"select": "щелчок выбирает, перетаскивание двигает",
             "nodes": "тяни узлы выбранной фигуры, ПКМ удаляет узел",
             "polygon": "ЛКМ — точка, двойной щелчок / ПКМ / Enter — замкнуть, Esc — отмена",
             "line": "ЛКМ — точка, ширина задаётся полем ниже",
             "building": "щелчок ставит дом заданного размера, протяжка доворачивает",
             "crossing": "щелчок ставит мост, протяжка доворачивает; размеры — авто или числами",
             "ruler": "протяни — покажет расстояние; галка «кругом» меряет радиусом",
             "vision": "щелчок — что видно из этой точки настоящей моделью линии огня"}

    # Какое зонирование имеет смысл при каком инструменте. Дом — всегда застройка, переправа
    # сама решает, что пробивает, а линейке и видимости тип не нужен вовсе. Раньше выбор висел
    # независимо, и «вода» при инструменте «дом» выглядела так, будто сейчас поставится пруд.
    TOOL_TYPES = {"polygon": SHAPE_TYPES, "line": SHAPE_TYPES, "building": ("building",)}

    def _sync_types(self):
        allowed = self.TOOL_TYPES.get(self.tool.get(), ())
        if allowed and self.shape_type.get() not in allowed:
            if self.tool.get() == "building":
                self._prev_type = self.shape_type.get()
            self._set_type(allowed[0])
        elif not allowed:
            pass
        elif self.tool.get() != "building" and self._prev_type:
            self._set_type(self._prev_type)     # вернуть то, чем рисовали до дома
            self._prev_type = None
        for t, b in self._type_btns.items():
            b.state(["!disabled"] if t in allowed else ["disabled"])

    def _tool_changed(self):
        """Панель размеров показывается только для того инструмента, которому она нужна:
        постоянно висящие поля перестают читаться."""
        if not hasattr(self, "tool_label"):
            return
        self._sync_types()
        self.tool_label.config(text=self.HINTS.get(self.tool.get(), ""))
        self.house_box.pack_forget()
        self.line_box.pack_forget()
        self.tool_box.pack_forget()
        self.bridge_box.pack_forget()
        if self.tool.get() == "building":
            self.house_box.pack(anchor="w", fill="x", after=self.tool_label)
        elif self.tool.get() == "line":
            self.line_box.pack(anchor="w", fill="x", after=self.tool_label)
            self._refresh_line_presets()
        elif self.tool.get() == "crossing":
            self.bridge_box.pack(anchor="w", fill="x", after=self.tool_label)
        elif self.tool.get() in ("ruler", "vision"):
            self.tool_box.pack(anchor="w", fill="x", after=self.tool_label)

    def _refresh_line_presets(self):
        for w in self.line_btns.winfo_children():
            w.destroy()
        for i, (name, width) in enumerate(LINE_PRESETS.get(self.shape_type.get(), [])):
            ttk.Button(self.line_btns, text=f"{name} {width} м", width=15,
                       command=lambda v=width: self._set_width(v)).grid(
                row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")

    def _set_width(self, v):
        self.width_m.set(float(v))
        self.status.config(text=f"ширина линии: {v} м")

    def _set_house(self, w, h):
        self.house_len.set(float(w))
        self.house_wid.set(float(h))
        self.status.config(text=f"размер строения: {w} × {h} м")

    def _house_size(self):
        try:
            return max(float(self.house_len.get()), 1.0), max(float(self.house_wid.get()), 1.0)
        except (tk.TclError, ValueError):
            return 18.0, 10.0

    def _tab_file(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="файл")
        self.name_var = tk.StringVar(value=self.doc.name)
        ttk.Label(f, text="ИМЯ КАРТЫ", style="Head.TLabel").pack(anchor="w")
        ttk.Entry(f, textvariable=self.name_var).pack(fill="x", pady=(4, 0))
        ttk.Label(f, text="пишется в maps/", style="Dim.TLabel").pack(anchor="w")
        ttk.Label(f, text="КЛЕТКА ПОЛЕЙ, М", style="Head.TLabel").pack(anchor="w", pady=(10, 0))
        self.cell_var = tk.DoubleVar(value=self.doc.cell_m)
        ttk.Entry(f, textvariable=self.cell_var, width=8).pack(anchor="w", pady=(4, 0))
        ttk.Button(f, text="сохранить и собрать   Ctrl+S", command=self.save_map).pack(
            fill="x", pady=(12, 1))
        ttk.Label(f, text="пишется <имя>.vector.json и тут же собираются поля и граф дорог — "
                          "карта сразу готова к бою.",
                  style="Dim.TLabel", wraplength=310, justify="left").pack(anchor="w", pady=6)
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Button(f, text="закрыть карту", command=lambda: self.on_close()).pack(fill="x")
        ttk.Label(f, text=P.describe(), wraplength=316, foreground="#888").pack(anchor="w", pady=8)

    def _set_type(self, t):
        allowed = self.TOOL_TYPES.get(self.tool.get(), ())
        if allowed and t not in allowed:
            label = dict((v, k) for k, v in self.TOOLS).get(self.tool.get(), self.tool.get())
            self.status.config(text=f"инструмент «{label}» не рисует «{TYPE_RU[t]}»")
            return
        self.shape_type.set(t)
        self.width_m.set(DEFAULT_WIDTH[t])
        if hasattr(self, "type_label"):
            note = "  ·  вырезает лес и дороги" if t == "open" else \
                   f"  ·  ширина линии {DEFAULT_WIDTH[t]:.0f} м"
            self.type_label.config(text=TYPE_RU[t] + note)
        if hasattr(self, "line_btns") and self.tool.get() == "line":
            self._refresh_line_presets()

    # --- вид

    def _first_open(self):
        self.fit_view()
        self._refresh_points()
        self._refresh_shapes()

    def on_resize(self, ev):
        self.W, self.H = max(2, ev.width), max(2, ev.height)
        self.draw()

    def fit_view(self):
        self.view.fit(self.W, self.H, self.doc.size_m)
        self.view_angle.set(0.0)
        self.draw()

    def zoom_by(self, k, anchor=None):
        ax, ay = anchor if anchor else (self.W / 2, self.H / 2)
        before = self.view.to_world(self.W, self.H, ax, ay)
        self.view.zoom = float(np.clip(self.view.zoom * k, 0.004, 8.0))
        after = self.view.to_world(self.W, self.H, ax, ay)
        self.view.cx += before[0] - after[0]
        self.view.cy += before[1] - after[1]
        self.draw()

    def rotate_view(self, deg):
        self._set_view_angle(self.view.angle + deg)
        self.view_angle.set(self.view.angle)

    def _set_view_angle(self, deg):
        self.view.angle = ((deg + 180) % 360) - 180
        self.draw()

    def on_wheel(self, ev):
        self.zoom_by(1.15 if ev.delta > 0 else 1 / 1.15, anchor=(ev.x, ev.y))

    # --- отрисовка

    def _S(self, p):
        return self.view.to_screen(self.W, self.H, p[0], p[1])

    def draw(self):
        if self.W < 4 or self.H < 4:
            return
        W_m, H_m = self.doc.size_m
        out = Image.new("RGB", (self.W, self.H), BG)

        for u in self.doc.underlays:
            if not u.visible:
                continue
            coeffs = (u.world_to_image() @ self.view.inv(self.W, self.H))[:2].flatten()
            warped = u.image.transform((self.W, self.H), Image.AFFINE, tuple(coeffs),
                                       resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))
            a = warped.getchannel("A").point(lambda v, o=u.opacity: int(v * o))
            out.paste(warped.convert("RGB"), (0, 0), a)

        if self.show_grid.get():
            surface, cell = self.doc.surface()
            lut = np.zeros((max(P.TILE_COLORS) + 1, 4), dtype=np.uint8)
            for k, c in P.TILE_COLORS.items():
                lut[k] = (*c, 255)
            img = Image.fromarray(np.ascontiguousarray(lut[surface].transpose(1, 0, 2)[::-1]))
            gy = surface.shape[1]
            w2i = _translate(0, gy) @ _scale(1 / cell, -1 / cell)
            coeffs = (w2i @ self.view.inv(self.W, self.H))[:2].flatten()
            warped = img.transform((self.W, self.H), Image.AFFINE, tuple(coeffs),
                                   resample=Image.NEAREST, fillcolor=(0, 0, 0, 0))
            a = warped.getchannel("A").point(lambda v, o=self.grid_alpha: int(v * o))
            out.paste(warped.convert("RGB"), (0, 0), a)

        if self._vision:
            warped = self._vision_overlay()
            out.paste(warped.convert("RGB"), (0, 0), warped.getchannel("A"))

        over = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        self._draw_shapes(d)
        self._draw_frame(d)
        if self.show_graph.get():
            self._draw_graph(d)
        self._draw_markers(d)
        if self._vision:
            self._draw_vision(d)
        if self._ruler:
            self._draw_ruler(d)
        self._draw_hud(d)
        out = Image.alpha_composite(out.convert("RGBA"), over).convert("RGB")

        self._photo = ImageTk.PhotoImage(out)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _draw_shapes(self, d):
        """Фигуры — полупрозрачной заливкой с контуром. Сквозь них видно сетку: весь смысл
        показа сетки в том, чтобы замечать расхождение между тем, что нарисовано, и тем, что
        досталось бою (тонкая дорога, узкая лесополоса)."""
        order = {"water": 0, "forest": 1, "road": 2}
        idx = sorted(range(len(self.doc.shapes)),
                     key=lambda i: order.get(self.doc.shapes[i].get("type"), 9))
        for i in idx:
            sh = self.doc.shapes[i]
            sel = (i == self.selected)
            col = TYPE_COLOR.get(sh.get("type"), (200, 200, 200))
            outline = (255, 240, 150) if sel else tuple(min(255, int(c * 1.5) + 40) for c in col)
            if sh["kind"] == "polygon":
                pts = [self._S(p) for p in sh["points"]]
                if len(pts) >= 3:
                    d.polygon(pts, fill=(*col, 120), outline=(*outline, 230))
            elif sh["kind"] == "line":
                # пунктир рисуем ТЕМ ЖЕ разбиением, что уйдёт в сетку: иначе на чертеже полоса
                # сплошная, в бою дырявая, и понять, где прорехи, нельзя
                w = max(1, int(sh.get("width_m", 8) * self.view.zoom))
                for run in self._line_runs(sh):
                    pts = [self._S(p) for p in run]
                    d.line(pts, fill=(*col, 170), width=w, joint="curve")
                    if sel:
                        d.line(pts, fill=(*outline, 255), width=1)
            elif sh["kind"] == "building":
                cx, cy, bw, bh, ang = sh["rect_m"]
                pts = [self._S(p) for p in vectormap._rect_points(cx, cy, bw, bh, ang)]
                d.polygon(pts, fill=(*col, 210), outline=(*outline, 255))
            elif sh["kind"] == "crossing":
                # рисуем ровно ту полосу, что пробьётся в воде, — иначе на экране кружок, а в
                # бою узкий проезд (или наоборот), и понять, где перейдут реку, нельзя
                c, length, width, ang = vectormap.crossing_geom(self.doc.vec, sh)
                pts = [self._S(q) for q in vectormap._rect_points(c[0], c[1], length,
                                                                  max(width, 4.0), ang)]
                d.polygon(pts, fill=(210, 175, 110, 210), outline=(255, 230, 160, 255))
            if sel and sh["kind"] in ("polygon", "line") and self.tool.get() == "nodes":
                for p in sh["points"]:
                    x, y = self._S(p)
                    d.rectangle([x - 4, y - 4, x + 4, y + 4], fill=(255, 240, 150),
                                outline=(40, 40, 40))
        if self._draft:
            pts = [self._S(p) for p in self._draft]
            col = TYPE_COLOR[self.shape_type.get()]
            if len(pts) > 1:
                d.line(pts, fill=(*col, 255), width=2)
            for x, y in pts:
                d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))

    @staticmethod
    def _line_runs(sh):
        """Куски линии с учётом пунктира — ровно те, что растеризуются в поля."""
        dash = sh.get("dash_m")
        if not dash:
            return [sh["points"]]
        return vectormap.dash_polyline(sh["points"], dash[0], dash[1])

    def _draw_frame(self, d):
        W_m, H_m = self.doc.size_m
        S = self._S
        band = 300.0
        # полосы развёртывания: свои внизу, враги вверху (reset() в wargame_env). Карта, у
        # которой всё укрытие на одной половине, несправедлива, а на глаз это не видно.
        d.polygon([S((0, 0)), S((W_m, 0)), S((W_m, band)), S((0, band))], fill=(80, 140, 255, 30))
        d.polygon([S((0, H_m - band)), S((W_m, H_m - band)), S((W_m, H_m)), S((0, H_m))],
                  fill=(255, 90, 90, 30))
        if self.view.zoom * 300.0 > 14:
            x = 0.0
            while x <= W_m:
                d.line([S((x, 0)), S((x, H_m))], fill=(255, 255, 255, 22))
                x += 300.0
            y = 0.0
            while y <= H_m:
                d.line([S((0, y)), S((W_m, y))], fill=(255, 255, 255, 22))
                y += 300.0
        d.line([S((0, 0)), S((W_m, 0)), S((W_m, H_m)), S((0, H_m)), S((0, 0))],
               fill=(210, 210, 210, 130), width=2)

    def _draw_graph(self, d):
        """Перекрёстки и участки дорог. Смотреть на это стоит по двум причинам: перекрёсток —
        это ориентир «узел дороги», по которому отдаются приказы, и он же будущий узел маршрута.
        А ещё граф сразу показывает РАЗРЫВ сети: если кусок карты отрезан (обычно забыт мост),
        его узлы окрасятся красным, и это видно до всякого боя."""
        # во время протяжки граф не пересчитываем: он строится перебором пересечений, и на
        # театре это доли секунды — заметный рывок на каждом кадре перетаскивания
        if self._drag and self._drag[0] in ("move", "node", "building", "building_rot",
                                            "crossing_rot"):
            return
        g = self.doc.graph()
        parent = list(range(len(g["nodes"])))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for e in g["edges"]:
            ra, rb = find(e["a"]), find(e["b"])
            if ra != rb:
                parent[ra] = rb
        comps = {}
        for i in range(len(g["nodes"])):
            comps.setdefault(find(i), []).append(i)
        main = max(comps.values(), key=len) if comps else []
        main = set(main)

        for e in g["edges"]:
            # ведём ребро ПО ДОРОГЕ, а не хордой между узлами: на извилистой дороге хорда
            # прочерчивала прямую через полкарты и выглядела как «концы соединились сами»
            path = [self._S(p) for p in e.get("path") or [g["nodes"][e["a"]], g["nodes"][e["b"]]]]
            col = (240, 210, 130, 150) if e["a"] in main else (240, 120, 110, 180)
            if len(path) > 1:
                d.line(path, fill=col, width=2, joint="curve")
        for i, n in enumerate(g["nodes"]):
            x, y = self._S(n)
            col = (255, 225, 150, 235) if i in main else (255, 120, 110, 240)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col, outline=(30, 28, 24, 255))
        self._graph_stats = (len(g["nodes"]), len(g["edges"]), len(comps))

    def _label(self, d, x, y, text, col=(255, 255, 255, 235)):
        d.rectangle([x - 3, y - 2, x + 8 + 7.2 * len(text), y + 16], fill=(20, 20, 24, 215))
        d.text((x + 2, y), text, fill=col, font=self.font)

    def _draw_ruler(self, d):
        """Линейка: расстояние между точками, а кругом — радиус с кольцами. На карте всё решают
        размеры (90 м леса перекрывают обзор, ближе 45 м укрытие не работает, рубеж держится в
        255 м), а на глаз при свободном зуме их не отличить."""
        a, b = self._ruler
        pa, pb = self._S(a), self._S(b)
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        col = (255, 255, 255, 230)
        if self.ruler_circle.get():
            r = dist * self.view.zoom
            for frac in (0.25, 0.5, 0.75, 1.0):
                rr = r * frac
                d.ellipse([pa[0] - rr, pa[1] - rr, pa[0] + rr, pa[1] + rr],
                          outline=(255, 255, 255, 220 if frac == 1.0 else 90),
                          width=2 if frac == 1.0 else 1)
                if frac in (0.5, 1.0) and rr > 24:
                    self._label(d, pa[0] + rr * 0.7, pa[1] - rr * 0.7 - 8,
                                f"{dist * frac:.0f} м")
        d.line([pa, pb], fill=col, width=2)
        for p in (pa, pb):
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=col)
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        self._label(d, mx + 8, my - 20, f"{dist:.0f} м  ({dist / P.M_PER_UNIT:.1f} ед)")

    def _vision_overlay(self):
        """Зона видимости поверх карты. Сглаживание тут не косметика: сетка 15 м на километровом
        радиусе даёт ступеньку в полклетки на границе тени, и вместо «за рощей не видно» глаз
        читает пиксельную кашу. Растягиваем билинейно и слегка размываем — форма тени остаётся,
        лесенка уходит."""
        center, field, cell = self._vision
        v = np.clip(field, 0.0, 1.0)
        rgba = np.zeros((*v.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255 - (243 * (1 - v)).astype(np.uint8)      # свет -> тёплый, тень -> синяя
        rgba[..., 1] = 226 - (212 * (1 - v)).astype(np.uint8)
        rgba[..., 2] = 150 - (124 * (1 - v)).astype(np.uint8)
        rgba[..., 3] = (46 + 120 * (1 - v)).astype(np.uint8)
        img = Image.fromarray(np.ascontiguousarray(rgba.transpose(1, 0, 2)[::-1]))
        k = 4                                                       # подрастим ДО поворота
        img = img.resize((img.width * k, img.height * k), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(k * 0.6))
        gy = v.shape[1] * k
        w2i = _translate(0, gy) @ _scale(k / cell, -k / cell)
        coeffs = (w2i @ self.view.inv(self.W, self.H))[:2].flatten()
        return img.transform((self.W, self.H), Image.AFFINE, tuple(coeffs),
                             resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

    def _draw_vision(self, d):
        """Кольца дальностей поверх зоны видимости: у нас автомат бьёт до 195 м, пулемёт до 405,
        достают все до 1005 — без колец «видно далеко» ничего не значит."""
        center, mask, cell = self._vision
        cx, cy = self._S(center)
        acc = (255, 226, 150)
        for r_m in (200, 400, 600, 800, 1000):
            if r_m > self.vision_r.get() + 1:
                break
            r = r_m * self.view.zoom
            if r < 14:
                continue
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*acc, 110 if r_m % 400 else 170))
            self._label(d, cx + r * 0.71 + 2, cy - r * 0.71 - 8, f"{r_m} м", (*acc, 235))
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 255, 255, 240))

    def _draw_hud(self, d):
        """Линейка и стрелка на север. Без линейки масштаб на глаз не определяется вовсе:
        при свободном зуме «этот лес большой» ничего не значит, а размеры тут решают всё —
        90 м леса перекрывают луч, 30 м нет."""
        acc = (224, 180, 92)
        # линейка: берём «круглую» длину около пятой части окна
        target = self.W * 0.22 / max(self.view.zoom, 1e-9)
        nice = min([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                   key=lambda v: abs(math.log10(max(v, 1e-9) / max(target, 1e-9))))
        px = nice * self.view.zoom
        x0, y0 = 18, self.H - 26
        d.line([(x0, y0), (x0 + px, y0)], fill=(*acc, 220), width=2)
        for x in (x0, x0 + px):
            d.line([(x, y0 - 6), (x, y0 + 4)], fill=(*acc, 220), width=2)
        label = f"{nice} м" if nice < 1000 else f"{nice / 1000:.0f} км"
        d.text((x0, y0 - 20), label, fill=(*acc, 235), font=self.font)

        # стрелка на север: при повёрнутом виде без неё легко потерять, где чей край
        cx, cy = self.W - 40, 44
        a = math.radians(self.view.angle)
        nx, ny = math.sin(a), -math.cos(a)                # мировой +Y на экране
        d.line([(cx - nx * 16, cy - ny * 16), (cx + nx * 16, cy + ny * 16)],
               fill=(*acc, 200), width=2)
        d.polygon([(cx + nx * 18, cy + ny * 18),
                   (cx + nx * 8 - ny * 6, cy + ny * 8 + nx * 6),
                   (cx + nx * 8 + ny * 6, cy + ny * 8 - nx * 6)], fill=(*acc, 230))
        d.text((cx - 4 + nx * 26, cy - 8 + ny * 26), "С", fill=(*acc, 235), font=self.font)

    def _draw_markers(self, d):
        for kind in P.MARKER_KINDS:
            col = P.MARKER_COLORS[kind]
            for i, (x_m, y_m) in enumerate(self.doc.markers[kind]):
                x, y = self._S((x_m, y_m))
                if kind == "zones":
                    r = P.ZONE_RADIUS * P.M_PER_UNIT * self.view.zoom   # рубеж удержания, как в бою
                    d.ellipse([x - r, y - r, x + r, y + r], outline=(*col, 90))
                    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(*col, 230))
                    d.text((x + 8, y - 7), f"об.{i + 1}", fill=(*col, 255), font=self.font_small)
                else:
                    d.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(*col, 230))
                    name = P.SLOT_NAMES[i] if i < len(P.SLOT_NAMES) else f"слот {i + 1}"
                    d.text((x + 7, y - 7), name, fill=(*col, 255), font=self.font_small)

    # --- попадание мышью

    def _world(self, ev):
        return self.view.to_world(self.W, self.H, ev.x, ev.y)

    def _tol_m(self, px=8.0):
        return px / max(self.view.zoom, 1e-9)

    def _hit_shape(self, p):
        """Что под курсором. Идём с конца: последние нарисованные лежат сверху."""
        tol = self._tol_m()
        for i in range(len(self.doc.shapes) - 1, -1, -1):
            sh = self.doc.shapes[i]
            if sh["kind"] == "polygon" and len(sh["points"]) >= 3 and _point_in_poly(p, sh["points"]):
                return i
            if sh["kind"] == "line" and _dist_to_polyline(p, sh["points"]) <= max(
                    sh.get("width_m", 8) / 2, tol):
                return i
            if sh["kind"] == "building":
                cx, cy, bw, bh, ang = sh["rect_m"]
                if _point_in_poly(p, vectormap._rect_points(cx, cy, bw, bh, ang)):
                    return i
            if sh["kind"] == "crossing":
                q = sh["point"]
                if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 <= max(20.0, tol) ** 2:
                    return i
        return None

    def _hit_node(self, p):
        if self.selected is None:
            return None
        sh = self.doc.shapes[self.selected]
        if sh["kind"] not in ("polygon", "line"):
            return None
        tol = self._tol_m(9)
        for k, q in enumerate(sh["points"]):
            if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 <= tol * tol:
                return k
        return None

    def _hit_marker(self, ev):
        for kind in P.MARKER_KINDS:
            for i, q in enumerate(self.doc.markers[kind]):
                x, y = self._S(q)
                if (x - ev.x) ** 2 + (y - ev.y) ** 2 <= 100:
                    return kind, i
        return None

    # --- мышь

    def _start_pan(self, ev):
        self._drag = ("pan", (ev.x, ev.y, self.view.cx, self.view.cy))

    def on_press(self, ev):
        if self._panning:
            return self._start_pan(ev)
        p = self._world(ev)
        if self.u_move.get() and self._current_underlay():
            u = self._current_underlay()
            self._drag = ("underlay", (p, u.cx, u.cy, u))
            return
        tool = self.tool.get()
        if tool == "ruler":
            self._ruler = [p, p]
            self._drag = ("ruler", None)
            self.draw()
            return
        if tool == "vision":
            # видимость живёт, пока держишь кнопку: это взгляд, а не объект карты, и оставаться
            # на экране поверх чертежа ему незачем
            self._drag = ("vision", None)
            self._update_vision(p)
            return
        if tool == "marker":
            hit = self._hit_marker(ev)
            self.push_undo()
            if hit:
                self._drag = ("marker", hit)
            else:
                self._place_marker(p)
            return
        if tool in ("polygon", "line"):
            self._draft = (self._draft or []) + [[p[0], p[1]]]
            self.draw()
            return
        if tool == "building":
            self.push_undo()
            w, h = self._house_size()
            self.doc.shapes.append({"kind": "building",
                                    "rect_m": [round(p[0], 1), round(p[1], 1), w, h, 0.0],
                                    "capacity": 1})
            self.selected = len(self.doc.shapes) - 1
            # протяжка только доворачивает: размер задан числами, и сбивать его движением мыши
            # незачем — дома на карте отличаются именно размером
            self._drag = ("building_rot", (p, self.selected))
            self.doc.bump()
            self.draw()
            return
        if tool == "crossing":
            hit = self._hit_shape(p)
            if hit is not None and self.doc.shapes[hit]["kind"] == "crossing":
                self.push_undo()                       # повторная протяжка доворачивает мост
                self.selected = hit
                self._drag = ("crossing_rot", (self.doc.shapes[hit]["point"], hit))
                return
            self.push_undo()
            sh = {"kind": "crossing", "point": [round(p[0], 1), round(p[1], 1)]}
            if not self.bridge_auto.get():
                # ручные размеры: длина — насколько мост перекрывает реку, ширина — проезд
                sh["length_m"] = float(self.bridge_len.get())
                sh["width_m_road"] = float(self.bridge_wid.get())
            self.doc.shapes.append(sh)
            self.selected = len(self.doc.shapes) - 1
            self._drag = ("crossing_rot", (p, self.selected))
            self.doc.bump()
            self.draw()
            return
        if tool == "nodes":
            k = self._hit_node(p)
            if k is not None:
                self.push_undo()
                self._drag = ("node", (self.selected, k))
                return
        hit = self._hit_shape(p)
        self.selected = hit
        if hit is not None and tool == "select":
            self.push_undo()
            self._drag = ("move", (p, copy.deepcopy(self.doc.shapes[hit])))
        self._refresh_shapes()
        self.draw()

    def on_motion(self, ev):
        self.on_hover(ev)
        if not self._drag:
            return
        mode, data = self._drag
        p = self._world(ev)
        if mode == "vision":
            self._update_vision(p)
        elif mode == "ruler":
            self._ruler[1] = p
            self.draw()
        elif mode == "pan":
            x0, y0, cx0, cy0 = data
            # тащим МИР под курсором: при повёрнутом виде «влево» на экране — не «влево» в мире
            self.view.cx, self.view.cy = cx0, cy0
            w0 = self.view.to_world(self.W, self.H, x0, y0)
            w1 = self.view.to_world(self.W, self.H, ev.x, ev.y)
            self.view.cx = cx0 + (w0[0] - w1[0])
            self.view.cy = cy0 + (w0[1] - w1[1])
            self.draw()
        elif mode == "underlay":
            p0, ux, uy, u = data
            u.cx, u.cy = ux + (p[0] - p0[0]), uy + (p[1] - p0[1])
            self.draw()
        elif mode == "marker":
            kind, i = data
            self.doc.markers[kind][i] = [p[0], p[1]]
            self.draw()
        elif mode == "node":
            si, k = data
            self.doc.shapes[si]["points"][k] = [round(p[0], 1), round(p[1], 1)]
            self.doc.bump()
            self.draw()
        elif mode == "move":
            p0, orig = data
            dx, dy = p[0] - p0[0], p[1] - p0[1]
            sh = self.doc.shapes[self.selected]
            if orig["kind"] in ("polygon", "line"):
                sh["points"] = [[round(q[0] + dx, 1), round(q[1] + dy, 1)] for q in orig["points"]]
            elif orig["kind"] == "building":
                r = list(orig["rect_m"])
                sh["rect_m"] = [round(r[0] + dx, 1), round(r[1] + dy, 1), r[2], r[3], r[4]]
            elif orig["kind"] == "crossing":
                sh["point"] = [round(orig["point"][0] + dx, 1), round(orig["point"][1] + dy, 1)]
            self.doc.bump()
            self.draw()
        elif mode == "crossing_rot":
            p0, si = data
            dx, dy = p[0] - p0[0], p[1] - p0[1]
            if math.hypot(dx, dy) > 1:
                self.doc.shapes[si]["angle_deg"] = round(math.degrees(math.atan2(dy, dx)), 1)
                self.doc.bump()
                self.draw()
        elif mode == "building_rot":
            p0, si = data
            dx, dy = p[0] - p0[0], p[1] - p0[1]
            if math.hypot(dx, dy) > 1:
                r = self.doc.shapes[si]["rect_m"]
                r[4] = round(math.degrees(math.atan2(dy, dx)), 1)
                self.doc.bump()
                self.draw()
        elif mode == "building":
            p0, si = data
            dx, dy = p[0] - p0[0], p[1] - p0[1]
            ln = math.hypot(dx, dy)
            if ln > 1:
                # тащим от центра: длина задаёт размер, направление — поворот дома вдоль улицы
                self.doc.shapes[si]["rect_m"] = [round(p0[0], 1), round(p0[1], 1),
                                                 round(max(ln * 2, 6.0), 1),
                                                 round(max(ln * 1.2, 4.0), 1),
                                                 round(math.degrees(math.atan2(dy, dx)), 1)]
                self.doc.bump()
                self.draw()

    def _update_vision(self, p):
        t0 = time.perf_counter()
        try:
            cell = float(self.cell_var.get())
        except (tk.TclError, ValueError, AttributeError):
            cell = self.doc.cell_m
        tm = self.doc.terrain_map(cell)
        field = viewshed(tm, p, float(self.vision_r.get()), P.M_PER_UNIT)
        self._vision = (p, field, cell)
        r_cells = float(self.vision_r.get()) / cell
        in_circle = max(1.0, math.pi * r_cells * r_cells)
        seen = float((field > 0.02).sum()) / in_circle
        self.status.config(text=f"просматривается {min(seen, 1.0) * 100:.0f}% круга радиусом "
                                f"{self.vision_r.get():.0f} м · счёт "
                                f"{(time.perf_counter() - t0) * 1000:.0f} мс")
        self.draw()

    def on_release(self, ev):
        if self._drag and self._drag[0] == "vision":
            self._vision = None
            self._drag = None
            self.draw()
            return
        if self._drag and self._drag[0] in ("move", "node", "building", "building_rot",
                                            "crossing_rot", "marker"):
            self._changed()
        self._drag = None
        self._refresh_points()

    def on_double(self, ev):
        if self._draft:
            self.finish_draft()

    def on_right(self, ev):
        p = self._world(ev)
        if self._draft:
            self.finish_draft()
            return
        if self.tool.get() == "marker":
            hit = self._hit_marker(ev)
            if hit:
                self.push_undo()
                self.doc.markers[hit[0]].pop(hit[1])
                self.draw()
                self._refresh_points()
            return
        if self.tool.get() == "nodes":
            k = self._hit_node(p)
            if k is not None:
                sh = self.doc.shapes[self.selected]
                if len(sh["points"]) > (3 if sh["kind"] == "polygon" else 2):
                    self.push_undo()
                    sh["points"].pop(k)
                    self._changed()
                else:
                    self.status.config(text="узлов и так минимум — удалите фигуру целиком")
                return
        hit = self._hit_shape(p)
        if hit is not None:
            self.selected = hit
            self._refresh_shapes()
            self.draw()

    def on_hover(self, ev):
        x, y = self._world(ev)
        W_m, H_m = self.doc.size_m
        where = "" if (0 <= x <= W_m and 0 <= y <= H_m) else "  вне карты"
        self.status.config(text=f"{x:7.0f} x {y:7.0f} м   "
                                f"({x / P.M_PER_UNIT:.1f}, {y / P.M_PER_UNIT:.1f} ед){where}")

    # --- правка

    def _changed(self):
        # Замер тут НЕ запускается. Раньше он шёл после каждой правки, и на большой карте это
        # выглядело как зависание: сетка 666x666 плюс сотни лучей на каждый штрих. Теперь — F5.
        self.doc.bump()
        self.draw()
        self._refresh_shapes()

    def push_undo(self):
        self.undo.append(self.doc.snapshot())
        del self.undo[:-60]
        self.redo.clear()

    def finish_draft(self):
        if not self._draft:
            return
        t = self.shape_type.get()
        pts = [[round(x, 1), round(y, 1)] for x, y in self._draft]
        tool = self.tool.get()
        if tool == "polygon" and len(pts) >= 3:
            self.push_undo()
            self.doc.shapes.append({"kind": "polygon", "type": t, "points": pts})
        elif tool == "line" and len(pts) >= 2:
            self.push_undo()
            sh = {"kind": "line", "type": t, "width_m": float(self.width_m.get()), "points": pts}
            if self.dash_on.get():
                sh["dash_m"] = [float(self.dash_len.get()), float(self.dash_gap.get())]
            self.doc.shapes.append(sh)
        else:
            self.status.config(text="точек мало: полигону нужно 3, линии 2")
            return
        self._draft = None
        self.selected = len(self.doc.shapes) - 1
        self._changed()

    def cancel_draft(self):
        self._draft = None
        self._ruler = None
        self._vision = None                 # Esc убирает и замеры, чтобы не мешали рисовать
        self.draw()

    def do_round(self):
        """Скруглить выбранную фигуру — срезание углов по Чайкину, по разу за нажатие.

        Рисуется всё щелчками, и углы выходят ломаные; у настоящей опушки и дороги их не бывает.
        Каждый угол заменяется двумя точками на четверти и трёх четвертях стороны: форма
        сохраняется, а изломы исчезают. У линии концы остаются на месте — иначе дорога отползала
        бы от перекрёстка, к которому её привязали."""
        if self.selected is None:
            return
        sh = self.doc.shapes[self.selected]
        if sh["kind"] not in ("polygon", "line") or len(sh["points"]) < 3:
            self.status.config(text="скруглять можно полигон или линию от трёх точек")
            return
        self.push_undo()
        pts = [tuple(p) for p in sh["points"]]
        closed = sh["kind"] == "polygon"
        src = pts + [pts[0]] if closed else pts
        out = [] if closed else [pts[0]]
        for a, b in zip(src[:-1], src[1:]):
            out.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            out.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        if not closed:
            out.append(pts[-1])
        if len(out) > 240:                    # дальше дробить бессмысленно: клетка всё съест
            self.status.config(text="узлов уже достаточно — дальше скруглять нечего")
            return
        sh["points"] = [[round(x, 1), round(y, 1)] for x, y in out]
        self._changed()

    def delete_selected(self):
        if self.selected is None:
            return
        self.push_undo()
        self.doc.shapes.pop(self.selected)
        self.selected = None
        self._changed()

    def do_undo(self):
        if self.undo:
            self.redo.append(self.doc.snapshot())
            self.doc.restore(self.undo.pop())
            self.selected = None
            self._changed()
            self._refresh_points()

    def do_redo(self):
        if self.redo:
            self.undo.append(self.doc.snapshot())
            self.doc.restore(self.redo.pop())
            self.selected = None
            self._changed()
            self._refresh_points()

    def do_mirror(self):
        """Зеркалим карту по Y — та же операция, которой test_mirror.py проверяет симметрию
        среды: нарисовал половину, отразил, и разница в исходе боя точно не от карты."""
        self.push_undo()
        H = self.doc.size_m[1]
        for sh in self.doc.shapes:
            if sh["kind"] in ("polygon", "line"):
                sh["points"] = [[x, round(H - y, 1)] for x, y in sh["points"]]
            elif sh["kind"] == "building":
                r = sh["rect_m"]
                sh["rect_m"] = [r[0], round(H - r[1], 1), r[2], r[3], -r[4]]
            elif sh["kind"] == "crossing":
                sh["point"] = [sh["point"][0], round(H - sh["point"][1], 1)]
        self._changed()

    def _refresh_shapes(self):
        if not hasattr(self, "shape_info"):
            return
        kinds = {}
        for s in self.doc.shapes:
            key = f"{s['kind']} {s.get('type', '')}".strip()
            kinds[key] = kinds.get(key, 0) + 1
        lines = [f"фигур всего: {len(self.doc.shapes)}", ""]
        lines += [f"  {k:<22}{v}" for k, v in sorted(kinds.items())]
        if self.show_graph.get():
            n, e, c = getattr(self, "_graph_stats", (0, 0, 0))
            lines += ["", f"дороги: {n} узлов, {e} участков"]
            if c > 1:
                lines.append(f"  сеть РАЗОРВАНА на {c} части — проверьте переправы")
        if self.selected is not None and self.selected < len(self.doc.shapes):
            sh = self.doc.shapes[self.selected]
            lines += ["", f"выбрана: {sh['kind']} {TYPE_RU.get(sh.get('type'), '')}"]
            if sh["kind"] in ("polygon", "line"):
                extra = f", ширина {sh.get('width_m', 0):.0f} м" if sh["kind"] == "line" else ""
                lines.append(f"  узлов {len(sh['points'])}{extra}")
            elif sh["kind"] == "building":
                r = sh["rect_m"]
                lines.append(f"  {r[2]:.0f}x{r[3]:.0f} м, угол {r[4]:.0f}°")
        self.shape_info.delete("1.0", "end")
        self.shape_info.insert("end", "\n".join(lines))

    # --- подложки

    def _current_underlay(self):
        if self.u_index is None or self.u_index >= len(self.doc.underlays):
            return None
        return self.doc.underlays[self.u_index]

    def add_underlay(self):
        path = filedialog.askopenfilename(
            title="подложка", filetypes=[("изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if not path:
            return
        img = Image.open(path).convert("RGBA")
        if max(img.size) > MAX_LAYER_PX:              # рисуем в метрах, лишние пиксели — память зря
            k = MAX_LAYER_PX / max(img.size)
            img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
        W_m, H_m = self.doc.size_m
        self.doc.underlays.append(Underlay(img, os.path.basename(path), (W_m / 2, H_m / 2),
                                           max(W_m / img.width, H_m / img.height), path=path))
        self.u_index = len(self.doc.underlays) - 1
        self._sync_layer()
        self.draw()

    def del_underlay(self):
        if self._current_underlay() is not None:
            self.doc.underlays.pop(self.u_index)
            self.u_index = None
            self.draw()

    def _base_scale(self, u):
        return max(self.doc.size_m[0] / u.image.width, self.doc.size_m[1] / u.image.height)

    def _sync_layer(self):
        u = self._current_underlay()
        if not u:
            return
        self._syncing = True
        self.u_visible.set(u.visible)
        self.u_opacity.set(u.opacity)
        self.u_angle.set(u.angle)
        self.u_scale.set(math.log2(max(u.m_per_px, 1e-9) / self._base_scale(u)))
        self._syncing = False

    def _apply_underlay(self):
        u = self._current_underlay()
        if not u or self._syncing:
            return
        u.visible = self.u_visible.get()
        u.opacity = float(self.u_opacity.get())
        u.angle = float(self.u_angle.get())
        u.m_per_px = self._base_scale(u) * (2.0 ** float(self.u_scale.get()))
        self.draw()

    # --- точки

    def _place_marker(self, p):
        kind = self.marker_kind.get()
        W_m, H_m = self.doc.size_m
        if not (0 <= p[0] <= W_m and 0 <= p[1] <= H_m):
            return
        if kind in ("friendly", "enemy") and len(self.doc.markers[kind]) >= P.N_SIDE:
            self.status.config(text=f"{P.MARKER_RU[kind]}: уже {P.N_SIDE} — столько же, сколько "
                                    "слотов в составе; лишние сценарий не примет")
            return
        self.doc.markers[kind].append([p[0], p[1]])
        self.draw()
        self._refresh_points()

    def clear_markers(self, kind):
        self.push_undo()
        self.doc.markers[kind] = []
        self.draw()
        self._refresh_points()

    def _scenario_problems(self):
        p = []
        for kind in ("friendly", "enemy"):
            got = len(self.doc.markers[kind])
            if got != P.N_SIDE:
                p.append(f"{P.MARKER_RU[kind]}: {got} из {P.N_SIDE}")
        if not self.doc.markers["zones"]:
            p.append("нет ни одного объекта захвата")
        return p

    def _refresh_points(self):
        if not hasattr(self, "points_info"):
            return
        lines = [f"состав стороны: {P.N_SIDE} слотов", ""]
        for kind in P.MARKER_KINDS:
            pts = self.doc.markers[kind]
            lines.append(f"{P.MARKER_RU[kind]}: {len(pts)}")
            for i, (x, y) in enumerate(pts):
                tag = (f"об.{i + 1}" if kind == "zones"
                       else (P.SLOT_NAMES[i] if i < len(P.SLOT_NAMES) else f"слот {i + 1}"))
                lines.append(f"   {tag:<10} {x / P.M_PER_UNIT:6.1f} {y / P.M_PER_UNIT:6.1f} ед")
        problems = self._scenario_problems()
        lines += ["", "сценарий готов" if not problems else "сценарий неполон:"]
        lines += ["  " + s for s in problems]
        self.points_info.delete("1.0", "end")
        self.points_info.insert("end", "\n".join(lines))

    # --- замер

    def schedule_measure(self, delay=500):
        if self._measure_job:
            self.after_cancel(self._measure_job)
        self._measure_job = self.after(delay, self.remeasure)

    def remeasure(self):
        """Замер по требованию (F5) — одной строкой в состоянии. Полная панель убрана из окна,
        сами числа никуда не делись: metrics хранится и доступен проверкам."""
        self._measure_job = None
        self.status.config(text="меряю…")
        self.update_idletasks()
        try:
            cell = float(self.cell_var.get())
        except (tk.TclError, ValueError):
            cell = self.doc.cell_m
        surface, _ = self.doc.surface(cell)
        m = measure(surface, cell, n_pairs=400)
        self.metrics = m
        self.status.config(
            text=f"сетка {surface.shape[0]}x{surface.shape[1]} по {cell:.0f} м · "
                 f"лес {m['frac'][1] * 100:.0f}% застр {m['frac'][2] * 100:.0f}% "
                 f"дор {m['frac'][4] * 100:.0f}% · видимость {m['vis'] * 100:.0f}% · "
                 f"строений {m['comps']} · "
                 + ("годна" if not m["bad"] else "вырождена: " + "; ".join(m["bad"])))

    # --- файлы

    def save_map(self):
        name = self.name_var.get().strip()
        if not name:
            return
        path = os.path.join(P.MAPS, name + ".vector.json")
        if os.path.exists(path) and not messagebox.askyesno(
                "перезаписать?", f"{name}.vector.json уже есть. Заменить?"):
            return
        cell = float(self.cell_var.get())
        self.doc.vec["markers"] = self.doc.markers          # точки живут в самом векторе
        vectormap.save(self.doc.vec, path)
        prefix, surface, meta = vectormap.build(path, cell)
        self.doc.name = name
        self.doc.cell_m = cell
        self.status.config(text=f"сохранено: {name}.vector.json; собраны поля "
                                f"{surface.shape[0]}x{surface.shape[1]} по {cell:.0f} м и граф "
                                f"({len(meta['graph']['nodes'])} узлов)")

    def save_preview(self):
        name = self.name_var.get().strip()
        out_dir = os.path.join(P.MAPS, "preview")
        os.makedirs(out_dir, exist_ok=True)
        surface, _ = self.doc.surface(float(self.cell_var.get()))
        lut = np.zeros((max(P.TILE_COLORS) + 1, 3), dtype=np.uint8)
        for k, c in P.TILE_COLORS.items():
            lut[k] = c
        img = Image.fromarray(np.ascontiguousarray(lut[surface].transpose(1, 0, 2)[::-1]))
        scale = max(1, int(1200 / max(img.size)))
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        img.save(os.path.join(out_dir, name + ".png"))
        self.status.config(text=f"превью: maps/preview/{name}.png")

    def add_to_pool(self):
        """Пул — это то, на чём учится модель, и вырожденная карта там дороже отсутствующей."""
        name = self.name_var.get().strip()
        if self.metrics is None:
            self.remeasure()
        if self.metrics["bad"] and not messagebox.askyesno(
                "карта вырождена", "Замер считает карту вырожденной:\n\n"
                + "\n".join(self.metrics["bad"]) + "\n\nВсё равно добавить в пул?"):
            return
        path = os.path.join(P.MAPS, "platoon_pool.json")
        with open(path, "r", encoding="utf-8") as f:
            pool = json.load(f)
        if name in pool["good"]:
            self.status.config(text=f"{name} уже в пуле")
            return
        pool["good"].append(name)
        pool.get("rejected", {}).pop(name, None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pool, f, ensure_ascii=False, indent=2)
        self.status.config(text=f"{name} добавлена в maps/platoon_pool.json ({len(pool['good'])} карт)")

    def save_scenario(self):
        """Сценарий в формате scenario.py: координаты в ИГРОВЫХ единицах 0..ARENA. Среда им
        переопределяет и позиции сторон, и рубежи, поэтому число позиций обязано совпасть с
        составом — иначе scenario.validate() скажет «не загрузится», но уже на замере."""
        problems = self._scenario_problems()
        if problems and not messagebox.askyesno(
                "сценарий неполон", "\n".join(problems) + "\n\nВсё равно сохранить?"):
            return
        name = self.name_var.get().strip()
        if not os.path.exists(os.path.join(P.MAPS, name + ".fields.npz")):
            messagebox.showwarning("карта не собрана",
                                   f"Сценарий сошлётся на maps/{name}, а поля не собраны. "
                                   "Сохраните карту сначала.")
            return
        os.makedirs(P.SCENARIOS, exist_ok=True)
        path = filedialog.asksaveasfilename(initialdir=P.SCENARIOS, defaultextension=".json",
                                            initialfile=name + ".json",
                                            filetypes=[("сценарий", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scenario_dict(name, self.doc.markers), f, ensure_ascii=False, indent=2)
        self.status.config(text=f"сценарий: {os.path.relpath(path, P.ROOT)}")


# ---------------------------------------------------------------- геометрия попадания


def _point_in_poly(p, pts):
    x, y = p
    inside = False
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        if (ay > y) != (by > y):
            xx = ax + (y - ay) * (bx - ax) / (by - ay)
            if xx > x:
                inside = not inside
    return inside


def _dist_to_polyline(p, pts):
    best = 1e18
    for a, b in zip(pts[:-1], pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln2 = dx * dx + dy * dy
        t = 0.0 if ln2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ln2))
        best = min(best, math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy)))
    return best


# ---------------------------------------------------------------- приложение


def scenario_dict(name, markers):
    conv = lambda pts: [[round(x / P.M_PER_UNIT, 2), round(y / P.M_PER_UNIT, 2)]   # noqa: E731
                        for x, y in pts]
    return {"name": name, "map": f"maps/{name}", "map_seed": 0,
            "zones": conv(markers["zones"]), "friendly": conv(markers["friendly"]),
            "enemy": conv(markers["enemy"])}


def _glob_vectors():
    import glob as _g
    return (_g.glob(os.path.join(P.MAPS, "*.vector.json"))
            + _g.glob(os.path.join(P.MAPS, "crops", "*.vector.json")))


def load_doc(path):
    vec = vectormap.load(path)
    name = os.path.basename(path)[:-len(".vector.json")]
    cell = P.CELL_M
    meta_path = path[:-len(".vector.json")] + ".map.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cell = float(json.load(f).get("cell_m", P.CELL_M))
    return Doc(vec, cell, name)


class App:
    def __init__(self, root, doc=None):
        self.root = root
        apply_style(root)
        root.title("WarGame — редактор карт")
        root.geometry("1400x880")
        self.frame = None
        self.show_start() if doc is None else self.show_editor(doc)

    def show_start(self):
        self._swap(StartScreen(self.root, self.show_editor))
        self.root.title("WarGame — редактор карт")

    def show_editor(self, doc):
        self._swap(EditorFrame(self.root, doc, self.show_start))
        self.root.title(f"WarGame — редактор карт — {doc.name}")

    def _swap(self, frame):
        if self.frame is not None:
            self.frame.destroy()
        self.frame = frame
        frame.pack(fill="both", expand=True)


def main():
    ap = argparse.ArgumentParser(description="редактор карт WarGame")
    ap.add_argument("path", nargs="?", help="имя карты в maps/ или путь к <имя>.vector.json")
    args = ap.parse_args()

    doc = None
    if args.path:
        p = args.path if args.path.endswith(".vector.json") else \
            os.path.join(P.MAPS, args.path + ".vector.json")
        doc = load_doc(p)

    print(P.describe())
    root = tk.Tk()
    App(root, doc)
    root.mainloop()


if __name__ == "__main__":
    main()
