"""Редактор карт WarGame: слои, обводка настоящих карт по подложке, расстановка игровых точек.

Что это и зачем. Вырезка (maps/crop_platoon.py) берёт, что есть на авторской карте, и годность
выясняется постфактум. Редактор нужен, когда местность нужна ЗАДАННАЯ: проверить приказ на
конкретной конфигурации (опушка против села, дорога вдоль фронта, вода как непроходимая ось)
или обвести реальный участок по снимку.

Четыре вещи, вокруг которых собран интерфейс:

  * СЛОИ. Как в фотошопе, но клетками. Слой местности хранит тайлы, и у клетки есть состояние
    «пусто» — сквозь него виден слой ниже. Поэтому лес, дороги и застройку можно держать
    порознь, гасить и переставлять, а карта для игры собирается сведением стопки снизу вверх.
    Отдельный вид слоя — подложка-картинка со своей прозрачностью, поворотом и масштабом.
    Сохраняется то, ЧТО ВИДНО: скрытый слой в игровую карту не идёт (строка состояния об этом
    предупреждает), а сама стопка живёт рядом с картой в maps/<имя>.editor.npz.

  * ТОЧКИ. Объекты захвата и позиции сторон — не украшение: они пишутся в scenarios/*.json в
    формате scenario.py (игровые единицы 0..ARENA), который среда принимает и которым
    переопределяет и рубежи, и расстановку. Число позиций на сторону проверяется по составу
    из units.json, иначе сценарий молча не загрузится в модель.

  * ВИД. Зум, панорама и поворот на любой угол — независимо от сетки: обводить удобно, когда
    снимок лежит ровно, а не когда его угол совпал с осями карты.

  * ЗАМЕР. Карта на глаз и карта по числам — разные вещи: сплошной лес даёт 3% видимости и бой
    сводится к натыканию в упор, голая степь — 93% и перестрелку без манёвра (docs/JOURNAL.md,
    п. 3.3). Панель замера считает то же, по чему отбирался пул, и говорит «годна / вырождена»
    пока рисуешь.

Грабли формата, на которых уже горели (см. wargame_env.py и terrain.py):
  * сетка хранится как (Gx, Gy) и индексируется [gx, gy], а НЕ как изображение [row, col];
  * y растёт ВВЕРХ (свои внизу поля, враги вверху) — на экране это переворот;
  * в json пишутся РЕАЛЬНЫЕ метры (cell_m), среда делит их на m_per_unit, и пороги
    прозрачности сравниваются в ИГРОВЫХ единицах.

Запуск из корня проекта (нужны numpy, Pillow, tkinter; обучающих зависимостей нет):
    py -3.12 editor/map_editor.py
    py -3.12 editor/map_editor.py platoon_crop_3      # сразу открыть карту из maps/
"""
import argparse
import copy
import json
import math
import os
import sys

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
    from PIL import Image, ImageDraw, ImageTk
except ImportError as e:                                    # noqa: BLE001
    print(f"нужны tkinter и Pillow: {e}")
    raise SystemExit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P                                          # noqa: E402
from measure import measure                                  # noqa: E402
import terrain                                               # noqa: E402

BG = (16, 17, 20)
EMPTY = -1                   # «пусто» в слое местности: сквозь клетку виден слой ниже
MAX_LAYER_PX = 3000          # подложку крупнее ужимаем при импорте: рисуем мы всё равно в клетках


# ---------------------------------------------------------------- геометрия


def _m(a, b, c, d, e, f):
    return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]], dtype=np.float64)


def _translate(tx, ty):
    return _m(1, 0, tx, 0, 1, ty)


def _scale(sx, sy):
    return _m(sx, 0, 0, 0, sy, 0)


def _rotate(deg):
    t = math.radians(deg)
    return _m(math.cos(t), -math.sin(t), 0, math.sin(t), math.cos(t), 0)


class View:
    """Мир (клетки, y вверх) -> экран (пиксели, y вниз). Матрицей, а не формулами: поворот вида,
    поворот подложки и обратное преобразование под курсором — одна и та же арифметика, и
    расписанная руками она разъезжается (в проекте это уже стоило перепутанной оси Y)."""

    def __init__(self, zoom=8.0):
        self.cx = self.cy = 0.0
        self.zoom = zoom
        self.angle = 0.0

    def matrix(self, w, h):
        return (_translate(w / 2, h / 2) @ _scale(self.zoom, -self.zoom)
                @ _rotate(self.angle) @ _translate(-self.cx, -self.cy))

    def inv(self, w, h):
        return np.linalg.inv(self.matrix(w, h))

    def to_screen(self, w, h, fx, fy):
        p = self.matrix(w, h) @ np.array([fx, fy, 1.0])
        return float(p[0]), float(p[1])

    def to_world(self, w, h, sx, sy):
        p = self.inv(w, h) @ np.array([sx, sy, 1.0])
        return float(p[0]), float(p[1])

    def fit(self, w, h, Gx, Gy):
        if w < 2 or h < 2:
            return
        self.angle = 0.0
        self.zoom = min(w / (Gx + 2), h / (Gy + 2))
        self.cx, self.cy = Gx / 2.0, Gy / 2.0


# ---------------------------------------------------------------- слои


class Layer:
    kind = "?"

    def __init__(self, name, visible=True, opacity=1.0, locked=False):
        self.name = name
        self.visible = visible
        self.opacity = float(opacity)
        self.locked = bool(locked)
        self.ver = 0            # счётчик правок — по нему инвалидируется кэш картинки слоя

    def label(self):
        flags = ("●" if self.visible else "○") + ("з" if self.locked else " ")
        return f"{flags} {self.name[:22]:<22} {int(self.opacity * 100):3d}%"


class TileLayer(Layer):
    """Слой местности: та же сетка (Gx, Gy), но с EMPTY там, где слой ничего не кладёт."""

    kind = "tiles"

    def __init__(self, grid, name, **kw):
        super().__init__(name, **kw)
        self.grid = grid.astype(np.int8)

    def clone(self, name=None):
        return TileLayer(self.grid.copy(), name or (self.name + " копия"),
                         visible=self.visible, opacity=self.opacity, locked=self.locked)


class ImageLayer(Layer):
    """Подложка: картинка, положенная в мир. scale — клеток на пиксель картинки, angle —
    поворот против часовой, (cx, cy) — куда в мире смотрит середина картинки."""

    kind = "image"

    def __init__(self, image, name, center, scale, angle=0.0, opacity=0.6, path=None, **kw):
        super().__init__(name, opacity=opacity, **kw)
        self.image = image
        self.path = path
        self.cx, self.cy = center
        self.scale = float(scale)
        self.angle = float(angle)

    def clone(self, name=None):
        return ImageLayer(self.image, name or (self.name + " копия"), (self.cx, self.cy),
                          self.scale, self.angle, self.opacity, self.path,
                          visible=self.visible, locked=self.locked)

    def world_to_image(self):
        w, h = self.image.size
        return (_translate(w / 2, h / 2) @ _scale(1 / self.scale, -1 / self.scale)
                @ _rotate(-self.angle) @ _translate(-self.cx, -self.cy))


class Doc:
    """Карта в работе: стопка слоёв (снизу вверх), маркеры, размер клетки."""

    def __init__(self, shape, cell_m, name, layers=None):
        self.Gx, self.Gy = shape
        self.cell_m = float(cell_m)
        self.name = name
        self.layers = layers if layers is not None else [
            TileLayer(np.zeros(shape, dtype=np.int8), "основа")]
        self.active = len(self.layers) - 1
        self.markers = {k: [] for k in P.MARKER_KINDS}
        self.version = 0
        self._comp = None
        self._comp_ver = -1

    @property
    def shape(self):
        return (self.Gx, self.Gy)

    def bump(self):
        self.version += 1

    def tile_layers(self):
        return [L for L in self.layers if L.kind == "tiles"]

    def composite(self):
        """Карта, какой её увидит игра: ВИДИМЫЕ слои местности сводятся снизу вверх, EMPTY
        пропускает то, что лежит ниже. Основа — «открытое», иначе дыра в нижнем слое означала
        бы неопределённый тайл, а среда такого не принимает."""
        if self._comp is None or self._comp_ver != self.version:
            out = np.zeros(self.shape, dtype=np.int8)
            for L in self.layers:
                if L.kind == "tiles" and L.visible:
                    m = L.grid >= 0
                    out[m] = L.grid[m]
            self._comp, self._comp_ver = out, self.version
        return self._comp

    def hidden_tile_layers(self):
        return [L for L in self.layers if L.kind == "tiles" and not L.visible]

    def active_layer(self):
        return self.layers[self.active] if 0 <= self.active < len(self.layers) else None

    # снимок для отмены: сетки копируем, пиксели подложек — нет (они не правятся)
    def snapshot(self):
        return ([L.clone(L.name) for L in self.layers], self.active,
                copy.deepcopy(self.markers))

    def restore(self, snap):
        self.layers, self.active, self.markers = snap[0], snap[1], copy.deepcopy(snap[2])
        self.bump()


# ---------------------------------------------------------------- стартовый экран


class StartScreen(ttk.Frame):
    """Холста при запуске нет: карта либо создаётся с явно выбранным размером, либо открывается.
    Пустая сетка «по умолчанию» — это скрытое решение за пользователя, а размер здесь не
    косметика: он должен совпасть с ареной, иначе карта покроет её угол."""

    def __init__(self, master, on_open):
        super().__init__(master, padding=24)
        self.on_open = on_open
        ttk.Label(self, text="WarGame — редактор карт", font=("Segoe UI", 16)).pack(anchor="w")
        ttk.Label(self, text=P.describe(), foreground="#888").pack(anchor="w", pady=(2, 16))

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Button(row, text="Создать новую карту…", command=self.new_map).pack(side="left")
        ttk.Button(row, text="Открыть карту…", command=self.open_dialog).pack(side="left", padx=6)

        ttk.Label(self, text="карты в maps/", foreground="#888").pack(anchor="w", pady=(18, 2))
        box = ttk.Frame(self)
        box.pack(fill="both", expand=True)
        self.lst = tk.Listbox(box, height=14, width=70, activestyle="none")
        self.lst.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, command=self.lst.yview)
        sb.pack(side="left", fill="y")
        self.lst.config(yscrollcommand=sb.set)
        self.lst.bind("<Double-Button-1>", lambda e: self.open_selected())
        ttk.Button(self, text="Открыть выбранную", command=self.open_selected).pack(anchor="w", pady=8)

        self.files = []
        for path in sorted(_glob_maps()):
            name = os.path.basename(path)[:-4]
            try:
                g = np.load(path, mmap_mode="r")
                cell_m = _meta_of(path).get("cell_m", P.CELL_M)
                span = g.shape[0] * cell_m
                mark = "" if abs(span - P.ARENA_M) < P.ARENA_M * 0.05 else "   ← другой масштаб"
                layers = "  · со слоями" if os.path.exists(path[:-4] + ".editor.npz") else ""
                self.lst.insert("end", f"{name:<22} {g.shape[0]}x{g.shape[1]} по {cell_m:.0f} м "
                                       f"= {span:.0f} м{mark}{layers}")
                self.files.append(path)
            except Exception as exc:                          # noqa: BLE001
                self.lst.insert("end", f"{name}  — не читается: {exc}")
                self.files.append(None)

    def new_map(self):
        NewMapDialog(self, self._create)

    def _create(self, gx, gy, cell_m):
        self.on_open(Doc((gx, gy), cell_m, "new_map"))

    def open_dialog(self):
        path = filedialog.askopenfilename(initialdir=P.MAPS, filetypes=[("сетка карты", "*.npy")])
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

        self.gx = tk.IntVar(value=P.GRID_N)
        self.gy = tk.IntVar(value=P.GRID_N)
        self.cell = tk.DoubleVar(value=P.CELL_M)
        for i, (label, var) in enumerate((("клеток по X", self.gx), ("клеток по Y", self.gy),
                                          ("метров в клетке", self.cell))):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=var, width=10).grid(row=i, column=1, sticky="w", padx=6)
            var.trace_add("write", lambda *_: self.refresh())

        self.info = ttk.Label(f, text="", justify="left")
        self.info.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 6))
        ttk.Button(f, text=f"под текущий масштаб ({P.GRID_N}x{P.GRID_N} по {P.CELL_M:.0f} м)",
                   command=self.preset).grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Button(f, text="создать", command=self.ok).grid(row=5, column=0, columnspan=2,
                                                            sticky="ew", pady=(8, 0))
        self.refresh()
        self.grab_set()

    def preset(self):
        self.gx.set(P.GRID_N)
        self.gy.set(P.GRID_N)
        self.cell.set(P.CELL_M)

    def _values(self):
        try:
            return int(self.gx.get()), int(self.gy.get()), float(self.cell.get())
        except (tk.TclError, ValueError):
            return None

    def refresh(self):
        v = self._values()
        if not v or min(v[0], v[1]) < 4 or v[2] <= 0:
            self.info.config(text="размеры не разобраны", foreground="#c66")
            return
        gx, gy, cell = v
        span_x, span_y = gx * cell, gy * cell
        ok = (abs(span_x - P.ARENA_M) < P.ARENA_M * 0.05
              and abs(span_y - P.ARENA_M) < P.ARENA_M * 0.05)
        # Сетка обязана покрыть ровно арену: среда делит cell_m на m_per_unit и ждёт этого.
        # Иначе бой пойдёт на углу карты — env про это предупреждает при загрузке.
        self.info.config(
            text=f"поле {span_x:.0f} x {span_y:.0f} м, клетка {cell / P.M_PER_UNIT:.2f} ед\n"
                 + ("совпадает с ареной" if ok else
                    f"НЕ совпадает с ареной {P.ARENA_M:.0f} x {P.ARENA_M:.0f} м — "
                    "среда возьмёт лишь угол карты"),
            foreground="#8b8" if ok else "#c96")

    def ok(self):
        v = self._values()
        if not v:
            return
        self.destroy()
        self.on_ok(*v)


# ---------------------------------------------------------------- сам редактор


class EditorFrame(ttk.Frame):
    TOOLS = [("кисть", "brush"), ("линия", "line"), ("прямоугольник", "rect"),
             ("заливка", "fill"), ("точка", "marker")]

    def __init__(self, master, doc, on_close):
        super().__init__(master)
        self.doc = doc
        self.on_close = on_close
        self.view = View()
        self.undo, self.redo = [], []
        self.tile = tk.IntVar(value=1)
        self.tool = tk.StringVar(value="brush")
        self.brush = tk.IntVar(value=2)
        self.marker_kind = tk.StringVar(value="zones")
        self.metrics = None

        self._layer_cache = {}           # id(слой) -> (версия, RGBA-картинка)
        self._preview = None             # (слой, сетка) на время протяжки линии/прямоугольника
        self._drag = None
        self._measure_job = None
        self._panning = False
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
        self.canvas.bind("<ButtonPress-3>", self.on_right)
        self.canvas.bind("<B3-Motion>", self.on_right_motion)
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        side = ttk.Frame(self, width=348)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        nb = ttk.Notebook(side)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._tab_draw(nb)
        self._tab_layers(nb)
        self._tab_points(nb)
        self._tab_measure(nb)
        self._tab_file(nb)

        self.status = ttk.Label(side, text="", anchor="w", wraplength=330, justify="left")
        self.status.pack(fill="x", padx=8, pady=(0, 6))

        self._bound = []
        for seq, fn in (
                *[(str(i + 1), lambda t=i: self.tile.set(t)) for i in P.TILE_NAMES],
                *[(k, lambda v=v: self.tool.set(v))
                  for k, v in zip("qwety", ("brush", "line", "rect", "fill", "marker"))],
                ("<Control-z>", self.do_undo), ("<Control-y>", self.do_redo),
                ("<Control-s>", self.save_map), ("<F5>", self.remeasure),
                ("<plus>", lambda: self.zoom_by(1.25)), ("<minus>", lambda: self.zoom_by(0.8)),
                ("0", self.fit_view),
                ("<bracketleft>", lambda: self.brush.set(max(1, self.brush.get() - 1))),
                ("<bracketright>", lambda: self.brush.set(min(12, self.brush.get() + 1))),
                ("<KeyPress-space>", lambda: setattr(self, "_panning", True)),
                ("<KeyRelease-space>", lambda: setattr(self, "_panning", False))):
            self._bind_hotkey(seq, fn)

    def _bind_hotkey(self, seq, fn):
        """Горячая клавиша не должна срабатывать, пока набирают имя карты: «1» в поле ввода —
        это единица, а не переключение тайла."""
        def handler(ev, fn=fn):
            if isinstance(self.focus_get(), (tk.Entry, ttk.Entry, tk.Text, tk.Listbox)):
                return None
            fn()
            return "break"

        self.winfo_toplevel().bind(seq, handler)
        self._bound.append(seq)

    def destroy(self):
        """Снять горячие клавиши: они висят на окне, а не на этом кадре, и после возврата на
        стартовый экран били бы по уничтоженным виджетам."""
        top = self.winfo_toplevel()
        for seq in getattr(self, "_bound", []):
            top.unbind(seq)
        super().destroy()

    def _tab_draw(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="рисование")
        ttk.Label(f, text="тайл  [1-5]").pack(anchor="w")
        for tid, ru in P.TILE_NAMES.items():
            row = ttk.Frame(f)
            row.pack(anchor="w", fill="x")
            sw = tk.Canvas(row, width=15, height=15, highlightthickness=0)
            sw.create_rectangle(0, 0, 15, 15, fill="#%02x%02x%02x" % P.TILE_COLORS[tid], outline="")
            sw.pack(side="left", padx=(0, 5))
            ttk.Radiobutton(row, text=ru, value=tid, variable=self.tile).pack(side="left")
        ttk.Label(f, text="ПКМ — ластик: стирает в «пусто»,\nоткрывая слой ниже",
                  foreground="#888", justify="left").pack(anchor="w", pady=(4, 0))

        ttk.Separator(f).pack(fill="x", pady=6)
        ttk.Label(f, text="инструмент  [q w e t y]").pack(anchor="w")
        for label, val in self.TOOLS:
            ttk.Radiobutton(f, text=label, value=val, variable=self.tool).pack(anchor="w")
        ttk.Label(f, text="кисть, клеток  [ ]").pack(anchor="w", pady=(6, 0))
        ttk.Scale(f, from_=1, to=12, variable=self.brush, orient="horizontal",
                  command=lambda v: self.brush.set(int(float(v)))).pack(fill="x")

        ttk.Separator(f).pack(fill="x", pady=6)
        for label, cmd in (("отменить  Ctrl+Z", self.do_undo), ("вернуть  Ctrl+Y", self.do_redo),
                           ("зеркалить слой по Y", self.do_mirror),
                           ("повернуть карту на 90°", self.do_rot90),
                           ("очистить слой", self.do_clear)):
            ttk.Button(f, text=label, command=cmd).pack(fill="x", pady=1)

        ttk.Separator(f).pack(fill="x", pady=6)
        ttk.Label(f, text="вид: колесо — зум, СКМ или пробел — тащить").pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=2)
        for label, cmd in (("−", lambda: self.zoom_by(0.8)), ("+", lambda: self.zoom_by(1.25)),
                           ("вписать", self.fit_view), ("↺ 90°", lambda: self.rotate_view(90)),
                           ("↻ 90°", lambda: self.rotate_view(-90))):
            ttk.Button(row, text=label, width=7, command=cmd).pack(side="left", padx=1)
        ttk.Label(f, text="поворот вида, °").pack(anchor="w", pady=(6, 0))
        self.view_angle = tk.DoubleVar(value=0.0)
        ttk.Scale(f, from_=-180, to=180, variable=self.view_angle, orient="horizontal",
                  command=lambda v: self._set_view_angle(float(v))).pack(fill="x")

    def _tab_layers(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="слои")
        ttk.Label(f, text="сверху — верхний слой; рисование идёт в выбранный",
                  foreground="#888", wraplength=320).pack(anchor="w")
        self.layer_list = tk.Listbox(f, height=8, activestyle="none", exportselection=False,
                                     font=("Consolas", 9))
        self.layer_list.pack(fill="x", pady=4)
        self.layer_list.bind("<<ListboxSelect>>", lambda e: self._select_from_list())
        self.layer_list.bind("<Double-Button-1>", lambda e: self.rename_layer())

        for row_defs in (
                (("+ местность", self.add_tile_layer), ("+ подложка…", self.add_image_layer),
                 ("дублировать", self.duplicate_layer), ("удалить", self.delete_layer)),
                (("вверх", lambda: self.move_layer(+1)), ("вниз", lambda: self.move_layer(-1)),
                 ("объединить вниз", self.merge_down), ("свести всё", self.flatten))):
            row = ttk.Frame(f)
            row.pack(fill="x", pady=1)
            for label, cmd in row_defs:
                ttk.Button(row, text=label, width=14, command=cmd).pack(side="left", padx=1)
        ttk.Button(f, text="переименовать…", command=self.rename_layer).pack(fill="x", pady=(1, 4))

        line = ttk.Frame(f)
        line.pack(fill="x")
        self.layer_visible = tk.BooleanVar(value=True)
        self.layer_locked = tk.BooleanVar(value=False)
        ttk.Checkbutton(line, text="видим", variable=self.layer_visible,
                        command=self._apply_layer).pack(side="left")
        ttk.Checkbutton(line, text="замок", variable=self.layer_locked,
                        command=self._apply_layer).pack(side="left", padx=10)

        self.layer_opacity = tk.DoubleVar(value=1.0)
        ttk.Label(f, text="прозрачность слоя").pack(anchor="w", pady=(6, 0))
        ttk.Scale(f, from_=0.05, to=1.0, variable=self.layer_opacity, orient="horizontal",
                  command=lambda v: self._apply_layer()).pack(fill="x")

        self.img_box = ttk.LabelFrame(f, text="подложка", padding=6)
        self.img_box.pack(fill="x", pady=8)
        self.layer_move = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.img_box, text="двигать мышью (вместо рисования)",
                        variable=self.layer_move).pack(anchor="w")
        self.layer_angle = tk.DoubleVar(value=0.0)
        self.layer_scale = tk.DoubleVar(value=0.0)          # логарифмический: 2**v от «вписать»
        for label, var, lo, hi in (("поворот, °", self.layer_angle, -180, 180),
                                   ("масштаб (log2)", self.layer_scale, -4, 4)):
            ttk.Label(self.img_box, text=label).pack(anchor="w", pady=(4, 0))
            ttk.Scale(self.img_box, from_=lo, to=hi, variable=var, orient="horizontal",
                      command=lambda v: self._apply_layer()).pack(fill="x")
        ttk.Button(self.img_box, text="вписать в карту", command=self.fit_layer).pack(fill="x", pady=4)

    def _tab_points(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="точки")
        ttk.Label(f, text="инструмент «точка»: ЛКМ — поставить или тащить,\nПКМ — убрать",
                  justify="left").pack(anchor="w")
        for kind in P.MARKER_KINDS:
            ttk.Radiobutton(f, text=P.MARKER_RU[kind], value=kind,
                            variable=self.marker_kind).pack(anchor="w", pady=(4, 0))
        self.points_info = tk.Text(f, height=12, width=36, font=("Consolas", 9), relief="flat",
                                   background="#1e1e1e", foreground="#ddd")
        self.points_info.pack(pady=6)
        for kind in P.MARKER_KINDS:
            ttk.Button(f, text=f"очистить: {P.MARKER_RU[kind]}",
                       command=lambda k=kind: self.clear_markers(k)).pack(fill="x", pady=1)
        ttk.Button(f, text="сохранить сценарий…", command=self.save_scenario).pack(fill="x", pady=(8, 0))

    def _tab_measure(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="замер")
        ttk.Label(f, text="меряется СВЕДЁННАЯ карта — то, что видно",
                  foreground="#888").pack(anchor="w")
        self.info = tk.Text(f, height=21, width=36, font=("Consolas", 9), relief="flat",
                            background="#1e1e1e", foreground="#ddd")
        self.info.pack(pady=4)
        ttk.Button(f, text="перемерить  F5", command=self.remeasure).pack(fill="x", pady=4)

    def _tab_file(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text="файл")
        self.name_var = tk.StringVar(value=self.doc.name)
        ttk.Label(f, text="имя карты (пишется в maps/)").pack(anchor="w")
        ttk.Entry(f, textvariable=self.name_var).pack(fill="x", pady=2)
        for label, cmd in (("сохранить карту  Ctrl+S", self.save_map),
                           ("сохранить превью PNG", self.save_preview),
                           ("добавить в пул обучения", self.add_to_pool),
                           ("сохранить сценарий…", self.save_scenario)):
            ttk.Button(f, text=label, command=cmd).pack(fill="x", pady=1)
        ttk.Label(f, text="карта для игры — это СВЕДЁННАЯ стопка: скрытые слои в неё не идут.\n"
                          "Сама стопка ложится рядом в maps/<имя>.editor.npz.",
                  foreground="#888", wraplength=310, justify="left").pack(anchor="w", pady=6)
        ttk.Separator(f).pack(fill="x", pady=8)
        ttk.Button(f, text="закрыть карту", command=lambda: self.on_close()).pack(fill="x")
        ttk.Label(f, text=P.describe(), wraplength=310, foreground="#888").pack(anchor="w", pady=8)

    # --- вид

    def _first_open(self):
        self.fit_view()
        self._refresh_layers(select_model=self.doc.active)
        self._refresh_points()
        self.remeasure()

    def on_resize(self, ev):
        self.W, self.H = max(2, ev.width), max(2, ev.height)
        self.draw()

    def fit_view(self):
        self.view.fit(self.W, self.H, *self.doc.shape)
        self.view_angle.set(0.0)
        self.draw()

    def zoom_by(self, k, anchor=None):
        ax, ay = anchor if anchor else (self.W / 2, self.H / 2)
        before = self.view.to_world(self.W, self.H, ax, ay)
        self.view.zoom = float(np.clip(self.view.zoom * k, 0.5, 80.0))
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

    @staticmethod
    def _render_tiles(grid):
        """Картинка слоя местности: EMPTY -> прозрачно (сквозь него виден слой ниже)."""
        lut = np.zeros((max(P.TILE_COLORS) + 1, 4), dtype=np.uint8)
        for k, c in P.TILE_COLORS.items():
            lut[k] = (*c, 255)
        rgba = lut[np.clip(grid, 0, max(P.TILE_COLORS))]
        rgba[..., 3] = np.where(grid < 0, 0, 255)
        # (Gx,Gy) -> картинка: строка = gy, потом переворот, чтобы верх = большой Y
        return Image.fromarray(np.ascontiguousarray(rgba.transpose(1, 0, 2)[::-1]))

    def _tile_rgba(self, L):
        # Протяжка линии/прямоугольника рисуется МИМО кэша: сетка предпросмотра меняется на
        # каждое движение мыши, а версия слоя при этом не растёт — кэш отдавал бы первый кадр.
        if self._preview is not None and self._preview[0] is L:
            return self._render_tiles(self._preview[1])
        cached = self._layer_cache.get(id(L))
        if cached is None or cached[0] != L.ver:
            self._layer_cache[id(L)] = (L.ver, self._render_tiles(L.grid))
        return self._layer_cache[id(L)][1]

    def _warp(self, img, world_to_img, resample):
        """Кладём картинку из её собственных координат в экранные одним аффинным преобразованием.
        Стоимость зависит от размера окна, а не картинки, — поэтому подложка 3000x3000 и зум 40
        не тормозят."""
        coeffs = (world_to_img @ self.view.inv(self.W, self.H))[:2].flatten()
        return img.transform((self.W, self.H), Image.AFFINE, tuple(coeffs),
                             resample=resample, fillcolor=(0, 0, 0, 0))

    def draw(self):
        if self.W < 4 or self.H < 4:
            return
        Gx, Gy = self.doc.shape
        out = Image.new("RGB", (self.W, self.H), BG)

        for L in self.doc.layers:                     # снизу вверх
            if not L.visible:
                continue
            if L.kind == "image":
                warped = self._warp(L.image, L.world_to_image(), Image.BILINEAR)
            else:
                warped = self._warp(self._tile_rgba(L), _translate(0, Gy) @ _scale(1, -1),
                                    Image.NEAREST)
            a = warped.getchannel("A").point(lambda v, o=L.opacity: int(v * o))
            out.paste(warped.convert("RGB"), (0, 0), a)

        over = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        S = lambda fx, fy: self.view.to_screen(self.W, self.H, fx, fy)   # noqa: E731

        # полосы развёртывания: свои внизу, враги вверху (reset() в wargame_env). Карта, у
        # которой всё укрытие на одной половине, несправедлива, а на глаз это не видно.
        band = 300.0 / self.doc.cell_m
        d.polygon([S(0, 0), S(Gx, 0), S(Gx, band), S(0, band)], fill=(80, 140, 255, 30))
        d.polygon([S(0, Gy - band), S(Gx, Gy - band), S(Gx, Gy), S(0, Gy)], fill=(255, 90, 90, 30))

        step = max(1.0, round(300.0 / self.doc.cell_m))
        if self.view.zoom * step > 12:
            i = 0.0
            while i <= max(Gx, Gy):
                if i <= Gx:
                    d.line([S(i, 0), S(i, Gy)], fill=(255, 255, 255, 26))
                if i <= Gy:
                    d.line([S(0, i), S(Gx, i)], fill=(255, 255, 255, 26))
                i += step
        d.line([S(0, 0), S(Gx, 0), S(Gx, Gy), S(0, Gy), S(0, 0)], fill=(210, 210, 210, 120), width=2)

        self._draw_markers(d, S)
        out = Image.alpha_composite(out.convert("RGBA"), over).convert("RGB")

        self._photo = ImageTk.PhotoImage(out)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _draw_markers(self, d, S):
        for kind in P.MARKER_KINDS:
            col = P.MARKER_COLORS[kind]
            for i, (fx, fy) in enumerate(self.doc.markers[kind]):
                x, y = S(fx, fy)
                if kind == "zones":
                    r = P.ZONE_RADIUS / P.CELL_UNITS * self.view.zoom   # рубеж удержания, как в бою
                    d.ellipse([x - r, y - r, x + r, y + r], outline=(*col, 90))
                    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(*col, 230))
                    d.text((x + 8, y - 6), f"об.{i + 1}", fill=(*col, 255))
                else:
                    d.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(*col, 230))
                    name = P.SLOT_NAMES[i] if i < len(P.SLOT_NAMES) else f"слот {i + 1}"
                    d.text((x + 7, y - 6), name, fill=(*col, 255))

    # --- слои: список и параметры

    def _list_index(self, model_index):
        return len(self.doc.layers) - 1 - model_index

    def _refresh_layers(self, select_model=None):
        # Кэш картинок слоёв держится по id объекта, а операции над стопкой создают новые
        # объекты (клоны при отмене, слияние, сведение). id освободившегося объекта может быть
        # переиспользован — тогда кэш отдал бы чужую картинку. Чистим на каждой перестройке.
        self._layer_cache.clear()
        self.layer_list.delete(0, "end")
        for L in reversed(self.doc.layers):           # сверху — верхний слой, как в фотошопе
            self.layer_list.insert("end", L.label())
        if select_model is None:
            select_model = self.doc.active
        select_model = max(0, min(select_model, len(self.doc.layers) - 1))
        self.doc.active = select_model
        self.layer_list.selection_clear(0, "end")
        self.layer_list.selection_set(self._list_index(select_model))
        self._sync_layer_controls()

    def _select_from_list(self):
        sel = self.layer_list.curselection()
        if sel:
            self.doc.active = len(self.doc.layers) - 1 - sel[0]
            self._sync_layer_controls()

    def _sync_layer_controls(self):
        L = self.doc.active_layer()
        if L is None:
            return
        self._syncing = True
        self.layer_visible.set(L.visible)
        self.layer_locked.set(L.locked)
        self.layer_opacity.set(L.opacity)
        if L.kind == "image":
            self.layer_angle.set(L.angle)
            self.layer_scale.set(math.log2(max(L.scale, 1e-9) / self._base_scale(L)))
        self._syncing = False
        for child in self.img_box.winfo_children():
            child.configure(state=("normal" if L.kind == "image" else "disabled"))

    def _base_scale(self, L):
        Gx, Gy = self.doc.shape
        return max(Gx / L.image.width, Gy / L.image.height)

    def _apply_layer(self):
        L = self.doc.active_layer()
        if L is None or self._syncing:
            return
        L.visible = self.layer_visible.get()
        L.locked = self.layer_locked.get()
        L.opacity = float(self.layer_opacity.get())
        if L.kind == "image":
            L.angle = float(self.layer_angle.get())
            L.scale = self._base_scale(L) * (2.0 ** float(self.layer_scale.get()))
        self.doc.bump()
        keep = self.doc.active
        self._refresh_layers(select_model=keep)
        self.draw()
        self.schedule_measure()

    # --- слои: операции

    def add_tile_layer(self):
        self.push_undo()
        L = TileLayer(np.full(self.doc.shape, EMPTY, dtype=np.int8),
                      f"местность {len(self.doc.tile_layers()) + 1}")
        self.doc.layers.insert(self.doc.active + 1, L)
        self.doc.bump()
        self._refresh_layers(select_model=self.doc.active + 1)
        self.draw()
        self.status.config(text=f"добавлен пустой слой «{L.name}» — рисуйте, ниже всё видно")

    def add_image_layer(self):
        path = filedialog.askopenfilename(
            title="подложка", filetypes=[("изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if not path:
            return
        img = Image.open(path).convert("RGBA")
        if max(img.size) > MAX_LAYER_PX:              # рисуем в клетках, пиксели сверх — память зря
            k = MAX_LAYER_PX / max(img.size)
            img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
        self.push_undo()
        Gx, Gy = self.doc.shape
        L = ImageLayer(img, os.path.basename(path), (Gx / 2, Gy / 2),
                       max(Gx / img.width, Gy / img.height), path=path)
        self.doc.layers.insert(self.doc.active + 1, L)
        self.doc.bump()
        self._refresh_layers(select_model=self.doc.active + 1)
        self.draw()

    def duplicate_layer(self):
        L = self.doc.active_layer()
        if L is None:
            return
        self.push_undo()
        self.doc.layers.insert(self.doc.active + 1, L.clone())
        self.doc.bump()
        self._refresh_layers(select_model=self.doc.active + 1)
        self.draw()

    def delete_layer(self):
        if len(self.doc.tile_layers()) <= 1 and self.doc.active_layer().kind == "tiles":
            self.status.config(text="это единственный слой местности — удалять нечего, "
                                    "карте нужен хотя бы один")
            return
        self.push_undo()
        self.doc.layers.pop(self.doc.active)
        self.doc.bump()
        self._refresh_layers(select_model=min(self.doc.active, len(self.doc.layers) - 1))
        self.draw()
        self.schedule_measure()

    def move_layer(self, d):
        i, j = self.doc.active, self.doc.active + d
        if not (0 <= j < len(self.doc.layers)):
            return
        self.push_undo()
        self.doc.layers[i], self.doc.layers[j] = self.doc.layers[j], self.doc.layers[i]
        self.doc.bump()
        self._refresh_layers(select_model=j)
        self.draw()
        self.schedule_measure()

    def merge_down(self):
        """Слить активный слой местности с тем, что под ним. Прозрачность и видимость при этом
        НЕ учитываются: тайл — дискретная величина, полупрозрачного леса не бывает; прозрачность
        здесь только про показ."""
        i = self.doc.active
        if i == 0 or self.doc.layers[i].kind != "tiles" or self.doc.layers[i - 1].kind != "tiles":
            self.status.config(text="объединять можно только слой местности с таким же ниже")
            return
        self.push_undo()
        top, low = self.doc.layers[i], self.doc.layers[i - 1]
        m = top.grid >= 0
        low.grid[m] = top.grid[m]
        low.ver += 1
        self.doc.layers.pop(i)
        self.doc.bump()
        self._refresh_layers(select_model=i - 1)
        self.draw()
        self.schedule_measure()

    def flatten(self):
        """Свести ВИДИМЫЕ слои местности в один. Подложки не трогаем: это не пиксели карты, а
        опора для обводки, и терять их при сведении было бы неожиданно."""
        self.push_undo()
        comp = self.doc.composite().copy()
        images = [L for L in self.doc.layers if L.kind == "image"]
        self.doc.layers = [TileLayer(comp, "местность")] + images
        self.doc.bump()
        self._refresh_layers(select_model=0)
        self.draw()
        self.schedule_measure()
        self.status.config(text="слои местности сведены в один; подложки оставлены")

    def rename_layer(self):
        L = self.doc.active_layer()
        if L is None:
            return
        name = simpledialog.askstring("переименовать слой", "имя:", initialvalue=L.name, parent=self)
        if name:
            L.name = name
            self._refresh_layers(select_model=self.doc.active)

    # --- мышь

    def _cell_at(self, ev):
        fx, fy = self.view.to_world(self.W, self.H, ev.x, ev.y)
        gx, gy = int(math.floor(fx)), int(math.floor(fy))
        Gx, Gy = self.doc.shape
        return (gx, gy) if (0 <= gx < Gx and 0 <= gy < Gy) else None

    def _paint_target(self):
        """Слой, в который идёт кисть. Отказ — с объяснением: молчаливое «рисую, но не видно»
        хуже отказа (в фотошопе ровно так же)."""
        L = self.doc.active_layer()
        if L is None or L.kind != "tiles":
            self.status.config(text="активен слой-подложка: рисовать некуда — выберите слой местности")
            return None
        if L.locked:
            self.status.config(text=f"слой «{L.name}» под замком")
            return None
        if not L.visible:
            self.status.config(text=f"слой «{L.name}» скрыт — включите видимость, чтобы рисовать")
            return None
        return L

    def _start_pan(self, ev):
        self._drag = ("pan", (ev.x, ev.y, self.view.cx, self.view.cy))

    def _hit_marker(self, ev, radius=10):
        for kind in P.MARKER_KINDS:
            for i, (fx, fy) in enumerate(self.doc.markers[kind]):
                x, y = self.view.to_screen(self.W, self.H, fx, fy)
                if (x - ev.x) ** 2 + (y - ev.y) ** 2 <= radius * radius:
                    return kind, i
        return None

    def on_press(self, ev):
        if self._panning:
            return self._start_pan(ev)
        L = self.doc.active_layer()
        if self.layer_move.get() and L is not None and L.kind == "image":
            if L.locked:
                self.status.config(text=f"слой «{L.name}» под замком")
                return
            self.push_undo()
            self._drag = ("layer", (ev.x, ev.y, L.cx, L.cy, L))
            return
        tool = self.tool.get()
        if tool == "marker":
            hit = self._hit_marker(ev)
            self.push_undo()
            if hit:
                self._drag = ("marker", hit)
            else:
                self._place_marker(ev)
            return
        c = self._cell_at(ev)
        target = self._paint_target()
        if c is None or target is None:
            return
        self.push_undo()
        if tool == "brush":
            self._drag = ("brush", c)
            self._stamp(target.grid, *c, self.tile.get())
            self._touch(target)
        elif tool == "fill":
            self._fill(target, *c, self.tile.get())
            self._touch(target)
        else:
            self._drag = (tool, c)

    def on_motion(self, ev):
        self.on_hover(ev)
        if not self._drag:
            return
        mode, data = self._drag
        if mode == "pan":
            x0, y0, cx0, cy0 = data
            # тащим МИР под курсором: при повёрнутом виде «влево» на экране — не «влево» в мире
            self.view.cx, self.view.cy = cx0, cy0
            w0 = self.view.to_world(self.W, self.H, x0, y0)
            w1 = self.view.to_world(self.W, self.H, ev.x, ev.y)
            self.view.cx = cx0 + (w0[0] - w1[0])
            self.view.cy = cy0 + (w0[1] - w1[1])
            self.draw()
        elif mode == "layer":
            x0, y0, lx0, ly0, L = data
            w0 = self.view.to_world(self.W, self.H, x0, y0)
            w1 = self.view.to_world(self.W, self.H, ev.x, ev.y)
            L.cx, L.cy = lx0 + (w1[0] - w0[0]), ly0 + (w1[1] - w0[1])
            self.draw()
        elif mode == "marker":
            kind, i = data
            self.doc.markers[kind][i] = list(self.view.to_world(self.W, self.H, ev.x, ev.y))
            self.draw()
        elif mode in ("brush", "erase"):
            c = self._cell_at(ev)
            target = self._paint_target()
            if c and target:
                self._stroke(target.grid, data, c, EMPTY if mode == "erase" else self.tile.get())
                self._drag = (mode, c)
                self._touch(target)
        elif mode in ("line", "rect"):
            c = self._cell_at(ev)
            target = self._paint_target()
            if c and target:
                g = target.grid.copy()
                if mode == "line":
                    self._stroke(g, data, c, self.tile.get())
                else:
                    (x0, y0), (x1, y1) = data, c
                    g[min(x0, x1):max(x0, x1) + 1, min(y0, y1):max(y0, y1) + 1] = self.tile.get()
                self._preview = (target, g)
                self.draw()

    def on_release(self, ev):
        if self._preview is not None:
            target, g = self._preview
            target.grid = g
            self._preview = None
            self._touch(target)
        if self._drag and self._drag[0] in ("brush", "erase", "line", "rect", "marker"):
            self.schedule_measure()
        self._drag = None
        self._refresh_points()

    def on_right(self, ev):
        if self.tool.get() == "marker":
            hit = self._hit_marker(ev)
            if hit:
                self.push_undo()
                self.doc.markers[hit[0]].pop(hit[1])
                self.draw()
                self._refresh_points()
            return
        c = self._cell_at(ev)
        target = self._paint_target()
        if c is None or target is None:
            return
        self.push_undo()
        self._drag = ("erase", c)
        self._stamp(target.grid, *c, EMPTY)          # ластик: открыть слой ниже
        self._touch(target)

    def on_right_motion(self, ev):
        if self._drag and self._drag[0] == "erase":
            self.on_motion(ev)

    def on_hover(self, ev):
        fx, fy = self.view.to_world(self.W, self.H, ev.x, ev.y)
        Gx, Gy = self.doc.shape
        inside = 0 <= fx < Gx and 0 <= fy < Gy
        if inside:
            comp = self.doc.composite()
            tile = P.TILE_NAMES[int(comp[int(fx), int(fy)])]
            L = self.doc.active_layer()
            if L is not None and L.kind == "tiles":
                own = int(L.grid[int(fx), int(fy)])
                tile += "  · в слое: " + ("пусто" if own < 0 else P.TILE_NAMES[own])
        else:
            tile = "вне карты"
        ux, uy = P.units_of_cell(fx, fy)
        self.status.config(text=f"{fx * self.doc.cell_m:7.0f} x {fy * self.doc.cell_m:7.0f} м   "
                                f"({ux:.1f}, {uy:.1f} ед)   {tile}")

    # --- правка сетки

    def _touch(self, layer):
        layer.ver += 1
        self.doc.bump()
        self.draw()

    def push_undo(self):
        self.undo.append(self.doc.snapshot())
        del self.undo[:-60]
        self.redo.clear()

    def _stamp(self, grid, gx, gy, value):
        r = self.brush.get() - 1
        Gx, Gy = grid.shape
        grid[max(0, gx - r):min(Gx, gx + r + 1), max(0, gy - r):min(Gy, gy + r + 1)] = value

    def _stroke(self, grid, a, b, value):
        """Мазок отрезком, а не точками: на быстром движении события приходят через десяток
        клеток, и кисть оставляла бы пунктир."""
        (x0, y0), (x1, y1) = a, b
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(n + 1):
            self._stamp(grid, int(round(x0 + (x1 - x0) * i / n)),
                        int(round(y0 + (y1 - y0) * i / n)), value)

    def _fill(self, layer, gx, gy, value):
        """Область берём по ВИДИМОЙ карте (что залил бы глаз), а пишем в активный слой —
        так «залить озеро» работает и когда вода нарисована слоем ниже."""
        comp = self.doc.composite()
        src = int(comp[gx, gy])
        Gx, Gy = self.doc.shape
        seen = np.zeros(self.doc.shape, dtype=bool)
        stack = [(gx, gy)]
        while stack:
            x, y = stack.pop()
            if not (0 <= x < Gx and 0 <= y < Gy) or seen[x, y] or comp[x, y] != src:
                continue
            seen[x, y] = True
            layer.grid[x, y] = value
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def do_undo(self):
        if self.undo:
            self.redo.append(self.doc.snapshot())
            self.doc.restore(self.undo.pop())
            self._after_stack_change()

    def do_redo(self):
        if self.redo:
            self.undo.append(self.doc.snapshot())
            self.doc.restore(self.redo.pop())
            self._after_stack_change()

    def _after_stack_change(self):
        self._layer_cache.clear()
        self._refresh_layers(select_model=self.doc.active)
        self.draw()
        self._refresh_points()
        self.schedule_measure()

    def do_mirror(self):
        """Зеркалим активный слой по Y — та же операция, которой test_mirror.py проверяет
        симметрию среды: нарисовал половину, отразил, и разница в исходе боя точно не от карты."""
        L = self._paint_target()
        if L is None:
            return
        self.push_undo()
        Gy = self.doc.shape[1]
        L.grid[:, Gy - Gy // 2:] = L.grid[:, :Gy // 2][:, ::-1]
        self._touch(L)
        self.schedule_measure()

    def do_rot90(self):
        """Поворот САМОЙ карты: крутятся все слои местности разом, иначе стопка разъедется."""
        self.push_undo()
        for L in self.doc.tile_layers():
            L.grid = np.rot90(L.grid).copy()
            L.ver += 1
        self.doc.Gx, self.doc.Gy = self.doc.Gy, self.doc.Gx
        self.doc.markers = {k: [] for k in P.MARKER_KINDS}   # точки после поворота бессмысленны
        self.doc.bump()
        self._layer_cache.clear()
        self.fit_view()
        self._refresh_points()
        self.schedule_measure()
        self.status.config(text="карта повёрнута; точки сброшены — расставьте заново")

    def do_clear(self):
        L = self._paint_target()
        if L is None:
            return
        self.push_undo()
        L.grid[:] = EMPTY if self.doc.layers.index(L) > 0 else 0
        self._touch(L)
        self.schedule_measure()

    def fit_layer(self):
        L = self.doc.active_layer()
        if L is None or L.kind != "image":
            return
        self.push_undo()
        Gx, Gy = self.doc.shape
        L.cx, L.cy = Gx / 2, Gy / 2
        L.angle = 0.0
        L.scale = self._base_scale(L)
        self._sync_layer_controls()
        self.draw()

    # --- точки

    def _place_marker(self, ev):
        kind = self.marker_kind.get()
        fx, fy = self.view.to_world(self.W, self.H, ev.x, ev.y)
        Gx, Gy = self.doc.shape
        if not (0 <= fx < Gx and 0 <= fy < Gy):
            return
        if kind in ("friendly", "enemy") and len(self.doc.markers[kind]) >= P.N_SIDE:
            self.status.config(text=f"{P.MARKER_RU[kind]}: уже {P.N_SIDE} — столько же, сколько "
                                    f"слотов в составе; лишние сценарий не примет")
            return
        self.doc.markers[kind].append([fx, fy])
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
            for i, (fx, fy) in enumerate(pts):
                ux, uy = P.units_of_cell(fx, fy)
                tag = (f"об.{i + 1}" if kind == "zones"
                       else (P.SLOT_NAMES[i] if i < len(P.SLOT_NAMES) else f"слот {i + 1}"))
                lines.append(f"   {tag:<10} {ux:6.1f} {uy:6.1f} ед")
        problems = self._scenario_problems()
        lines += ["", "сценарий готов" if not problems else "сценарий неполон:"]
        lines += ["  " + s for s in problems]
        self.points_info.delete("1.0", "end")
        self.points_info.insert("end", "\n".join(lines))

    # --- замер

    def schedule_measure(self, delay=400):
        if self._measure_job:
            self.after_cancel(self._measure_job)
        self._measure_job = self.after(delay, self.remeasure)

    def remeasure(self):
        self._measure_job = None
        self.info.delete("1.0", "end")
        self.info.insert("end", "меряю…")
        self.update_idletasks()
        grid = self.doc.composite()
        m = measure(grid, self.doc.cell_m)
        self.metrics = m
        Gx, Gy = self.doc.shape
        span = Gx * self.doc.cell_m
        fit = abs(span - P.ARENA_M) < P.ARENA_M * 0.05
        lines = [f"сетка   {Gx}x{Gy} по {self.doc.cell_m:.0f} м = {span:.0f} м",
                 f"арена   {P.ARENA_M:.0f} м" + ("" if fit else "   ← НЕ СОВПАДАЕТ"),
                 f"слоёв местности {len(self.doc.tile_layers())}"
                 + (f", скрыто {len(self.doc.hidden_tile_layers())}"
                    if self.doc.hidden_tile_layers() else ""), ""]
        for tid, ru in P.TILE_NAMES.items():
            lines.append(f"{ru:<10}{m['frac'][tid] * 100:5.1f}%")
        lines += ["",
                  f"видимость 300-900 м: {m['vis'] * 100:.0f}%   ({m['pairs']} пар)",
                  "годно 25-85%; пул сейчас 27-81%",
                  f"строений {m['comps']}, макс габарит {m['span_m']:.0f} м", ""]
        lines.append("ГОДНА" if not m["bad"] else "ВЫРОЖДЕНА:")
        lines += ["  " + b for b in m["bad"]]
        self.info.delete("1.0", "end")
        self.info.insert("end", "\n".join(lines))

    # --- файлы

    def save_map(self):
        name = self.name_var.get().strip()
        if not name:
            return
        npy = os.path.join(P.MAPS, name + ".npy")
        if os.path.exists(npy) and not messagebox.askyesno("перезаписать?",
                                                           f"{name}.npy уже есть. Заменить?"):
            return
        np.save(npy, self.doc.composite().astype(np.int8))
        # building_capacity намеренно пустой: он нужен импортированным картам, где несколько
        # реальных домов слиплись в один компонент сетки. Здесь компоненты режутся на кварталы
        # 2x2 клетки (terrain.BLOCK_CELLS), то есть строение = один отряд по умолчанию.
        # Блок "editor" среда не читает — это память редактора: точки и описание стопки слоёв.
        meta = {"cell_m": self.doc.cell_m, "building_capacity": {}, "source": "map_editor",
                "editor": {"markers": self.doc.markers, "layers": self._layers_meta()}}
        with open(os.path.join(P.MAPS, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        grids = {f"L{i}": L.grid for i, L in enumerate(self.doc.layers) if L.kind == "tiles"}
        np.savez_compressed(os.path.join(P.MAPS, name + ".editor.npz"), **grids)
        self.doc.name = name
        hidden = self.doc.hidden_tile_layers()
        note = (f"; СКРЫТО слоёв: {len(hidden)} — в карту они не вошли" if hidden else "")
        self.status.config(text=f"сохранено: maps/{name}.npy + .json + .editor.npz{note}")

    def _layers_meta(self):
        out = []
        for L in self.doc.layers:
            d = {"kind": L.kind, "name": L.name, "visible": L.visible,
                 "opacity": round(L.opacity, 3), "locked": L.locked}
            if L.kind == "image":
                d.update(path=L.path, cx=round(L.cx, 3), cy=round(L.cy, 3),
                         scale=round(L.scale, 6), angle=round(L.angle, 2))
            out.append(d)
        return out

    def save_preview(self):
        name = self.name_var.get().strip()
        out_dir = os.path.join(P.MAPS, "preview")
        os.makedirs(out_dir, exist_ok=True)
        lut = np.zeros((max(P.TILE_COLORS) + 1, 3), dtype=np.uint8)
        for k, c in P.TILE_COLORS.items():
            lut[k] = c
        rgb = lut[self.doc.composite()]
        img = Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1]))
        img = img.resize((img.width * 8, img.height * 8), Image.NEAREST)
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
        составом — иначе scenario.validate() честно скажет «не загрузится», но уже на замере."""
        problems = self._scenario_problems()
        if problems and not messagebox.askyesno(
                "сценарий неполон", "\n".join(problems) + "\n\nВсё равно сохранить?"):
            return
        name = self.name_var.get().strip()
        if not os.path.exists(os.path.join(P.MAPS, name + ".npy")):
            messagebox.showwarning("карта не сохранена",
                                   f"Сценарий сошлётся на maps/{name}, а такой карты нет. "
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


# ---------------------------------------------------------------- загрузка/приложение


def scenario_dict(name, markers):
    conv = lambda pts: [[round(v, 2) for v in P.units_of_cell(*p)] for p in pts]   # noqa: E731
    return {"name": name, "map": f"maps/{name}", "map_seed": 0,
            "zones": conv(markers["zones"]), "friendly": conv(markers["friendly"]),
            "enemy": conv(markers["enemy"])}


def _glob_maps():
    import glob as _g
    return [p for p in _g.glob(os.path.join(P.MAPS, "*.npy")) if not p.endswith(".editor.npy")]


def _meta_of(path):
    meta_path = path[:-4] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_doc(path):
    """Карта + стопка слоёв, если она сохранена рядом. Нет стопки — карта становится одним
    слоем: так открываются и вырезки, которые редактор в глаза не видел."""
    grid = np.load(path)
    meta = _meta_of(path)
    cell_m = float(meta.get("cell_m", P.CELL_M))
    name = os.path.basename(path)[:-4]
    ed_meta = meta.get("editor") or {}
    layers, lost = [], []
    npz_path = path[:-4] + ".editor.npz"
    if ed_meta.get("layers") and os.path.exists(npz_path):
        with np.load(npz_path) as z:
            # Сетки берём ПО ПОРЯДКУ, а не по имени ключа: если подложка не нашлась на диске,
            # она выпадает из стопки, и нумерация ключей разъезжается с индексами слоёв.
            keys = sorted(z.files, key=lambda k: int(k[1:]))
            ti = 0
            for d in ed_meta["layers"]:
                if d["kind"] == "tiles":
                    if ti >= len(keys):
                        continue
                    arr = z[keys[ti]]
                    ti += 1
                    layers.append(TileLayer(arr, d.get("name", "местность"),
                                            visible=d.get("visible", True),
                                            opacity=d.get("opacity", 1.0),
                                            locked=d.get("locked", False)))
                elif d.get("path") and os.path.exists(d["path"]):
                    img = Image.open(d["path"]).convert("RGBA")
                    if max(img.size) > MAX_LAYER_PX:
                        k = MAX_LAYER_PX / max(img.size)
                        img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
                    layers.append(ImageLayer(img, d.get("name", "подложка"),
                                             (d.get("cx", 0), d.get("cy", 0)), d.get("scale", 1.0),
                                             d.get("angle", 0.0), d.get("opacity", 0.6),
                                             d["path"], visible=d.get("visible", True),
                                             locked=d.get("locked", False)))
                else:
                    lost.append(d.get("name", "подложка"))
    if not any(L.kind == "tiles" for L in layers):
        layers = [TileLayer(grid, "местность")] + [L for L in layers if L.kind == "image"]
    doc = Doc(grid.shape, cell_m, name, layers)
    for kind in P.MARKER_KINDS:
        doc.markers[kind] = [list(map(float, p)) for p in (ed_meta.get("markers") or {}).get(kind, [])]
    if lost:
        print("подложки не найдены на диске и пропущены: " + ", ".join(lost))
    return doc


class App:
    def __init__(self, root, doc=None):
        self.root = root
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
    ap.add_argument("name", nargs="?", help="имя карты в maps/ без расширения — открыть сразу")
    args = ap.parse_args()

    doc = load_doc(os.path.join(P.MAPS, args.name + ".npy")) if args.name else None
    print(P.describe())
    root = tk.Tk()
    App(root, doc)
    root.mainloop()


if __name__ == "__main__":
    main()
