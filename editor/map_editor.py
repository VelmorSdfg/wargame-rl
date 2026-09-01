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


def _timer_resolution(ms=1):
    """Попросить у Windows миллисекундный тик таймера. Возвращает, чем вернуть как было.

    Часы Windows по умолчанию будят таймеры раз в 15.625 мс, поэтому tkinter-овский after(16) —
    это «больше пятнадцати с половиной», то есть ожидание ВТОРОГО тика: 31 мс. На кадре в
    десяток миллисекунд это давало ровно 24 к/с, причём ОДИНАКОВЫХ при любой сцене — частоту
    задавали часы, а не отрисовка, и потому она не менялась ни от масштаба, ни от прострелов,
    ни от числа кусков. Просить мелкий тик глобально не принято без нужды (он держит процессор
    в тонусе и греет ноутбук), поэтому на выходе возвращаем прежний."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        winmm = ctypes.WinDLL("winmm")
        if winmm.timeBeginPeriod(int(ms)) != 0:
            return None
        return lambda: winmm.timeEndPeriod(int(ms))
    except Exception:                                        # noqa: BLE001
        return None                                          # чужая система — живём как жили


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
import view3d                                                # noqa: E402
import paint                                                 # noqa: E402
import tiles as tilemod                                      # noqa: E402


VIEW_STEP_M = 15.0         # верхний предел шага выборки по лучу, В МЕТРАХ (раньше шаг был
#                            жёстко полклетки, и на грубой клетке выходил слишком редким).
#
#                            ОТМЕНЁННАЯ ЗАМЕРОМ ГИПОТЕЗА. Казалось, что сарай 8x6 не даёт тени
#                            в просмотре потому, что луч через него перешагивает. Сгущение шага
#                            не помогло НИ РАЗУ: 15 м — нет тени (26 мс), 8 м — нет (134 мс),
#                            5 м — нет (582 мс), 3 м — нет (1276 мс). Дело не в выборке, а в
#                            разрешении САМОЙ КАРТИНКИ: тень восьмиметрового дома уже клетки в
#                            30 м, и в ту же клетку попадают лучи, прошедшие мимо, — а поле
#                            собирается максимумом. При клетке показа 10 м тень появляется
#                            сразу. Так что мельчить шаг незачем; мельчить надо клетку.


def viewshed(tm, center_m, radius_m, m_per_unit, max_rays=1440):
    """Что видно из точки: поле 0..1 — сколько ЗАПАСА ПРОЗРАЧНОСТИ осталось у луча в этой клетке.

    Не «видно / не видно», а плавная величина, и это не украшение. В нашей модели лес не
    перекрывает обзор сразу: луч копит толщину и гаснет, набрав порог (90 м леса). Значит у
    опушки видно вглубь, а дальше сходит на нет — двоичная картинка это врала, показывая либо
    сплошной свет, либо сплошную тень, и именно поэтому «за лесами показывало плохо».

    Считается numpy'ем, все лучи разом: за мышью пером на питоне не угнаться, а так выходит
    десяток миллисекунд, и картинку можно тащить.

    РЕЛЬЕФ учитывается так же, как в бою: гребень перекрывает обзор. Раньше здесь смотрели
    только на материал — лес и строения, — и холм между наблюдателем и целью для просмотра не
    существовал вовсе: обратный скат подсвечивался как видимый. Обманывало это вдвойне, потому
    что маска ложится на объёмную местность, обтекает холм, и глаз заключает, что холм учтён.
    Бой при этом рельеф считал (terrain.blocked), и замер годности карты тоже — расходился
    ровно тот показ, по которому выбирают позиции.

    Считается это накоплением МАКСИМАЛЬНОГО УГЛА подъёма вдоль луча: клетка видна, пока земля в
    ней поднимается не ниже самого крутого угла, взятого до неё. Одна np.maximum.accumulate по
    той же оси, по которой уже копится толщина растительности, — цена почти нулевая.

    Глаза наблюдателя и цели подняты на EYE_UNITS (2 м роста), как в бою: без этого складка
    глубиной в метр считалась бы укрытием.

    Оговорка о приближении: бой копит толщину ОТДЕЛЬНО по каждому материалу, здесь она копится
    общей нормированной суммой. Различие проявляется только на луче, прошедшем и лес, и здание;
    у здания порог нулевой, оно гасит луч сразу, поэтому на практике картинка совпадает.
    """
    cx, cy = center_m[0] / m_per_unit, center_m[1] / m_per_unit
    R = radius_m / m_per_unit
    step = min(tm.cell * 0.5, VIEW_STEP_M / float(m_per_unit))
    n_steps = max(2, int(R / step))
    n_rays = int(np.clip(2.0 * math.pi * R / step, 360, max_rays))

    ang = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False, dtype=np.float32)
    dist = (np.arange(1, n_steps + 1, dtype=np.float32) * step)[None, :]
    xs = cx + np.cos(ang)[:, None] * dist
    ys = cy + np.sin(ang)[:, None] * dist
    inside = (xs >= 0) & (ys >= 0) & (xs < tm.width_m) & (ys < tm.height_m)
    gx = np.clip((xs / tm.cell).astype(np.int32), 0, tm.Gx - 1)
    gy = np.clip((ys / tm.cell).astype(np.int32), 0, tm.Gy - 1)

    vb = getattr(tm, "f_blocks_vec", None)
    if vb is not None:
        # ПОМЕХИ ИЗ ВЕКТОРА, но разложенные по клеткам ОДИН РАЗ НА ОКНО (vector_occluders).
        #
        # Сперва здесь стоял честный перебор: каждая фигура против всех точек лучей. Смысл был
        # верный, цена — нет: на театре это десятки миллионов проверок, и просмотр стоил 332 мс
        # на движение мыши вместо трёх. Клетки для ПОКАЗА достаточно, а порог существования,
        # ради которого вектор и заводился, снят иначе — отметкой по ЛЮБОМУ касанию, а не по
        # доле покрытия: сарай 8x6 свою клетку получает.
        #
        # Бой при этом считает точной геометрией (terrain.blocked): показ огрубляет тень мелкого
        # дома до клетки, но не теряет её.
        blocks = vb[gx, gy]
        lim = tm.f_see_vec[gx, gy]
    else:
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

    # --- гребень: то же правило, что в terrain.blocked, только сразу на весь луч
    if getattr(tm, "has_relief", False):
        eye = terrain.EYE_UNITS
        h_here = tm.f_height[gx, gy]
        gx0 = int(np.clip(cx / tm.cell, 0, tm.Gx - 1))
        gy0 = int(np.clip(cy / tm.cell, 0, tm.Gy - 1))
        h_eye = float(tm.f_height[gx0, gy0]) + eye
        # угол на ВЕРХ цели (глаза стоящего человека), а перекрывает — САМА земля
        ang_to = (h_here + eye - h_eye) / np.maximum(dist, 1e-6)
        ang_land = (h_here - h_eye) / np.maximum(dist, 1e-6)
        need = np.maximum.accumulate(ang_land, axis=1)
        # до входа в клетку: сама клетка себя не заслоняет
        need = np.concatenate([np.full((need.shape[0], 1), -np.inf, dtype=np.float32),
                               need[:, :-1]], axis=1)
        alive = alive & (ang_to >= need)
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


def _lighten(col, k=0.30):
    """Цвет типа в полную силу — им рисуется незакрытая фигура.

    Цвета местности намеренно приглушены (лес 26,58,30), и чертёж такого же цвета на лесу же и
    теряется: незакрытую фигуру было почти не видно. Оттенок оставляем тот же, но выводим на
    полную яркость и добавляем белого — тогда фигура читается и на тёмной земле, и на белом
    поле топостиля."""
    m = max(col[:3]) or 1
    full = [min(255, int(c * 190 / m)) for c in col[:3]]
    return tuple(int(c + (255 - c) * k) for c in full)


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


def _mix(a, b, t):
    """Смешать два цвета вида #rrggbb. Нужен для плавного тумблера: ложе едет от серого к
    янтарному вместе с бегунком, иначе переключение выглядит как рывок."""
    t = max(0.0, min(1.0, float(t)))
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(ca, cb))


class Switch(tk.Frame):
    """Тумблер вместо квадрата с крестиком.

    clam рисует ttk.Checkbutton как серый квадрат с крестом, и включённое от выключенного на
    глаз отличается только тем, есть в квадрате крест или нет — на тёмной панели это читается
    хуже всего в интерфейсе. Тумблер показывает состояние формой, цветом и положением бегунка,
    то есть виден боковым зрением, не вчитываясь.

    Подпись раскладывается на имя и пояснение: строка «объёмный вид — камера в пространстве»
    длиннее панели в 352 px, и тире посреди переноса читается как список из двух пунктов."""
    W, H, PAD = 40, 20, 3          # длина ложа, высота, зазор от края до бегунка
    KNOB_ON = "#f0dfb4"            # бегунок включённого — светлее янтаря, чтобы не слиться

    def __init__(self, master, text, hint="", variable=None, command=None):
        super().__init__(master, background=UI["panel"], highlightthickness=0, bd=0)
        self.var = variable if variable is not None else tk.BooleanVar(value=False)
        self.cmd = command
        self._pos = 1.0 if self.var.get() else 0.0
        self._hot = False
        self._job = None
        self.cv = tk.Canvas(self, width=self.W, height=self.H, background=UI["panel"],
                            highlightthickness=0, bd=0)
        self.cv.pack(side="left", padx=(0, 9), pady=1, anchor="n")
        box = tk.Frame(self, background=UI["panel"])
        box.pack(side="left", fill="x", expand=True)
        self.lab = tk.Label(box, text=text, background=UI["panel"], foreground=UI["text"],
                            font=("Segoe UI", 9), anchor="w", justify="left")
        self.lab.pack(anchor="w")
        self.hint = None
        if hint:
            self.hint = tk.Label(box, text=hint, background=UI["panel"], foreground=UI["dim"],
                                 font=("Segoe UI", 8), anchor="w", justify="left",
                                 wraplength=248)
            self.hint.pack(anchor="w")
        for w in (self, self.cv, box, self.lab, self.hint):
            if w is None:
                continue
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)
        self._trace = self.var.trace_add("write", self._changed)
        self._paint()

    def toggle(self):
        self.var.set(not self.var.get())
        if self.cmd:
            self.cmd()

    def _click(self, _ev=None):
        self.toggle()
        return "break"

    def _enter(self, _ev=None):
        self._hot = True
        self._paint()

    def _leave(self, _ev=None):
        self._hot = False
        self._paint()

    def _changed(self, *_a):
        """Значение могли поставить и мимо тумблера — горячей клавишей или кодом. Рисунок
        должен идти за переменной, а не за щелчком."""
        if self.winfo_exists() and self._job is None:
            self._step()

    def _step(self):
        if not self.winfo_exists():
            return
        tgt = 1.0 if self.var.get() else 0.0
        self._pos += (tgt - self._pos) * 0.45
        if abs(tgt - self._pos) < 0.02:
            self._pos, self._job = tgt, None
        else:
            self._job = self.after(16, self._step)
        self._paint()

    def _pill(self, x0, y0, x1, y1, color):
        r = (y1 - y0) / 2.0
        self.cv.create_oval(x0, y0, x0 + 2 * r, y1, fill=color, outline=color)
        self.cv.create_oval(x1 - 2 * r, y0, x1, y1, fill=color, outline=color)
        self.cv.create_rectangle(x0 + r, y0, x1 - r, y1, fill=color, outline=color)

    def _paint(self):
        if not self.winfo_exists():
            return
        cv, on = self.cv, self._pos
        cv.delete("all")
        edge = UI["accent"] if (on > 0.5 or self._hot) else UI["line"]
        self._pill(0, 0, self.W, self.H, edge)
        self._pill(1, 1, self.W - 1, self.H - 1, _mix(UI["field"], UI["accent_dim"], on))
        d = self.H - 2 * self.PAD
        x = self.PAD + on * (self.W - 2 * self.PAD - d)
        cv.create_oval(x, self.PAD, x + d, self.PAD + d,
                       fill=_mix(UI["dim"], self.KNOB_ON, on), outline="")
        self.lab.configure(foreground=UI["text"] if (on > 0.5 or self._hot) else UI["dim"])

    def destroy(self):
        if self._job is not None:
            self.after_cancel(self._job)
        try:
            self.var.trace_remove("write", self._trace)
        except (tk.TclError, ValueError):
            pass
        super().destroy()

# Материалы клетки — и только они. Рельеф здесь был чужим: он не материал, за клетку не спорит
# (на одной высоте бывает и лес, и поле), фигурой не остаётся и живёт в другом файле. Ему свой
# блок в панели — см. РЕЛЬЕФ.
SHAPE_TYPES = ("forest", "water", "building", "road", "open")
TYPE_RU = {"forest": "лес", "water": "вода", "building": "застройка", "road": "дорога",
           "open": "поле (вырезает)", "relief": "рельеф"}
TYPE_COLOR = {"forest": P.TILE_COLORS[1], "water": P.TILE_COLORS[3],
              "building": P.TILE_COLORS[2], "road": P.TILE_COLORS[4], "open": P.TILE_COLORS[0],
              "relief": (150, 120, 84)}
DEFAULT_WIDTH = {"road": 8.0, "water": 30.0, "forest": 25.0, "building": 20.0, "open": 40.0,
                 "relief": 400.0}

# Высоты в метрах. Отрицательные — лощина и овраг: то же самое поле, только вниз. Ширина у
# линии рельефа — это ширина гряды, а не тропинки, поэтому счёт идёт сотнями метров.
RELIEF_PRESETS = [("бугор", 15), ("холм", 35), ("высота", 60),
                  ("лощина", -15), ("овраг", -30), ("котловина", -45)]

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


def _smooth_mask(v):
    """Сгладить маску видимости ПОКАЗА ради.

    Считается она в клетке боя (15 м), и на экране это кубики: граница тени идёт ступеньками в
    полклетки. Три прохода скользящего среднего с радиусом в клетку убирают ступеньку, форму тени
    оставляют. Числа в строке состояния берутся с НЕсглаженного поля — сглаживание тут только
    для глаза."""
    a = np.asarray(v, dtype=np.float32)
    for _ in range(3):
        b = a.copy()
        b[1:] += a[:-1]; b[:-1] += a[1:]
        b[:, 1:] += a[:, :-1]; b[:, :-1] += a[:, 1:]
        b[1:, 1:] += a[:-1, :-1]; b[:-1, :-1] += a[1:, 1:]
        b[1:, :-1] += a[:-1, 1:]; b[:-1, 1:] += a[1:, :-1]
        a = b / 9.0
    return a


def _split(img):
    """RGBA -> (цвет, маска) один раз. paste всё равно переведёт картинку в режим холста, и если
    не разделить заранее, перевод повторяется на каждом кадре."""
    if img is None:
        return None
    return img.convert("RGB"), img.getchannel("A")


class View:
    """Мир (МЕТРЫ, y вверх) -> экран (пиксели, y вниз). Одной матрицей: поворот вида, поворот
    подложки и обратное преобразование под курсором — одна и та же арифметика, а руками
    расписанная она разъезжается (в проекте это уже стоило перепутанной оси Y)."""

    def __init__(self, zoom=0.3):
        self.cx = self.cy = 0.0
        self.zoom = zoom                                     # пикселей на метр
        self.angle = 0.0
        self._mkey = None
        self._m = self._minv = None

    def matrix(self, w, h):
        """Мир -> экран. Матрица ЗАПОМИНАЕТСЯ: её просили по разу на каждую точку каждой фигуры,
        и на театре это выходило четырнадцать тысяч сборок матрицы за кадр — сорок три
        миллисекунды из пятидесяти одной уходили сюда, а не на рисование."""
        key = (w, h, self.cx, self.cy, self.zoom, self.angle)
        if self._mkey != key:
            self._m = (_translate(w / 2, h / 2) @ _scale(self.zoom, -self.zoom)
                       @ _rotate(self.angle) @ _translate(-self.cx, -self.cy))
            self._minv = np.linalg.inv(self._m)
            self._mkey = key
        return self._m

    def inv(self, w, h):
        self.matrix(w, h)
        return self._minv

    def to_screen(self, w, h, x, y):
        p = self.matrix(w, h) @ np.array([x, y, 1.0])
        return float(p[0]), float(p[1])

    def to_screen_many(self, w, h, pts):
        """Пачкой: список точек мира -> список точек экрана. Одно умножение вместо цикла."""
        a = np.asarray(pts, dtype=np.float64)
        m = self.matrix(w, h)
        xy = a @ m[:2, :2].T + m[:2, 2]
        return [(float(x), float(y)) for x, y in xy]

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
        self.version = 0                 # версия ВЕКТОРА: по ней пересобираются картинки кусков
        self.hversion = 0                # версия ВЫСОТЫ: она их не трогает, поэтому считается
        #                                  отдельно — иначе каждый штамп рельефа заставлял бы
        #                                  пересчитывать всю местность вокруг камеры
        self._surface = None
        self._surface_key = None
        self._height = None
        self._height_key = None
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

    def bump_height(self):
        self.hversion += 1

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

    def height_m(self, cell_m=None):
        """Поле высот в МЕТРАХ для показа.

        Считается ОТДЕЛЬНО от сетки типов, хотя раньше шло вместе с ней. Причина: высота теперь
        растровый слой, и получить её в нужной клетке — это пересчёт готового растра (единицы
        миллисекунд), а не растеризация всех фигур карты (на театре — треть секунды). Правка
        рельефа не должна стоить пересборки местности, которая от высоты не зависит."""
        cell = cell_m or max(self.cell_m, max(self.size_m) / PREVIEW_MAX)
        key = (self.hversion, round(cell, 3))
        if self._height_key != key:
            gx = max(1, int(round(self.size_m[0] / cell)))
            gy = max(1, int(round(self.size_m[1] / cell)))
            self._height = vectormap.height_field(self.vec, gx, gy, cell)
            self._height_key = key
        return self._height

    def surface(self, cell_m=None, stale_ok=False):
        """Растеризация в сетку. Для показа клетка берётся покрупнее (PREVIEW_MAX).

        stale_ok — отдать прошлую, даже если вектор изменился: во время перетаскивания важнее
        отзывчивость, чем точность предпросмотра."""
        cell = cell_m or max(self.cell_m, max(self.size_m) / PREVIEW_MAX)
        key = (self.version, round(cell, 3))
        if stale_ok and self._surface is not None:
            return self._surface, self._surface_key[1]
        if self._surface_key != key:
            self._surface = vectormap.surface_window(self.vec, cell, 0.0, 0.0,
                                                     self.size_m[0], self.size_m[1])
            self._surface_key = key
        return self._surface, cell

    def snapshot(self):
        """Слепок для отмены. Карта высот идёт целиком: это растр, обратной операции у штампа
        нет, и без копии «отменить холм» было бы нечем. Полкилобайта на шаг для боевой карты —
        приемлемая цена, шестьдесят шагов держим."""
        hm = self.vec.get("height")
        return (copy.deepcopy(self.vec["shapes"]), copy.deepcopy(self.markers),
                None if hm is None else {"cell_m": hm["cell_m"], "h": hm["h"].copy()},
                copy.deepcopy(self.vec.get("blown", [])))

    def restore(self, snap):
        self.vec["shapes"] = copy.deepcopy(snap[0])
        self.markers = copy.deepcopy(snap[1])
        if len(snap) > 2:
            if snap[2] is None:
                self.vec.pop("height", None)
            else:
                self.vec["height"] = {"cell_m": snap[2]["cell_m"], "h": snap[2]["h"].copy()}
        # Взорванные мосты откатываются вместе с фигурами: иначе «отменить снос» вернуло бы
        # переправу, а карта продолжала бы считать её снесённой намеренно.
        if len(snap) > 3:
            self.vec["blown"] = copy.deepcopy(snap[3])
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
        self.relief_h = tk.DoubleVar(value=35.0)
        self.relief_slope = tk.DoubleVar(value=150.0)   # длина склона в МЕТРАХ, своя у штампа
        self.relief_width = tk.DoubleVar(value=400.0)   # ширина гряды или лощины
        self.relief_abs = tk.BooleanVar(value=False)    # выровнять до отметки, а не прибавить
        self.show_shade = tk.BooleanVar(value=True)
        self.view_angle = tk.DoubleVar(value=0.0)
        self.show_fps = tk.BooleanVar(value=True)
        self.map_style = tk.StringVar(value="vector")   # стиль местности в объёме
        self.bridge_auto = tk.BooleanVar(value=True)
        self.bridge_ford = tk.BooleanVar(value=False)
        self.bridge_len = tk.DoubleVar(value=80.0)
        self.bridge_wid = tk.DoubleVar(value=8.0)
        self.ruler_circle = tk.BooleanVar(value=False)
        # Линейка по умолчанию ГАСНЕТ, как отпустил кнопку. Она измерение, а не объект карты:
        # оставаясь на экране, мешает смотреть ровно на то, что померили. Кому нужно сравнить
        # два расстояния подряд или показать замер — тумблер «оставлять линейку».
        self.ruler_keep = tk.BooleanVar(value=False)
        self.vision_r = tk.DoubleVar(value=1000.0)
        self.dash_on = tk.BooleanVar(value=False)
        self.dash_len = tk.DoubleVar(value=45.0)
        self.dash_gap = tk.DoubleVar(value=30.0)
        self.marker_kind = tk.StringVar(value="zones")
        self.show_grid = tk.BooleanVar(value=False)      # сетка — отладочный слой, не фон
        # Узлы дорог по умолчанию СКРЫТЫ. Пока они рисовались только на плане, поверх чертежа,
        # это была терпимая мелочь; в объёме тот же слой ложится на местность и перекрывает её
        # кружками и линиями всюду, где есть дорога, — а нужен он в разборе сети, а не постоянно.
        # Предупреждение о разрыве сети от тумблера теперь НЕ зависит: оно в сведениях и
        # показывается всегда, иначе выключенный показ молча гасил бы и диагностику.
        self.show_graph = tk.BooleanVar(value=False)
        # Дома в объёме — КОРОБКАМИ. У строения в бою высоты нет (оно перекрывает обзор целиком
        # независимо от неё), так что высота показная — как срез грунта под картой. Но плоское
        # чёрное пятно на местности не читается как дом, и село с высоты выглядит асфальтом.
        self.show_houses3d = tk.BooleanVar(value=True)
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
        self.sel = []                   # выделенные фигуры: их может быть несколько
        self.selected = None            # главная из выделенных — для узлов и сведений
        self.clipboard = []             # скопированные фигуры
        self._box = None                # рамка выделения, пока её тянут
        self.snap_on = tk.BooleanVar(value=True)
        self.mode3d = tk.BooleanVar(value=False)
        self.cam = view3d.Camera()
        self._cam_drag = None
        self._photo = None              # картинка tk на холсте, переиспользуется между кадрами
        self._fps_item = None           # счётчик кадров поверх холста
        self._fps_ms = None             # сглаженная цена кадра
        self._t_frame = 0.0
        self._last_view_key = None      # вид на прошлом кадре: сравнением ловим движение
        self._hq_job = None             # отложенная чистовая перерисовка
        self._fly = set()               # какие стрелки зажаты: непрерывный полёт в объёме
        self._fly_v = [0.0, 0.0]        # скорость полёта, м/с — с разгоном и выбегом
        self._fly_job = None
        self._zoom_goal = None          # (множитель, экранная точка) — плавное приближение
        self._fly_t = 0.0
        self._gl = None                 # объёмный вид на видеокарте, заводится по первой нужде
        self._tiles = None              # кусочная местность: подробность растёт при подлёте
        self._dirty_rect = None         # что поменялось с прошлой сборки кусков, в метрах
        self._tile_job = None           # ожидание досчёта тайлов в фоне
        self._tiles_built = -1
        self._gl_off = False            # видеокарты нет — больше не пробуем, рисуем программно
        self._warp_cache = {}           # готовые подложки плана по ключу (вид, версия карты)
        self._prev_type = None          # чем рисовали до переключения на дом
        self._draft = None              # незакрытая фигура: список точек
        self._saved_at = (self.doc.version, self.doc.hversion)   # на чём сохранились
        self._hover = None              # где курсор на земле: тянущаяся за ним нить
        self._hover_job = None
        self._ruler = None              # два конца линейки, в метрах
        self._box3d = None              # рамка выделения в объёме, в ПИКСЕЛЯХ экрана
        self._vision = None             # (точка, маска видимости, клетка, угол окна)
        self._vis_key = None            # чем помечена залитая маска: чтобы не заливать зря
        self._vis_tm = None             # собранный кусок местности под видимость, с ключом
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
        self.canvas.bind("<B3-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-3>", self.on_release)
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        side = ttk.Frame(self, width=352)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        nb = self.nb = ttk.Notebook(side)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._tab_draw(nb)
        self._tab_view(nb)
        self._tab_file(nb)
        self.status = ttk.Label(side, text="", anchor="w", wraplength=336, justify="left",
                                style="Status.TLabel")
        self.status.pack(fill="x", padx=8, pady=(0, 8))
        self.tool.trace_add("write", lambda *_: self._tool_changed())

        self._bound = []
        for seq, fn in (
                *[(k, lambda v=v: self._set_type(v)) for k, v in zip("12345", SHAPE_TYPES)],
                ("6", lambda: (self.tool.set("relief_poly"), self._tool_changed())),
                *[(k, lambda v=v: self.tool.set(v))
                  for k, v in zip("qwertyu", [t[1] for t in self.TOOLS])],
                ("<Control-z>", self.do_undo), ("<Control-y>", self.do_redo),
                ("<Control-c>", self.copy_selected), ("<Control-v>", self.paste_clipboard),
                ("<Control-d>", self.duplicate_selected), ("<Control-a>", self.select_all),
                ("<Control-s>", self.save_map), ("<F5>", self.remeasure),
                ("<Delete>", self.delete_selected), ("<Escape>", self.cancel_draft),
                ("<Return>", self.finish_draft),
                ("<plus>", lambda: self.zoom_by(1.25)), ("<minus>", lambda: self.zoom_by(0.8)),
                ("0", self.fit_view),
                ("<KeyPress-space>", lambda: setattr(self, "_panning", True)),
                ("<KeyRelease-space>", lambda: setattr(self, "_panning", False)),
                *[(f"<KeyPress-{k}>", lambda d=d: self._fly_key(d, True))
                  for k, d in (("Up", "f"), ("Down", "b"), ("Left", "l"), ("Right", "r"),
                               ("w", "f"), ("s", "b"), ("a", "l"), ("d", "r"))],
                *[(f"<KeyRelease-{k}>", lambda d=d: self._fly_key(d, False))
                  for k, d in (("Up", "f"), ("Down", "b"), ("Left", "l"), ("Right", "r"),
                               ("w", "f"), ("s", "b"), ("a", "l"), ("d", "r"))]):
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

        # РЕЛЬЕФ отдельным блоком, а не шестой плашкой зонирования: он не материал, за клетку
        # не спорит, фигурой не остаётся и лежит в другом файле. Одинаковый вид у разных по
        # природе действий — та самая ложь интерфейса, из-за которой ищешь нарисованный холм в
        # списке фигур и не находишь.
        ttk.Separator(f).pack(fill="x", pady=9)
        Switch(f, "прилипание к узлам", "точка садится на ближайший узел соседней фигуры",
               self.snap_on).pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Label(f, text="выделение: Shift — добавить, рамкой по пустому месту — обвести, "
                          "Ctrl+A — всё, Ctrl+C/V — копировать, Ctrl+D — размножить",
                  style="Dim.TLabel", wraplength=310, justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(f, text="РЕЛЬЕФ  [6]", style="Head.TLabel").pack(anchor="w")
        rtools = self.relief_tools = ttk.Frame(f)
        rtools.pack(anchor="w", fill="x", pady=(4, 2))
        for label, val in (("печать полигоном", "relief_poly"), ("печать линией", "relief_line")):
            ttk.Radiobutton(rtools, text=label, value=val, variable=self.tool,
                            style="Toolbutton", width=17,
                            command=self._tool_changed).pack(side="left", padx=2)

        self.relief_box = ttk.Frame(f)
        ttk.Label(self.relief_box, text="ВЫСОТА", style="Head.TLabel").pack(anchor="w", pady=(6, 2))
        hrow = ttk.Frame(self.relief_box)
        hrow.pack(anchor="w", fill="x")
        ttk.Entry(hrow, textvariable=self.relief_h, width=7).pack(side="left")
        ttk.Label(hrow, text="м  (минус — вниз)").pack(side="left", padx=4)
        rgrid = ttk.Frame(self.relief_box)
        rgrid.pack(anchor="w", pady=(4, 0))
        for i, (name, h) in enumerate(RELIEF_PRESETS):
            ttk.Button(rgrid, text=f"{name} {h:+d}", width=13,
                       command=lambda h=h: self._set_relief(h)).grid(
                row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
        wrow = ttk.Frame(self.relief_box)
        wrow.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Label(wrow, text="ширина гряды").pack(side="left")
        ttk.Entry(wrow, textvariable=self.relief_width, width=6).pack(side="left", padx=(4, 0))
        ttk.Label(wrow, text="м  (для печати линией)").pack(side="left", padx=3)
        srow = ttk.Frame(self.relief_box)
        srow.pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Label(srow, text="склон").pack(side="left")
        ttk.Entry(srow, textvariable=self.relief_slope, width=6).pack(side="left", padx=(4, 0))
        ttk.Label(srow, text="м").pack(side="left", padx=3)
        for name, v in (("обрыв", 15.0), ("крутой", 60.0), ("пологий", 300.0)):
            ttk.Button(srow, text=name, width=7,
                       command=lambda v=v: self.relief_slope.set(v)).pack(side="left", padx=2)
        Switch(self.relief_box, "выровнять до отметки", "плато, терраса, дно ручья",
               self.relief_abs).pack(anchor="w", fill="x", pady=(6, 0))
        ttk.Label(self.relief_box,
                  text="форма вдавливается в карту высот и фигурой не остаётся:"
                       " высота у карты одна, растровая",
                  style="Dim.TLabel", wraplength=320, justify="left").pack(anchor="w", pady=(4, 0))

        self.bridge_box = ttk.Frame(f)
        ttk.Label(self.bridge_box, text="ПЕРЕПРАВА", style="Head.TLabel").pack(anchor="w",
                                                                               pady=(6, 2))
        Switch(self.bridge_box, "размеры по дороге и реке",
               "иначе берутся числами ниже", self.bridge_auto).pack(anchor="w",
                                                                   fill="x")
        Switch(self.bridge_box, "брод, а не мост",
               "проходим, но медленно: 0.45 пешим, 0.3 технике",
               self.bridge_ford).pack(anchor="w", fill="x", pady=(4, 0))
        brow = ttk.Frame(self.bridge_box)
        brow.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Label(brow, text="длина").pack(side="left")
        ttk.Entry(brow, textvariable=self.bridge_len, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(brow, text="ширина").pack(side="left")
        ttk.Entry(brow, textvariable=self.bridge_wid, width=6).pack(side="left", padx=4)

        self.tool_box = ttk.Frame(f)          # панель линейки и видимости
        Switch(self.tool_box, "линейка кругом", "радиус и кольца дальностей",
               self.ruler_circle, self.draw).pack(anchor="w", fill="x", pady=(6, 0))
        Switch(self.tool_box, "оставлять линейку", "не гасить замер после отпускания",
               self.ruler_keep, self.draw).pack(anchor="w", fill="x", pady=(6, 0))
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
        Switch(self.line_box, "пунктиром", "", self.dash_on).pack(anchor="w", pady=(6, 0))
        drow = ttk.Frame(self.line_box)
        drow.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Entry(drow, textvariable=self.dash_len, width=5).pack(side="left", padx=(0, 3))
        ttk.Label(drow, text="штрих").pack(side="left")
        ttk.Entry(drow, textvariable=self.dash_gap, width=5).pack(side="left", padx=(10, 3))
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
        self.shape_info = tk.Text(f, height=12, width=34, font=("Consolas", 9), relief="flat",
                                  background=UI["field"], foreground=UI["text"],
                                  highlightthickness=0, padx=8, pady=6)
        self.shape_info.pack(fill="x")
        self._tool_changed()

    HINTS = {"select": "щелчок выбирает, перетаскивание двигает",
             "nodes": "тяни узлы выбранной фигуры; ПКМ по узлу — удалить, "
                      "по границе — добавить",
             "polygon": "ЛКМ — точка, двойной щелчок / ПКМ / Enter — замкнуть, Esc — отмена",
             "line": "ЛКМ — точка, ширина задаётся полем ниже",
             "relief_poly": "печать холма или котловины: обводишь площадь, форма вдавливается"
                            " в карту высот и фигурой не остаётся",
             "relief_line": "печать гряды или лощины: ведёшь линию, ширина и склон — полями ниже",
             "building": "щелчок ставит дом заданного размера, протяжка доворачивает",
             "crossing": "щелчок ставит мост, протяжка доворачивает; размеры — авто или числами",
             "ruler": "протяни — покажет расстояние; галка «кругом» меряет радиусом",
             "vision": "щелчок — что видно из этой точки настоящей моделью линии огня"}

    # Какое зонирование имеет смысл при каком инструменте. Дом — всегда застройка, переправа
    # сама решает, что пробивает, а линейке и видимости тип не нужен вовсе. Раньше выбор висел
    # независимо, и «вода» при инструменте «дом» выглядела так, будто сейчас поставится пруд.
    TOOL_TYPES = {"polygon": SHAPE_TYPES, "line": SHAPE_TYPES, "building": ("building",)}
    RELIEF_TOOLS = ("relief_poly", "relief_line")     # печать формы в карту высот

    @staticmethod
    def draft_kind(tool):
        """Каким набором точек набирается фигура этим инструментом. Печати рельефа рисуются так
        же, как полигон и линия, но кладутся не в вектор, а в карту высот."""
        return {"polygon": "polygon", "relief_poly": "polygon",
                "line": "line", "relief_line": "line"}.get(tool)

    def _sync_relief_box(self):
        want = self.tool.get() in self.RELIEF_TOOLS
        if want:
            # строго под своими кнопками: в конец панели блок не влезает — там уже список фигур,
            # и tk просто не показывает то, чему не хватило места
            self.relief_box.pack(anchor="w", fill="x", after=self.relief_tools)
        else:
            self.relief_box.pack_forget()

    def _set_relief(self, h):
        self.relief_h.set(float(h))
        self.status.config(text=f"высота: {h:+.0f} м")

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
        self.relief_box.pack_forget()
        self._sync_relief_box()
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

    def _tab_view(self, nb):
        """Вид — своя вкладка между рисованием и файлом.

        Раньше он висел хвостом под инструментами: чтобы включить объём, надо было
        проскроллить мимо всего, чем рисуют, и граница между «нарисовать» и «посмотреть»
        пропадала.

        Срез грунта отсюда убран совсем. Это не выбор пользователя: без него объёмная
        карта висит в пустоте плоским листом и по ней не прочесть, где низина, а где
        насыпь. Галку, которая всегда стоит, держать в панели незачем."""
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="вид")

        ttk.Label(f, text="КАМЕРА", style="Head.TLabel").pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(5, 0))
        for label, cmd, w in (("−", lambda: self.zoom_by(0.8), 4),
                              ("+", lambda: self.zoom_by(1.25), 4),
                              ("вписать", self.fit_view, 9),
                              ("↺", lambda: self.rotate_view(90), 4),
                              ("↻", lambda: self.rotate_view(-90), 4)):
            ttk.Button(row, text=label, width=w, command=cmd).pack(side="left", padx=(0, 3))
        ttk.Label(f, text="колесо — зум · СКМ или пробел — тащить · в объёме стрелки"
                          " и WASD — лететь, ПКМ — вращать",
                  style="Dim.TLabel", wraplength=310, justify="left").pack(anchor="w",
                                                                          pady=(6, 0))

        ttk.Separator(f).pack(fill="x", pady=10)
        ttk.Label(f, text="ПРОСТРАНСТВО", style="Head.TLabel").pack(anchor="w", pady=(0, 6))
        Switch(f, "объёмный вид", "камера в пространстве вместо плана сверху",
               self.mode3d, self._toggle_3d).pack(fill="x", pady=3)

        ttk.Label(f, text="СТИЛЬ МЕСТНОСТИ (в объёме)", style="Head.TLabel").pack(
            anchor="w", pady=(12, 4))
        srow = ttk.Frame(f)
        srow.pack(anchor="w", fill="x")
        for label, val in (("живой", "vector"), ("клетки боя", "cells"), ("топо", "topo")):
            ttk.Radiobutton(srow, text=label, value=val, variable=self.map_style,
                            style="Toolbutton", width=11,
                            command=self.draw).pack(side="left", padx=2)
        ttk.Label(f, text="топо — как на карте 1:25000: горизонтали вместо тени, "
                          "по ним видно седловину и обратный скат",
                  style="Dim.TLabel", wraplength=310, justify="left").pack(anchor="w",
                                                                          pady=(4, 0))

        ttk.Separator(f).pack(fill="x", pady=10)
        ttk.Label(f, text="СЛОИ ПОВЕРХ КАРТЫ", style="Head.TLabel").pack(anchor="w",
                                                                        pady=(0, 6))
        for text, hint, var, cmd in (
                ("отмывка рельефа", "тень по склонам: где круто, там темнее",
                 self.show_shade, self.draw),
                ("узлы дорог", "перекрёстки и разрывы — то, чем ходит поиск пути",
                 self.show_graph, self._changed),
                ("дома объёмом", "коробки вместо плоских пятен (высота показная)",
                 self.show_houses3d, self.draw),
                ("сетка полей", "клетки, которыми карту читает бой",
                 self.show_grid, self.draw),
                ("счётчик кадров", "цена отрисовки в миллисекундах",
                 self.show_fps, self.draw)):
            Switch(f, text, hint, var, cmd).pack(fill="x", pady=3)

        ttk.Separator(f).pack(fill="x", pady=10)
        ttk.Label(f, text="F5 — пересчитать замер местности",
                  style="Dim.TLabel", wraplength=310, justify="left").pack(anchor="w")

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
        ttk.Button(f, text="←  в главное меню", command=self.back_to_menu).pack(fill="x")
        ttk.Label(f, text="список карт и создание новой; несохранённое спросит",
                  style="Dim.TLabel", wraplength=310, justify="left").pack(anchor="w", pady=(3, 0))
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
        if hasattr(self, "relief_box"):
            self._sync_relief_box()      # НЕ _tool_changed: тот сам зовёт _set_type -> рекурсия

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
        if self.mode3d.get():
            # Колесо задаёт ЦЕЛЬ, а не двигает камеру рывком: приближение доезжает за пару
            # десятых секунды, и глаз успевает связать «было» с «стало».
            f = 0.82 if ev.delta > 0 else 1 / 0.82
            base = self._zoom_goal[0] if self._zoom_goal else self.cam.dist
            h0 = self._zoom_goal[3] if self._zoom_goal else self._ground_h()
            self._zoom_goal = (float(np.clip(base * f, 60.0, 60000.0)), ev.x, ev.y, h0)
            self._fly_tick(start=True)
            return
        self.zoom_by(1.15 if ev.delta > 0 else 1 / 1.15, anchor=(ev.x, ev.y))

    # --- отрисовка

    def _S(self, p):
        return self.view.to_screen(self.W, self.H, p[0], p[1])

    # --- непрерывный полёт камеры
    #
    # Ощущение размера карты даёт не число в метрах, а ВРЕМЯ в пути. Прыжками мышью его не
    # получить: карта дёргается, и десять километров ощущаются как две с половиной. Поэтому
    # камера летит непрерывно, с разгоном и выбегом, а скорость растёт с высотой — как в
    # варгеймах: у земли ползёшь, сверху проносишься.

    FLY_SPEED = 0.55                    # доля удаления камеры в секунду
    FLY_ACCEL = 7.0                     # разгон и торможение, 1/с

    def _fly_key(self, d, down):
        if not self.mode3d.get():
            return
        if down:
            self._fly.add(d)
            self._fly_tick(start=True)
        else:
            self._fly.discard(d)

    def _fly_tick(self, start=False):
        """Шаг полёта. Сам себя перезапускает, пока есть скорость или незакрытое приближение;
        в покое таймер гаснет, чтобы не жечь кадры впустую."""
        if start and self._fly_job is not None:
            return
        now = time.perf_counter()
        dt = min(0.05, max(0.001, now - self._fly_t)) if self._fly_t else 1 / 60.0
        self._fly_t = now
        self._fly_job = None
        if not self.mode3d.get():
            self._fly.clear()
            self._fly_v = [0.0, 0.0]
            return

        want_f = (1.0 if "f" in self._fly else 0.0) - (1.0 if "b" in self._fly else 0.0)
        want_r = (1.0 if "r" in self._fly else 0.0) - (1.0 if "l" in self._fly else 0.0)
        speed = self.FLY_SPEED * self.cam.dist
        k = min(1.0, self.FLY_ACCEL * dt)
        self._fly_v[0] += (want_f * speed - self._fly_v[0]) * k
        self._fly_v[1] += (want_r * speed - self._fly_v[1]) * k

        moved = False
        if abs(self._fly_v[0]) + abs(self._fly_v[1]) > speed * 0.01:
            yaw = math.radians(self.cam.yaw)
            # вперёд — туда, куда смотрит камера, спроецировано на землю
            fx, fy = math.sin(yaw), math.cos(yaw)
            self.cam.tx += (self._fly_v[0] * fx + self._fly_v[1] * fy) * dt
            self.cam.ty += (self._fly_v[0] * fy - self._fly_v[1] * fx) * dt
            self.cam.clamp()
            moved = True
        else:
            self._fly_v = [0.0, 0.0]

        if self._zoom_goal is not None:
            target, sx, sy, h0 = self._zoom_goal
            ratio = target / max(self.cam.dist, 1e-6)
            step = ratio ** min(1.0, 12.0 * dt)          # доезжаем по доле оставшегося пути
            # Высоту земли берём ТУ ЖЕ, что при первом щелчке: она пересчитывается по точке, на
            # которую смотрит камера, а та во время наезда едет — и точка под курсором уползала.
            self.cam.zoom_at(step, sx, sy, (self.W, self.H), h0)
            if abs(math.log(max(target / max(self.cam.dist, 1e-6), 1e-6))) < 0.01:
                self._zoom_goal = None
            else:
                self._zoom_goal = (target, sx, sy, h0)
            moved = True

        # Тик планируется ПОСЛЕ отрисовки нарочно. Пробовал наоборот, чтобы сон шёл внахлёст
        # с кадром: частота выросла, но таймер начинает отсчёт в момент постановки, за время
        # кадра успевает истечь, и следующий тик идёт сразу. Цикл переставал спать, dt между
        # тиками начинал плавать вместе с ценой кадра — а наезд колесом считает долю
        # оставшегося пути ЗА ШАГ (ratio ** 12*dt), и на плавающем dt он идёт рывками.
        if moved:
            self.draw_3d()
        if self._fly or abs(self._fly_v[0]) + abs(self._fly_v[1]) > 0 or self._zoom_goal:
            self._fly_job = self.after(16, self._fly_tick)

    def _toggle_3d(self):
        self.cam.tx, self.cam.ty = self.doc.size_m[0] / 2, self.doc.size_m[1] / 2
        self.cam.dist = max(self.doc.size_m) * 1.9   # чтобы карта целиком помещалась в кадр
        self.cam.bounds = tuple(self.doc.size_m)
        self.cam.auto = True             # наклон ведётся за удалением, как в варгеймах
        self.cam.bias = 0.0
        self.cam.clamp()
        self.draw()
        self.status.config(text="объём: ЛКМ — тащить землю, Shift+ЛКМ — рамка выделения, "
                                "колесо — приблизить к курсору, ПКМ — повернуть и наклонить"
                           if self.mode3d.get() else "план")

    def _height_cell(self):
        """Клетка ОБЩЕГО поля высот. Кратна клетке мелкого уровня со степенью двойки нарочно:
        тогда сетки всех уровней подробности садятся на одни и те же узлы высоты, и стык уровней
        получается точным, без щелей."""
        k = 0
        while max(self.doc.size_m) / (tilemod.BASE_CELL * 2 ** k) > 400:
            k += 1
        return tilemod.BASE_CELL * (2 ** k)

    def _ground_h(self):
        """Высота земли под точкой, вокруг которой ходит камера. По ней хватается земля при
        перетаскивании и приближении, и на ней же лежит сама точка — иначе вблизи на холмистой
        карте камера уходит под грунт, а курсор промахивается мимо места."""
        return self.cam.tz

    def _sync_cam_ground(self):
        """Посадить точку обзора на землю. Во время протяжки НЕ трогаем: земля, за которую
        схватились, должна оставаться под курсором, а меняющаяся высота её бы уводила."""
        if self._cam_drag:
            return
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        if h is None:
            self.cam.tz = 0.0
            return
        gx = int(np.clip(self.cam.tx / cell, 0, h.shape[0] - 1))
        gy = int(np.clip(self.cam.ty / cell, 0, h.shape[1] - 1))
        self.cam.tz = float(h[gx, gy])

    def destroy(self):
        """Закрыли карту — остановить фоновый счёт тайлов: он держит копию вектора и поток."""
        if self._tiles is not None:
            self._tiles.stop()
            self._tiles = None
        # Полёт и наведение сюда добавлены после того, как check_gui поймал на закрытии
        # «invalid command name ..._fly_tick»: отложенный тик стрелял по уже разрушенному
        # виджету. Ошибка была и раньше, просто с задержкой в 16 мс попадала в это окно редко.
        for job in (self._hq_job, self._tile_job, self._fly_job, self._hover_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:                    # noqa: BLE001
                    pass
        self._hq_job = self._tile_job = self._fly_job = self._hover_job = None
        return super(EditorFrame, self).destroy()

    def pick_ground(self, sx, sy, steps=192):
        """Точка местности под пикселем в объёмном виде — с учётом рельефа.

        Плоскости мало: `Camera.ground_at` пересекает горизонталь на высоте цели, и на склоне
        промах доходит до сотен метров — курсор стоит на гребне, а точка ставится за ним. Поэтому
        идём лучом и ищем, где он ныряет под землю, потом уточняем делением пополам.

        Шагаем ПО ЭКРАНУ равномерно (по обратной величине глубины), а не по метрам: у горизонта
        один пиксель стоит километров, и равномерный шаг по земле там просто перепрыгнул бы всё.
        """
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        cam = self.cam
        near = cam.ground_at(sx, sy, (self.W, self.H), cam.tz)
        if near is None:
            return None
        eye = np.array([cam.tx - cam.dist * math.cos(math.radians(cam.pitch))
                        * math.sin(math.radians(cam.yaw)),
                        cam.ty - cam.dist * math.cos(math.radians(cam.pitch))
                        * math.cos(math.radians(cam.yaw))], dtype=np.float64)
        # Высоты в кадре преувеличены VSCALE, и луч обязан идти в ТОМ ЖЕ мире: иначе точка
        # уезжает от курсора тем сильнее, чем круче склон (замер: 8 пикселей в среднем).
        vs = view3d.VSCALE
        eye_z = cam.tz * vs + cam.dist * math.sin(math.radians(cam.pitch))

        def ground_at_xy(x, y):
            """Высота билинейно — так же, как её читает шейдер. По ближайшей клетке луч ловил
            бы ступеньку, и точка уезжала на десяток пикселей от курсора."""
            if h is None:
                return 0.0
            u = min(max(x / cell - 0.5, 0.0), h.shape[0] - 1.0)
            v = min(max(y / cell - 0.5, 0.0), h.shape[1] - 1.0)
            i0, j0 = int(u), int(v)
            i1, j1 = min(i0 + 1, h.shape[0] - 1), min(j0 + 1, h.shape[1] - 1)
            fu, fv = u - i0, v - j0
            a = h[i0, j0] * (1 - fu) + h[i1, j0] * fu
            b = h[i0, j1] * (1 - fu) + h[i1, j1] * fu
            return float(a * (1 - fv) + b * fv) * vs

        # луч от камеры через точку на плоскости цели
        d = np.array([near[0] - eye[0], near[1] - eye[1]], dtype=np.float64)
        dz = cam.tz * vs - eye_z
        far = max(2.0, 3.0 * max(self.doc.size_m))
        span = math.hypot(d[0], d[1], ) or 1.0
        t_end = far / span
        prev_t, prev_gap = 0.0, eye_z - ground_at_xy(eye[0], eye[1])
        for i in range(1, steps + 1):
            t = t_end * (i / steps) ** 2          # мельче вблизи, крупнее к горизонту
            x, y = eye[0] + d[0] * t, eye[1] + d[1] * t
            gap = (eye_z + dz * t) - ground_at_xy(x, y)
            if gap <= 0.0:                        # луч ушёл под землю между prev_t и t
                lo, hi = prev_t, t
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    xm, ym = eye[0] + d[0] * mid, eye[1] + d[1] * mid
                    if (eye_z + dz * mid) - ground_at_xy(xm, ym) <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                t = 0.5 * (lo + hi)
                return float(eye[0] + d[0] * t), float(eye[1] + d[1] * t)
            prev_t, prev_gap = t, gap
        return None                                # луч ушёл в небо

    def _tile_grid(self):
        if self._tiles is None or self._tiles.size_m != tuple(map(float, self.doc.size_m)):
            if self._tiles is not None:
                self._tiles.stop()
            self._tiles = tilemod.TileGrid(self.doc.vec, self.doc.size_m)
        self._tiles.set_mode(self.map_style.get())
        # ВО ВРЕМЯ ПРОТЯЖКИ куски не пересчитываем. Каждое движение мыши меняет версию карты, а
        # версия сбрасывает задетые куски: замер показал, что одно движение выбрасывало все 39
        # готовых, и кадр протяжки стоил 304 мс против 53 спокойного. Пометка задетой округи не
        # спасает — у крупного полигона она в пол-экрана.
        # Куда фигура едет, показывает ОБВОДКА поверх кадра, а местность догонит на отпускании:
        # там вызывается _changed, который и метит округу. Тем же правилом живёт граф дорог.
        #
        # Признак — ИМЕННО протяжка фигуры, а не общий «занят». Первая попытка взяла готовый
        # _graph_busy, а он считает занятостью и движение камеры — куски же от камеры зависят
        # напрямую, и они переставали обновляться совсем: набор покраснел сразу четырьмя
        # проверками, включая «печать рельефа сбрасывает округу» с нулём пересчитанных кусков.
        if self._drag and self._drag[0] in ("move", "node"):
            return self._tiles
        # Версия составная и ОДНА НА ВСЕ СТИЛИ: картинка куска зависит от фигур, а в топостиле
        # ещё и от высоты. Пока версия зависела от текущего стиля, переключение стиля само
        # читалось как смена карты — все готовые куски выбрасывались, и на их месте оставался
        # верхний уровень: вся карта в 512 точек, то есть грубый растр на несколько секунд.
        self._tiles.set_source(self.doc.vec, (self.doc.version, self.doc.hversion),
                               self._dirty_rect)
        self._dirty_rect = None
        return self._tiles

    def _tile_draws(self, grid, keys, cell_h):
        """Что рисовать: для каждого выбранного тайла — своя картинка, а пока она считается,
        кусок картинки предка. Дыр в кадре не бывает: предок есть всегда, самый грубый уровень
        считается сразу при входе в объём."""
        draws, used, waiting = [], [], 0
        for key in keys:
            surf, gen = grid.get_gen(key)
            src, mode = key, grid.mode
            if surf is None:
                waiting += 1
                src, surf, gen = grid.ready_ancestor(key)
                # Своего куска ещё нет. Предок годится, пока он немногим грубее; а вот
                # растянуть верхний уровень (вся карта в 512 точек — 20 м на точку на
                # театре) значит показать грубый растр вместо карты. Это и было видно при
                # переключении стиля: прежние куски выбрасывались все разом, и оставался
                # только верхний. Лучше тот же квадрат в прежнем стиле — цвет не тот, зато
                # подробность своя.
                if src is None or src[0] - key[0] >= 2:
                    alt, asurf, agen = grid.other_style(key)
                    if alt is not None:
                        src, surf, gen, mode = key, asurf, agen, alt
                if src is None:
                    continue
            # В ключе картинки на видеокарте — и вид, и НОМЕР ПЕРЕСЧЁТА. Без номера
            # правка на месте не показывалась: тайл пересчитывался, а видеокарта
            # оставляла прежнюю картинку, потому что ключ не менялся. Фигура появлялась
            # только после переключения стиля, когда ключ менялся целиком.
            gkey = (mode, gen) + src
            if not self._gl.has_tile(gkey):
                self._gl.upload_tile(gkey, surf)
            x0, y0, span = grid.rect(key)
            # у края карты тайл обрезаем: иначе местность свисает за срез грунта губой
            spx = min(span, self.doc.size_m[0] - x0)
            spy = min(span, self.doc.size_m[1] - y0)
            ax0, ay0, aspan = grid.rect(src)
            am = grid.margin_m(src[0], mode)
            full = aspan + 2 * am
            segs = int(np.clip(round(span / cell_h), 8, 128))
            draws.append((x0, y0, spx, spy, (x0 - ax0 + am) / full, (y0 - ay0 + am) / full,
                          spx / full, spy / full, gkey, segs))
            used.append(gkey)
        return draws, used, waiting

    def _building_boxes(self):
        """Дома для объёмного вида: (центр, стороны, поворот, низ, верх) в метрах.

        Низ берётся по САМОМУ НИЗКОМУ из четырёх углов следа и опускается ещё на метр. Иначе на
        склоне дом, посаженный на высоту центра, одним углом висит в воздухе, а другим тонет;
        так он просто врезан в подъём, как настоящий.

        Высота — из h_m фигуры, если её задали, иначе условная по следу."""
        if not self.show_houses3d.get():
            return []
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        out = []
        for sh in self.doc.shapes:
            if sh["kind"] != "building":
                continue
            cx, cy, bw, bh, ang = sh["rect_m"]
            pts = vectormap._rect_points(cx, cy, bw, bh, ang)
            if h is None:
                z0 = 0.0
            else:
                zs = [float(h[int(np.clip(q[0] / cell, 0, h.shape[0] - 1)),
                              int(np.clip(q[1] / cell, 0, h.shape[1] - 1))]) for q in pts]
                z0 = min(zs)
            hgt = float(sh.get("h_m") or view3d.building_height_m(bw, bh))
            out.append((cx, cy, bw, bh, ang, z0 - 1.0, z0 + hgt))
        return out

    def _gl_frame(self):
        """Кадр видеокартой или None, если её нет. Контекст, поле высот и картинки тайлов живут
        между кадрами: заводить их заново — 400 мс на контекст, от выигрыша не осталось бы
        ничего."""
        if self._gl_off:
            return None
        try:
            if self._gl is None:
                import view3d_gpu
                self._gl = view3d_gpu.GLView((self.W, self.H))
            cell_h = self._height_cell()
            height = self.doc.height_m(cell_h)
            self._gl.set_height(height, cell_h, key=(self.doc.hversion, cell_h))
            self._gl.set_buildings(self._building_boxes(),
                                   key=(self.doc.version, self.doc.hversion,
                                        self.show_houses3d.get()))
            grid = self._tile_grid()
            top = (grid.levels - 1, 0, 0)
            if grid.get(top) is None:
                grid.build_now(top)      # чтобы первый кадр не был пустым
            keys = grid.select(self.cam, (self.W, self.H))
            grid.request(keys)
            draws, used, waiting = self._tile_draws(grid, keys, cell_h)
            # векторная картинка вчетверо тяжелее клеточной (544x544 против 136x136), поэтому
            # и держим их меньше: штука это около 1.2 МБ памяти видеокарты с мипмапами.
            # Предел поднят следом за MAX_TILES: в кадре у земли бывает под 180 кусков, и при
            # прежних 80 хранилище чистилось КАЖДЫЙ кадр, выбрасывая то, что сейчас же снова
            # понадобится, — а заливка картинки это построение мипмапов на 544x544
            self._gl.keep_tiles(used, limit=tilemod.MAX_TILES if grid.mode == "vector" else 400)
            if self._vision:
                # маску пересобираем ТОЛЬКО когда она поменялась: она не зависит от камеры, а
                # заливка текстуры на каждый кадр съедала просмотр целиком
                key_v = (id(self._vision[1]), float(self.vision_r.get()))
                if key_v != getattr(self, "_vis_key", None):
                    self._vis_key = key_v
                    self._gl.set_overlay(*self._vision_rgba())
            else:
                self._gl.clear_overlay()
            img = self._gl.frame(self.cam, draws, size=(self.W, self.H),
                                 ground=True)
            self._await_tiles(waiting)
            if self._draft or self.draft_kind(self.tool.get()) or self._vision:
                return img                   # строку состояния занимает наведение, черновик
                #                              или доля просматриваемого круга
            lvl = min(k[0] for k in keys) if keys else 0
            fine = (grid.span(lvl) / tilemod.PAINT_PX if grid.mode == "vector"
                    else grid.cell(lvl))
            self.status.config(
                text="объём · высота камеры %.0f м · %s вблизи %.1f м · кусков %d%s · "
                     "ЛКМ тащит землю, колесо приближает к курсору, ПКМ вращает"
                     % (self.cam.dist * math.sin(math.radians(self.cam.pitch)),
                        "точка" if grid.mode == "vector" else "клетка боя",
                        fine, len(keys), " (%d считается)" % waiting if waiting else ""))
            return img
        except Exception as ex:                      # нет moderngl, нет GL, чужой драйвер
            self._gl, self._gl_off = None, True
            print("объём рисуем программно: видеокарта недоступна (%s)" % ex)
            return None

    def _await_tiles(self, waiting):
        """Тайлы считаются в фоне; как досчитаются — перерисовать. Опрашиваем редко: кадр стоит
        миллисекунды, а тайл десятки, и частый опрос только мешает счёту."""
        if self._tile_job is not None:
            self.after_cancel(self._tile_job)
            self._tile_job = None
        if waiting:
            self._tile_job = self.after(140, self._tick_tiles)

    def _tick_tiles(self):
        self._tile_job = None
        if not self.winfo_exists() or not self.mode3d.get():
            return
        built = self._tiles.built if self._tiles else 0
        if built != self._tiles_built:
            self._tiles_built = built
            self.draw_3d()
        else:
            self._tile_job = self.after(140, self._tick_tiles)

    def draw_3d(self, coarse=None):
        """Местность под свободной камерой.

        Кадр берёт видеокарта: местность кусками, у каждого своя подробность — вблизи клетка пять
        метров, вдали крупнее. Программный рисовальщик остаётся запасным, но кусков не умеет: он
        рисует всю карту одной грубой сеткой (замер: 42x42 — 22 мс, 85x85 — 66, 170x170 — 282)."""
        self._t_frame = time.perf_counter()
        self._sync_cam_ground()
        img = self._gl_frame()
        if img is None:
            cell = float(self.cell_var.get()) if hasattr(self, "cell_var") else self.doc.cell_m
            surface, cell_used = self.doc.surface(cell)
            height = self.doc.height_m(cell_used)
            if coarse is None:
                coarse = 3 if self._cam_drag else max(1, int(surface.shape[0] / 110))
            # надвыборка только на неподвижном кадре: вчетверо дороже, зато без лесенки по краям
            img = view3d.render(surface, height, cell_used, self.cam, (self.W, self.H),
                                coarse=coarse, ss=1 if self._cam_drag else 2,
                                ground=True, buildings=self._building_boxes())
        self._shapes_overlay_3d(img)
        self._draft_overlay_3d(img)
        self._ruler_overlay_3d(img)
        self._vision_rings_3d(img)
        self._blit(img)

    def _rotate_placed(self, kind, p):
        """Доворот только что поставленного дома или моста протяжкой — то же, что на плане."""
        p0, si = self._drag[1]
        dx, dy = float(p[0]) - p0[0], float(p[1]) - p0[1]
        if math.hypot(dx, dy) <= 1:
            return
        ang = round(math.degrees(math.atan2(dy, dx)), 1)
        sh = self.doc.shapes[si]
        if kind == "building_rot":
            sh["rect_m"][4] = ang
        else:
            sh["angle_deg"] = ang
        self._changed(dirty=self._shape_rect(sh))

    def _vision_rings_3d(self, img):
        """Кольца дальностей в объёме. Они ЛОЖАТСЯ НА ЗЕМЛЮ, а не рисуются плоским кругом на
        экране: на склоне плоский круг соврал бы про расстояние вдвое. У нас автомат бьёт до
        195 м, пулемёт до 405, достают все до 1005 — без колец «видно далеко» ничего не значит."""
        if not self._vision:
            return img
        center = self._vision[0]
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        d = ImageDraw.Draw(img)
        acc = (255, 226, 150)
        ang = np.linspace(0.0, 2.0 * math.pi, 97, dtype=np.float32)
        for r_m in (200, 400, 600, 800, 1000):
            if r_m > float(self.vision_r.get()) + 1:
                break
            xs = center[0] + r_m * np.cos(ang)
            ys = center[1] + r_m * np.sin(ang)
            if h is None:
                zs = np.zeros_like(xs)
            else:
                gx = np.clip((xs / cell).astype(np.int32), 0, h.shape[0] - 1)
                gy = np.clip((ys / cell).astype(np.int32), 0, h.shape[1] - 1)
                zs = h[gx, gy]
            sx, sy, zc = view3d.project(xs.astype(np.float32), ys.astype(np.float32),
                                        (zs * view3d.VSCALE).astype(np.float32), self.cam,
                                        (self.W, self.H))
            good = zc > 1.0
            for i in range(len(ang) - 1):
                if good[i] and good[i + 1]:
                    d.line([(sx[i], sy[i]), (sx[i + 1], sy[i + 1])],
                           fill=acc, width=2 if r_m % 400 == 0 else 1)
            j = int(np.argmax(good)) if good.any() else 0
            if good[j]:
                d.text((sx[j] + 4, sy[j] - 14), "%d м" % r_m, fill=acc, font=self.font_small)
        cz = 0.0 if h is None else float(h[int(np.clip(center[0] / cell, 0, h.shape[0] - 1)),
                                           int(np.clip(center[1] / cell, 0, h.shape[1] - 1))])
        sx, sy, zc = view3d.project(np.array([center[0]], dtype=np.float32),
                                    np.array([center[1]], dtype=np.float32),
                                    np.array([cz * view3d.VSCALE], dtype=np.float32),
                                    self.cam, (self.W, self.H))
        if zc[0] > 1.0:
            d.ellipse([sx[0] - 5, sy[0] - 5, sx[0] + 5, sy[0] + 5], fill=(255, 255, 255))
        return img

    def _ruler_overlay_3d(self, img):
        """Линейка в объёме: концы лежат на земле, длина считается по КАРТЕ, а не по экрану.
        Заодно показывает перепад высот между концами — на склоне это половина дела."""
        if not self._ruler:
            return img
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        pts = np.asarray(self._ruler, dtype=np.float32)
        if h is None:
            zs = np.zeros(len(pts), dtype=np.float32)
        else:
            gx = np.clip((pts[:, 0] / cell).astype(np.int32), 0, h.shape[0] - 1)
            gy = np.clip((pts[:, 1] / cell).astype(np.int32), 0, h.shape[1] - 1)
            zs = h[gx, gy]
        sx, sy, zc = view3d.project(pts[:, 0], pts[:, 1], zs * view3d.VSCALE, self.cam,
                                    (self.W, self.H))
        if not (zc > 1.0).all():
            return img
        d = ImageDraw.Draw(img)
        acc = (224, 180, 92)
        d.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=acc, width=2)
        for i in (0, 1):
            d.ellipse([sx[i] - 4, sy[i] - 4, sx[i] + 4, sy[i] + 4], fill=acc,
                      outline=(240, 240, 240))
        dist = float(np.hypot(pts[1, 0] - pts[0, 0], pts[1, 1] - pts[0, 1]))
        txt = "%.0f м   перепад %+.0f м" % (dist, float(zs[1] - zs[0]))
        d.text((0.5 * (sx[0] + sx[1]) + 8, 0.5 * (sy[0] + sy[1]) - 16), txt, fill=acc,
               font=self.font)
        return img

    def _project_ground(self, pts):
        """Мировые точки -> экранные, посаженные НА РЕЛЬЕФ. Возвращает точки и признак «перед
        камерой»: у того, что за спиной, проекция врёт, и такие отрезки рисовать нельзя.

        Высота берётся из того же поля и той же камерой, что рисует местность, — иначе обводка
        разъезжается с фигурой, которую обводит, и тем сильнее, чем круче склон."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if not len(pts):
            return [], np.zeros(0, dtype=bool)
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        if h is None:
            zs = np.zeros(len(pts), dtype=np.float32)
        else:
            gx = np.clip((pts[:, 0] / cell).astype(np.int32), 0, h.shape[0] - 1)
            gy = np.clip((pts[:, 1] / cell).astype(np.int32), 0, h.shape[1] - 1)
            zs = h[gx, gy]
        sx, sy, zc = view3d.project(pts[:, 0], pts[:, 1], zs * view3d.VSCALE, self.cam,
                                    (self.W, self.H))
        return [(float(sx[i]), float(sy[i])) for i in range(len(pts))], np.asarray(zc) > 1.0

    @staticmethod
    def _draw_runs(d, scr, vis, fill, width=2, closed=False):
        """Нарисовать ломаную ПРОБЕГАМИ подряд идущих видимых точек — по одному вызову на пробег.

        Сначала здесь был вызов на каждый ОТРЕЗОК, и на театре это выходило под две тысячи
        вызовов PIL за кадр: граф дорог стоил 12 мс, больше, чем вся остальная отрисовка. План
        всегда рисовал ломаную одним вызовом, и объёму незачем иначе. Разрыв нужен только там,
        где точка ушла за спину камеры: её проекция врёт, и соединять через неё нельзя."""
        order = list(range(len(scr))) + ([0] if closed and len(scr) > 2 else [])
        run = []
        for i in order:
            if vis[i]:
                run.append(scr[i])
            else:
                if len(run) > 1:
                    d.line(run, fill=fill, width=width, joint="curve")
                run = []
        if len(run) > 1:
            d.line(run, fill=fill, width=width, joint="curve")

    DRAPE_M = 25.0             # шаг дробления обводки по земле: чаще — дороже, реже — видно грань

    def _drape(self, pts, closed=False, step_m=None):
        """Разбить ломаную на частые точки ПО ЗЕМЛЕ и вернуть их вместе с разметкой.

        Зачем: сажать на рельеф одни вершины мало. Между ними отрезок рисуется прямым по экрану,
        и сторона, переваливающая через бугор, прочерчивается СКВОЗЬ него — обводка идёт не там,
        где проходит граница. Заметно это не только на глаз: ПКМ по границе ставит узел, и
        целиться приходится в нарисованную линию, а она лежала мимо земли.

        Разметка — на каждую точку пара (номер ребра, доля вдоль него). По ней потом считается,
        куда вставлять узел, и вставка попадает в ИСХОДНОЕ ребро, а не в дробление."""
        step = float(step_m or self.DRAPE_M)
        m = len(pts)
        if m < 2:
            return list(pts), [(0, 0.0)] * m
        out, mark = [], []
        last = m if closed else m - 1
        for k in range(last):
            a, b = pts[k], pts[(k + 1) % m]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            # верх ограничен нарочно: на карте 10 км сторона в километры иначе дала бы сотни
            # точек на ребро, а разницы против сорока уже не видно
            parts = max(1, min(48, int(L / step)))
            for j in range(parts):
                t = j / float(parts)
                out.append([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
                mark.append((k, t))
        if not closed:
            out.append(list(pts[-1]))
            mark.append((m - 2, 1.0))
        else:
            out.append(list(pts[0]))
            mark.append((0, 0.0))
        return out, mark

    def _line_3d(self, d, pts, fill, width=2, closed=False):
        """Ломаная по земле — дроблёная, поэтому лежит по склону всей длиной."""
        dense, _ = self._drape(pts, closed)
        scr, ok = self._project_ground(dense)
        if len(scr) >= 2:
            # замыкание уже в самих точках: дробление вернуло первую точку последней
            self._draw_runs(d, scr, ok, fill, width, closed=False)
        return scr, ok

    def _shapes_overlay_3d(self, img):
        """Выделенное, узлы и граф дорог поверх объёмного кадра.

        Пока этого слоя не было, в объёме нельзя было ни увидеть, что выделено, ни найти узел —
        а значит и править там было нечего: правка идёт ПО УЗЛАМ, а узлы не показывались.
        Рисуется ровно то же, что на плане, и теми же цветами: разойдись они, объём начал бы
        врать про то, с чем работаешь."""
        if not self.sel and not self.show_graph.get() and self._box3d is None:
            return img
        d = ImageDraw.Draw(img, "RGBA")

        if self.show_graph.get() and not self._graph_busy():
            g, main = self._graph_main()
            # Проецируем ВСЁ разом: участки и узлы одной пачкой. По отдельному вызову на участок
            # это была девяносто одна заводка numpy на кадр там, где хватает одной
            paths = [e.get("path") or [g["nodes"][e["a"]], g["nodes"][e["b"]]]
                     for e in g["edges"]]
            flat = [q for path in paths for q in path]
            scr, ok = self._project_ground(flat + list(g["nodes"]))
            k = 0
            for e, path in zip(g["edges"], paths):
                col = (240, 210, 130, 150) if e["a"] in main else (240, 120, 110, 180)
                self._draw_runs(d, scr[k:k + len(path)], ok[k:k + len(path)], col, 2)
                k += len(path)
            for i in range(len(g["nodes"])):
                if not ok[k + i]:
                    continue
                x, y = scr[k + i]
                col = (255, 225, 150, 235) if i in main else (255, 120, 110, 240)
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col, outline=(30, 28, 24, 255))

        for i in self.sel:
            if not 0 <= i < len(self.doc.shapes):
                continue
            sh = self.doc.shapes[i]
            outline = _lighten(TYPE_COLOR.get(sh.get("type", "field"), (220, 220, 220)))
            head = i == self.selected
            col = (*outline, 255 if head else 170)
            if sh["kind"] == "polygon":
                self._line_3d(d, sh["points"], col, width=2 if head else 1, closed=True)
            elif sh["kind"] == "line":
                self._line_3d(d, sh["points"], col, width=2 if head else 1)
            elif sh["kind"] == "building":
                cx, cy, bw, bh, ang = sh["rect_m"]
                self._line_3d(d, vectormap._rect_points(cx, cy, bw, bh, ang), col,
                              width=2 if head else 1, closed=True)
            elif sh["kind"] == "crossing":
                c, length, width_m, ang = vectormap.crossing_geom(self.doc.vec, sh)
                self._line_3d(d, vectormap._rect_points(c[0], c[1], length,
                                                        max(width_m, 4.0), ang),
                              (255, 230, 160, 255), width=2 if head else 1, closed=True)

        # Ручки узлов — только у ГЛАВНОЙ выделенной и только под инструментом «узлы», как на
        # плане: показывать их у всей пачки значит засыпать кадр квадратиками
        if self._box3d is not None:
            x0, y0, x1, y1 = self._box3d
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                        outline=(255, 240, 150, 200))
        if self.selected is not None and self.tool.get() == "nodes":
            sh = self.doc.shapes[self.selected]
            if sh["kind"] in ("polygon", "line"):
                scr, ok = self._project_ground(sh["points"])
                for k, (x, y) in enumerate(scr):
                    if ok[k]:
                        d.rectangle([x - 4, y - 4, x + 4, y + 4], fill=(255, 240, 150),
                                    outline=(40, 40, 40))
        return img

    def _draft_overlay_3d(self, img):
        """Незакрытая фигура поверх объёмного кадра.

        Точки лежат на ЗЕМЛЕ и проецируются той же камерой, что рисует местность, — поэтому
        черновик не плавает над рельефом, а лежит по склону. Отрезки, у которых конец за спиной
        камеры, пропускаем: их проекция врёт."""
        if not self._draft:
            return img
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        pts = np.asarray(self._draft, dtype=np.float32)
        if h is None:
            zs = np.zeros(len(pts), dtype=np.float32)
        else:
            gx = np.clip((pts[:, 0] / cell).astype(np.int32), 0, h.shape[0] - 1)
            gy = np.clip((pts[:, 1] / cell).astype(np.int32), 0, h.shape[1] - 1)
            zs = h[gx, gy]
        sx, sy, zc = view3d.project(pts[:, 0], pts[:, 1], zs * view3d.VSCALE, self.cam,
                                    (self.W, self.H))
        relief = self.tool.get() in self.RELIEF_TOOLS
        col = TYPE_COLOR["relief" if relief else self.shape_type.get()]
        light = _lighten(col)
        poly = self.draft_kind(self.tool.get()) == "polygon"
        ok = zc > 1.0
        scr = [(float(sx[i]), float(sy[i])) for i in range(len(pts))]
        # Нить до курсора: пока фигура не замкнута, видно, куда пойдёт следующая сторона.
        tip = None
        if self._hover is not None:
            hx, hy = self._hover
            hz = 0.0 if h is None else float(
                h[int(np.clip(hx / cell, 0, h.shape[0] - 1)),
                  int(np.clip(hy / cell, 0, h.shape[1] - 1))])
            tsx, tsy, tzc = view3d.project(np.array([hx], dtype=np.float32),
                                           np.array([hy], dtype=np.float32),
                                           np.array([hz * view3d.VSCALE], dtype=np.float32),
                                           self.cam, (self.W, self.H))
            if tzc[0] > 1.0:
                tip = (float(tsx[0]), float(tsy[0]))
        # Заливка будущей фигуры полупрозрачно: по одной обводке не понять, что закрашивается.
        ring = scr + ([tip] if tip else [])
        if poly and len(ring) >= 3 and all(ok) :
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).polygon(ring, fill=(*light, 80))
            img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"),
                      (0, 0))
        d = ImageDraw.Draw(img)

        def stroke(a, b, w=3, dark=True):
            """Светлая жила в тёмной оплётке: одним цветом чертёж теряется и на лесу, и на
            снегу топостиля."""
            if dark:
                d.line([a, b], fill=(20, 22, 26), width=w + 2)
            d.line([a, b], fill=light, width=w)

        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1]:
                stroke(scr[i], scr[i + 1])
        if tip and ok[-1]:
            stroke(scr[-1], tip, 2)
        if poly and len(pts) >= 2 and ok[0] and ok[-1]:
            d.line([tip or scr[-1], scr[0]], fill=light, width=1)
        for i in range(len(pts)):
            if ok[i]:
                d.ellipse([sx[i] - 5, sy[i] - 5, sx[i] + 5, sy[i] + 5],
                          fill=col, outline=(245, 245, 245), width=2)
        return img

    def draw(self):
        if self.W < 4 or self.H < 4:
            return
        if self.mode3d.get():
            return self.draw_3d()
        self._t_frame = time.perf_counter()
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

        stale = bool(self._drag or self._cam_drag)
        # Вид сдвинулся с прошлого кадра — значит его сейчас тащат, крутят или приближают.
        # Такой кадр рисуем быстрым пересчётом (полноэкранный поворот отмывки: 1.5 мс против 44),
        # а через 140 мс после остановки перерисовываем начисто.
        vkey = self._view_key()
        moving = self._last_view_key is not None and vkey != self._last_view_key
        self._last_view_key = vkey
        if moving:
            if self._hq_job is not None:
                self.after_cancel(self._hq_job)
            self._hq_job = self.after(140, self._redraw_hq)

        if self.show_shade.get() and self.map_style.get() == "topo":
            got = self._cached_warp("topo", (self.doc._height_key, moving),
                                    lambda: _split(self._contour_overlay(fast=moving)))
            if got is not None:
                out.paste(got[0], (0, 0), got[1])
        elif self.show_shade.get():
            got = self._cached_warp("shade", (self.doc._surface_key, moving),
                                    lambda: _split(self._shade_overlay(stale_ok=stale,
                                                                       fast=moving)))
            if got is not None:
                out.paste(got[0], (0, 0), got[1])

        if self.show_grid.get():
            # во время протяжки берём ПРОШЛУЮ сетку: пересборка полей стоит 67 мс на боевой
            # карте и около секунды на театре, а кадр без неё — 26 мс. Пересоберём на отпускании
            surface, cell = self.doc.surface(stale_ok=stale)

            def build_grid():
                # прозрачность кладём сразу в палитру, а не правим альфу готового кадра: точка
                # за точкой по полутора мегапикселям стоит впятеро дороже самого поворота
                a = int(255 * self.grid_alpha)
                lut = np.zeros((max(P.TILE_COLORS) + 1, 4), dtype=np.uint8)
                for k, c in P.TILE_COLORS.items():
                    lut[k] = (*c, a)
                img = Image.fromarray(np.ascontiguousarray(lut[surface].transpose(1, 0, 2)[::-1]))
                w2i = _translate(0, surface.shape[1]) @ _scale(1 / cell, -1 / cell)
                coeffs = (w2i @ self.view.inv(self.W, self.H))[:2].flatten()
                return _split(img.transform((self.W, self.H), Image.AFFINE, tuple(coeffs),
                                            resample=Image.NEAREST, fillcolor=(0, 0, 0, 0)))

            got = self._cached_warp("grid", (self.doc._surface_key, round(self.grid_alpha, 3)),
                                    build_grid)
            if got is not None:
                out.paste(got[0], (0, 0), got[1])

        if self._vision:
            got = self._vision_overlay()
            if got is not None:
                warped, at = got
                out.paste(warped.convert("RGB"), at, warped.getchannel("A"))

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
        # paste с маской вместо alpha_composite: тот требует перевода всего кадра в RGBA и
        # обратно — два прохода по полутора мегапикселям ради того же результата
        out.paste(over, (0, 0), over)

        self._blit(out)

    def _blit(self, img):
        """Показать готовый кадр. Картинка tk переиспользуется: завести новую и пересоздать
        элемент холста стоит 8 мс — больше, чем сама отрисовка объёма на видеокарте."""
        if (self._photo is None or self._photo.width() != img.width
                or self._photo.height() != img.height):
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            self._fps_item = None        # старый счётчик удалён вместе со всем холстом
        else:
            self._photo.paste(img)
        self._show_fps()

    def _show_fps(self):
        """Цена кадра и сколько таких кадров в секунду. Считаем ОТРИСОВКУ, а не промежуток между
        кадрами: редактор рисует по событию, и в покое промежуток был бы секундами, хотя кадр
        стоит миллисекунды. Число сглажено — иначе на нём ничего не разобрать.

        Счётчик — элемент холста, а не текст в картинке: рисовать его в кадр значит платить за
        показ цены показа."""
        ms = (time.perf_counter() - self._t_frame) * 1000.0
        self._fps_ms = ms if self._fps_ms is None else 0.82 * self._fps_ms + 0.18 * ms
        if not self.show_fps.get():
            if self._fps_item is not None:
                self.canvas.delete(self._fps_item)
                self._fps_item = None
            return
        txt = "%.0f к/с · %.1f мс" % (1000.0 / max(self._fps_ms, 0.05), self._fps_ms)
        if self._fps_item is None:
            # верхний ЛЕВЫЙ угол: справа вверху стрелка на север, а внизу слева линейка
            self._fps_item = self.canvas.create_text(
                12, 9, anchor="nw", text=txt, fill="#e0b45c", font=("Consolas", 10, "bold"))
        else:
            self.canvas.coords(self._fps_item, 12, 9)
            self.canvas.itemconfigure(self._fps_item, text=txt)
        self.canvas.tag_raise(self._fps_item)

    def _view_key(self):
        return (round(self.view.cx, 4), round(self.view.cy, 4), round(self.view.zoom, 8),
                round(self.view.angle, 5), self.W, self.H)

    def _cached_warp(self, name, key, build):
        """Готовая подложка плана под текущий вид. Каждая — полноэкранное аффинное
        преобразование: отмывка 19 мс, сетка полей 7. Пока вид и карта не менялись, пересчитывать
        их незачем, а перерисовка кадра случается на каждое движение мыши — на выборе фигуры, на
        протяжке линейки, на наведении."""
        full = (self._view_key(),) + tuple(key)
        got = self._warp_cache.get(name)
        if got is not None and got[0] == full:
            return got[1]
        img = build()
        self._warp_cache[name] = (full, img)
        return img

    def _redraw_hq(self):
        self._hq_job = None
        if not self.winfo_exists():     # карту успели закрыть, пока ждали
            return
        self._last_view_key = None      # чтобы кадр не посчитался «в движении» ещё раз
        self.draw()

    def _contour_overlay(self, fast=False):
        """Горизонтали на плане вместо отмывки — ТЕ ЖЕ линии, что и в объёме.

        Отмывка показывает, ГДЕ склон; горизонтали отвечают, НА СКОЛЬКО МЕТРОВ. Для карты,
        по которой считают линии огня и уклоны, второе важнее: по ним сразу видно седловину,
        обратный скат и то, что гряда выше леса.

        Линии берутся из общего набора paint.Contours и рисуются прямо в экранных
        координатах. Раньше здесь был свой растр горизонталей, который потом растягивался
        аффинным преобразованием: на плане выходили одни линии, в объёме другие, и при
        увеличении растр расплывался в полосы. Один источник — одна линия везде."""
        h = self.doc.height_m()
        if h is None or not np.any(h):
            return None
        cell = self.doc._height_key[1] if self.doc._height_key else self.doc.cell_m
        step = paint.contour_step(float(h.max() - h.min()))
        cs = paint.contour_set(h, cell, step)
        mult = cs.mult_for(1.0 / max(self.view.zoom, 1e-6))
        self._contour_step = step * mult
        inv = self.view.inv(self.W, self.H)
        corners = np.array([[0, 0, 1], [self.W, 0, 1], [0, self.H, 1], [self.W, self.H, 1]],
                           dtype=np.float64) @ inv[:2].T
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        lines = cs.select(x0, y0, x1, y1, mult)
        if not lines:
            return None
        # На протяжке рисуем в один пиксель без сглаживания: полноэкранная маска вдвое
        # крупнее стоит миллисекунды, а линия на движущемся кадре всё равно смазана.
        ss = 1 if fast else 2
        m = self.view.matrix(self.W, self.H)
        masks = (Image.new("L", (self.W * ss, self.H * ss), 0),
                 Image.new("L", (self.W * ss, self.H * ss), 0))
        draw = (ImageDraw.Draw(masks[0]), ImageDraw.Draw(masks[1]))
        width = (max(1, ss), max(2, 2 * ss))
        for k, pl in lines:
            xy = (pl @ m[:2, :2].T + m[:2, 2]) * ss
            i = 1 if (k % 5 == 0) else 0
            draw[i].line(xy.ravel().tolist(), fill=255, width=width[i], joint="curve")
        thin = np.asarray(masks[0].resize((self.W, self.H), Image.BOX) if ss > 1
                          else masks[0], dtype=np.float32)
        index = np.asarray(masks[1].resize((self.W, self.H), Image.BOX) if ss > 1
                           else masks[1], dtype=np.float32)
        thin = thin * (1.0 - index / 255.0)          # утолщённая главнее тонкой
        rgba = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        for a, col, top in ((thin, paint.TOPO_CONTOUR, 210), (index, paint.TOPO_INDEX, 255)):
            k = (a / 255.0)[:, :, None]
            rgba[:, :, :3] = (rgba[:, :, :3] * (1 - k) + np.array(col) * k).astype(np.uint8)
            rgba[:, :, 3] = np.maximum(rgba[:, :, 3], (a * (top / 255.0)).astype(np.uint8))
        return Image.fromarray(rgba)

    def _shade_overlay(self, stale_ok=False, fast=False):
        """Отмывка: тень по склонам. На плане высота иначе не видна вообще — нарисовал гряду и
        не знаешь, вышла она холмом или ямой, пока не откроешь объём."""
        surface, cell = self.doc.surface(stale_ok=stale_ok)
        h = self.doc.height_m()
        if h is None or not np.any(h):
            return None
        gx = np.gradient(h, cell, axis=0)
        gy = np.gradient(h, cell, axis=1)
        nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
        sun = view3d.SUN                                  # одно солнце на план и на объём
        lam = np.clip((-gx * sun[0] - gy * sun[1] + sun[2]) * nz, 0.0, 1.4)
        v = np.clip((lam - 0.75) * 2.4, -1.0, 1.0)          # >0 склон к солнцу, <0 в тень
        rgba = np.zeros((*h.shape, 4), dtype=np.uint8)
        rgba[..., 0] = np.where(v > 0, 255, 20)
        rgba[..., 1] = np.where(v > 0, 240, 24)
        rgba[..., 2] = np.where(v > 0, 205, 34)
        rgba[..., 3] = (np.abs(v) * 120).astype(np.uint8)
        img = Image.fromarray(np.ascontiguousarray(rgba.transpose(1, 0, 2)[::-1]))
        gy_n = h.shape[1]
        w2i = _translate(0, gy_n) @ _scale(1 / cell, -1 / cell)
        coeffs = (w2i @ self.view.inv(self.W, self.H))[:2].flatten()
        return img.transform((self.W, self.H), Image.AFFINE, tuple(coeffs),
                             resample=Image.NEAREST if fast else Image.BILINEAR,
                             fillcolor=(0, 0, 0, 0))

    def _draw_shapes(self, d):
        """Фигуры — полупрозрачной заливкой с контуром. Сквозь них видно сетку: весь смысл
        показа сетки в том, чтобы замечать расхождение между тем, что нарисовано, и тем, что
        досталось бою (тонкая дорога, узкая лесополоса)."""
        order = {"water": 0, "forest": 1, "road": 2}
        idx = sorted(range(len(self.doc.shapes)),
                     key=lambda i: order.get(self.doc.shapes[i].get("type"), 9))
        for i in idx:
            sh = self.doc.shapes[i]
            sel = (i in self.sel)
            col = TYPE_COLOR.get(sh.get("type"), (200, 200, 200))
            outline = (255, 240, 150) if sel else tuple(min(255, int(c * 1.5) + 40) for c in col)
            relief = sh.get("type") == "relief"
            if relief:
                # рельеф — не материал: заливать его цветом местности нельзя, иначе он спорит
                # с лесом и дорогами. Показываем контуром, вверх тёплым, вниз холодным
                up = float(sh.get("h_m", 0)) >= 0
                col = (214, 170, 96) if up else (110, 150, 210)
                outline = (255, 240, 150) if sel else col
            if sh["kind"] == "polygon":
                pts = self.view.to_screen_many(self.W, self.H, sh["points"])
                if len(pts) >= 3:
                    if relief:
                        d.polygon(pts, fill=(*col, 34), outline=(*outline, 235))
                        d.text((pts[0][0] + 6, pts[0][1] - 8), f"{sh.get('h_m', 0):+.0f} м",
                               fill=(*outline, 240), font=self.font_small)
                    else:
                        d.polygon(pts, fill=(*col, 120), outline=(*outline, 230))
            elif sh["kind"] == "line":
                # пунктир рисуем ТЕМ ЖЕ разбиением, что уйдёт в сетку: иначе на чертеже полоса
                # сплошная, в бою дырявая, и понять, где прорехи, нельзя
                w = max(1, int(sh.get("width_m", 8) * self.view.zoom))
                for run in self._line_runs(sh):
                    pts = self.view.to_screen_many(self.W, self.H, run)
                    d.line(pts, fill=(*col, 55 if relief else 170), width=w, joint="curve")
                    if relief:
                        d.line(pts, fill=(*outline, 220), width=2)
                        d.text((pts[0][0] + 6, pts[0][1] - 8), f"{sh.get('h_m', 0):+.0f} м",
                               fill=(*outline, 240), font=self.font_small)
                    if sel:
                        d.line(pts, fill=(*outline, 255), width=1)
            elif sh["kind"] == "building":
                cx, cy, bw, bh, ang = sh["rect_m"]
                pts = self.view.to_screen_many(
                    self.W, self.H, vectormap._rect_points(cx, cy, bw, bh, ang))
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
        if self._box:
            (ax, ay), (bx, by) = self._box
            q = self.view.to_screen_many(self.W, self.H,
                                         [[ax, ay], [bx, ay], [bx, by], [ax, by]])
            d.line(q + [q[0]], fill=(255, 240, 150, 190), width=1)
        if self._draft:
            pts = self.view.to_screen_many(self.W, self.H, self._draft)
            col = TYPE_COLOR[self.shape_type.get()]
            light = _lighten(col)
            tip = (self.view.to_screen(self.W, self.H, *self._hover)
                   if self._hover is not None else None)
            if len(pts) > 1:
                d.line(pts, fill=(20, 22, 26), width=4)
                d.line(pts, fill=(*light, 255), width=2)
            if tip:                      # нить до курсора — та же, что и в объёме
                d.line([pts[-1], tip], fill=(*light, 255), width=1)
                if self.draft_kind(self.tool.get()) == "polygon" and len(pts) >= 2:
                    d.line([tip, pts[0]], fill=(*light, 160), width=1)
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

    def _graph_main(self):
        """Граф дорог и НОМЕРА УЗЛОВ ГЛАВНОЙ ЧАСТИ сети. Считается здесь, а не в рисовании,
        потому что рисуют это двое — план и объём, — а разрыв сети обязан выглядеть одинаково
        в обоих. Два раза написанный обход рано или поздно разъезжается.

        Помним по версии карты: пока фигуры не менялись, части сети те же. Без этого обход шёл
        на КАЖДЫЙ кадр объёма — а там кадры идут подряд при полёте камеры, и на карте не
        меняется ничего."""
        key = self.doc.version
        cached = getattr(self, "_graph_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
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
        self._graph_stats = (len(g["nodes"]), len(g["edges"]), len(comps), len(main))
        self._graph_cache = (key, g, set(main))
        return self._graph_cache[1], self._graph_cache[2]

    def _graph_busy(self):
        """Граф не рисуем, пока что-то ДВИЖЕТСЯ.

        На плане правило было заведено для протяжки фигуры: граф строится перебором пересечений,
        и на театре это доли секунды — заметный рывок на каждом кадре перетаскивания. В объёме к
        тому же добавляется цена показа: даже с общей проекцией и рисованием пробегами граф
        стоит 6 мс на кадр, а при летящей камере его всё равно не разглядеть. Камера
        остановилась — граф на месте."""
        if self._drag and self._drag[0] in ("move", "node", "building",
                                            "building_rot", "crossing_rot"):
            return True
        if self.mode3d.get():
            return bool(self._cam_drag) or bool(self._fly) or self._zoom_goal is not None
        return False

    def _draw_graph(self, d):
        """Перекрёстки и участки дорог. Смотреть на это стоит по двум причинам: перекрёсток —
        это ориентир «узел дороги», по которому отдаются приказы, и он же будущий узел маршрута.
        А ещё граф сразу показывает РАЗРЫВ сети: если кусок карты отрезан (обычно забыт мост),
        его узлы окрасятся красным, и это видно до всякого боя."""
        if self._graph_busy():
            return
        g, main = self._graph_main()

        for e in g["edges"]:
            # ведём ребро ПО ДОРОГЕ, а не хордой между узлами: на извилистой дороге хорда
            # прочерчивала прямую через полкарты и выглядела как «концы соединились сами»
            path = self.view.to_screen_many(
                self.W, self.H, e.get("path") or [g["nodes"][e["a"]], g["nodes"][e["b"]]])
            col = (240, 210, 130, 150) if e["a"] in main else (240, 120, 110, 180)
            if len(path) > 1:
                d.line(path, fill=col, width=2, joint="curve")
        for i, n in enumerate(g["nodes"]):
            x, y = self._S(n)
            col = (255, 225, 150, 235) if i in main else (255, 120, 110, 240)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col, outline=(30, 28, 24, 255))

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

    def _vision_alpha(self, v, center, cell, origin):
        """Прозрачность маски: тень тем плотнее, чем меньше видно, но ЗА КРУГОМ ничего нет.

        Без обрезки по кругу тень заливала весь прямоугольник окна, в котором считалась
        видимость, — на светлой топокарте это читалось как чёрный квадрат вокруг точки."""
        gx = (np.arange(v.shape[0], dtype=np.float32) + 0.5) * cell + origin[0] - center[0]
        gy = (np.arange(v.shape[1], dtype=np.float32) + 0.5) * cell + origin[1] - center[1]
        r = np.hypot(gx[:, None], gy[None, :])
        radius = float(self.vision_r.get())
        fade = np.clip((radius - r) / max(radius * 0.06, cell), 0.0, 1.0)   # мягкая кромка
        return ((46 + 120 * (1 - v)) * fade).astype(np.uint8)

    def _vision_rgba(self):
        """Маска видимости цветом: (py, px, 4), строка 0 — это y=0. Одна и та же раскраска идёт
        и на план (там её ещё поворачивают), и в объём (там её кладёт на местность шейдер)."""
        center, field, cell, origin = self._vision
        v = _smooth_mask(np.clip(field, 0.0, 1.0))
        rgba = np.zeros((*v.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255 - (243 * (1 - v)).astype(np.uint8)      # свет -> тёплый, тень -> синяя
        rgba[..., 1] = 226 - (212 * (1 - v)).astype(np.uint8)
        rgba[..., 2] = 150 - (124 * (1 - v)).astype(np.uint8)
        rgba[..., 3] = self._vision_alpha(v, center, cell, origin)
        return np.ascontiguousarray(rgba.transpose(1, 0, 2)), cell, origin

    def _vision_overlay(self):
        """Зона видимости поверх плана: картинка и то место экрана, куда её положить.

        Две вещи здесь ради скорости, и обе нужны — на театре кадр с видимостью стоил 155 мс,
        это шесть кадров в секунду под зажатой мышью.

        Первое: сглаживаем поле на СВОЕЙ сетке (170 на 170 чисел), а не растянутую вчетверо
        картинку. Ступеньку в полклетки на границе тени всё равно убирает поворот с билинейной
        выборкой, а размытие вчетверо большего растра стоило две трети времени.

        Второе: поворачиваем не весь холст, а прямоугольник вокруг круга видимости. Круг радиусом
        900 м занимает четверть экрана, а платили мы за всё поле."""
        center, field, cell, origin = self._vision
        v = _smooth_mask(np.clip(field, 0.0, 1.0))
        rgba = np.zeros((*v.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255 - (243 * (1 - v)).astype(np.uint8)      # свет -> тёплый, тень -> синяя
        rgba[..., 1] = 226 - (212 * (1 - v)).astype(np.uint8)
        rgba[..., 2] = 150 - (124 * (1 - v)).astype(np.uint8)
        rgba[..., 3] = self._vision_alpha(v, center, cell, origin)
        img = Image.fromarray(np.ascontiguousarray(rgba.transpose(1, 0, 2)[::-1]))

        r = float(self.vision_r.get()) * 1.05
        xs, ys = [], []
        for dx, dy in ((-r, -r), (r, -r), (r, r), (-r, r)):
            sx, sy = self._S((center[0] + dx, center[1] + dy))
            xs.append(sx)
            ys.append(sy)
        bx0 = max(0, int(min(xs)) - 2)
        by0 = max(0, int(min(ys)) - 2)
        bx1 = min(self.W, int(max(xs)) + 2)
        by1 = min(self.H, int(max(ys)) + 2)
        if bx1 <= bx0 or by1 <= by0:
            return None
        gy = v.shape[1]
        w2i = _translate(0, gy) @ _scale(1 / cell, -1 / cell) @ _translate(-origin[0], -origin[1])
        m = w2i @ self.view.inv(self.W, self.H) @ _translate(bx0, by0)
        return img.transform((bx1 - bx0, by1 - by0), Image.AFFINE, tuple(m[:2].flatten()),
                             resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0)), (bx0, by0)

    def _draw_vision(self, d):
        """Кольца дальностей поверх зоны видимости: у нас автомат бьёт до 195 м, пулемёт до 405,
        достают все до 1005 — без колец «видно далеко» ничего не значит."""
        center, mask, cell, _origin = self._vision
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

    def _select(self, idx, add=False):
        """Выделить фигуры. Главной считается последняя названная: узлы правятся у неё, и о ней
        же пишутся сведения — так «выбор» остаётся привычным, когда выделена одна фигура."""
        idx = [i for i in idx if 0 <= i < len(self.doc.shapes)]
        if add:
            keep = [i for i in self.sel if i not in idx]
            idx = keep + [i for i in idx if i not in self.sel]
        self.sel = idx
        self.selected = self.sel[-1] if self.sel else None

    def _toggle_sel(self, i):
        if i in self.sel:
            self.sel = [k for k in self.sel if k != i]
        else:
            self.sel = self.sel + [i]
        self.selected = self.sel[-1] if self.sel else None

    @staticmethod
    def _shift(ev):
        return bool(getattr(ev, "state", 0) & 0x0001)

    def _in_box(self, x0, y0, x1, y1):
        """Фигуры, задетые рамкой. По ГАБАРИТУ, а не по точному пересечению: рамкой обводят,
        чтобы захватить всё в углу карты, и требовать полного попадания здесь неудобно."""
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        out = []
        for i, sh in enumerate(self.doc.shapes):
            b = vectormap._shape_bounds(sh)
            if not (b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1):
                out.append(i)
        return out

    def _tol_m_3d(self, p, px=8.0):
        """Сколько метров приходится на px экранных пикселей ВОЗЛЕ точки p в объёме.

        На плане мерка одна на весь кадр — пикселей на метр это свойство вида. В объёме не так:
        у горизонта метр стоит доли пикселя, под ногами — десятки. Пока сюда брали зум ПЛАНА
        (а брали: _tol_m — это 8 / view.zoom), допуск не имел отношения к тому, что на экране:
        план, подогнанный под боевую карту, давал 18 м, под театр — все 70. Оттого в объёме
        прилипало всё подряд, что оказалось в этом радиусе.

        Меряем честно: проецируем саму точку и её соседей в метре к востоку и к северу той же
        камерой, что рисует местность, и смотрим, во сколько пикселей превратился метр."""
        cell = self._height_cell()
        h = self.doc.height_m(cell)
        xs = np.asarray([p[0], p[0] + 1.0, p[0]], dtype=np.float32)
        ys = np.asarray([p[1], p[1], p[1] + 1.0], dtype=np.float32)
        if h is None:
            zs = np.zeros(3, dtype=np.float32)
        else:
            gx = np.clip((xs / cell).astype(np.int32), 0, h.shape[0] - 1)
            gy = np.clip((ys / cell).astype(np.int32), 0, h.shape[1] - 1)
            zs = h[gx, gy]
        sx, sy, zc = view3d.project(xs, ys, zs * view3d.VSCALE, self.cam, (self.W, self.H))
        if zc[0] <= 1.0:                     # точка за спиной камеры — проекция врёт
            return self._tol_m(px)
        d = max(math.hypot(float(sx[1] - sx[0]), float(sy[1] - sy[0])),
                math.hypot(float(sx[2] - sx[0]), float(sy[2] - sy[0])))
        return px / d if d > 1e-6 else self._tol_m(px)

    def _snap(self, p, tol=None):
        """Притянуть точку к ближайшему УЗЛУ соседней фигуры. Без этого две дороги, нарисованные
        встык, расходятся на метр-полтора, и в графе дорог вместо перекрёстка выходит разрыв —
        а на глаз всё выглядит сошедшимся.

        Переправы в цели НЕ входят, хотя раньше входили. Замысел прилипания — свести встык концы
        того, что рисуют руками; а переправа ставится САМА, на пересечении дороги с рекой, то
        есть сидит ровно там, где чаще всего и рисуешь, и концом ничему не приходится. Тянуть
        точку к ней значит мешать в самом неудобном месте карты."""
        if not self.snap_on.get():
            return p, False
        tol = max(self._tol_m() if tol is None else tol, 4.0)
        best, bd = None, tol * tol
        pts = []
        for sh in self.doc.shapes:
            if sh["kind"] in ("polygon", "line"):
                pts += sh["points"]
        if self._draft:
            pts += [self._draft[0]]
        for q in pts:
            d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
            if d < bd:
                best, bd = q, d
        if best is None:
            return p, False
        return (float(best[0]), float(best[1])), True

    def _hit_shape(self, p, tol=None):
        """Что под курсором. Идём с конца: последние нарисованные лежат сверху.

        Допуск передаётся снаружи, потому что в объёме он другой: там метр у горизонта стоит
        доли пикселя, а под ногами десятки, и зум плана к делу не относится (см. _tol_m_3d)."""
        tol = self._tol_m() if tol is None else tol
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
        if self.mode3d.get():
            grab = self.cam.ground_at(ev.x, ev.y, (self.W, self.H), self._ground_h())
            self._cam_drag = ("pan", grab) if grab else None
            return
        self._drag = ("pan", (ev.x, ev.y, self.view.cx, self.view.cy))

    def on_press(self, ev):
        self.canvas.focus_set()
        if self.mode3d.get():
            self.canvas.focus_set()      # чтобы Enter и Esc доходили до обработчиков
            if self.tool.get() in ("vision", "ruler"):
                p = self.pick_ground(ev.x, ev.y)
                if p is None:
                    return
                if self.tool.get() == "vision":
                    self._drag = ("vision", None)
                    self._update_vision(np.array(p, dtype=np.float32))
                else:
                    self._drag = ("ruler", None)
                    self._ruler = [list(p), list(p)]
                    self.draw_3d()
                return
            if self.draft_kind(self.tool.get()) or self.tool.get() in ("building", "crossing"):
                # рисующий инструмент забирает левую кнопку себе; землю тащит средняя
                p = self.pick_ground(ev.x, ev.y)
                if p is None:
                    self.status.config(text="здесь неба, а не земли — точку ставить не на что")
                    return
                if self.tool.get() == "building":
                    return self._place_building(np.asarray(p, dtype=np.float32))
                if self.tool.get() == "crossing":
                    return self._place_crossing(np.asarray(p, dtype=np.float32))
                q, snapped = self._snap(p, tol=self._tol_m_3d(p))
                self._draft = (self._draft or []) + [[q[0], q[1]]]
                if snapped:
                    self.status.config(text="точка притянута к соседнему узлу")
                self.draw_3d()
                return
            if self.tool.get() in ("select", "nodes") and self._press_pick_3d(ev):
                return
            # Хватаем ЗЕМЛЮ под курсором, а не считаем приращения углов: только так точка
            # остаётся под пальцем. Выше горизонта земли нет — там перетаскивание не с чего
            # начинать, и кнопка работает как поворот.
            grab = self.cam.ground_at(ev.x, ev.y, (self.W, self.H), self._ground_h())
            self._cam_drag = (("pan", grab) if grab else
                              ("rot", (ev.x, ev.y, self.cam.yaw, self.cam.bias)))
            return
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
        if self.draft_kind(tool):
            q, snapped = self._snap(p)
            self._draft = (self._draft or []) + [[q[0], q[1]]]
            if snapped:
                self.status.config(text="точка притянута к соседнему узлу")
            self.draw()
            return
        if tool == "building":
            return self._place_building(p)
        if tool == "crossing":
            return self._place_crossing(p)
        if tool == "nodes":
            k = self._hit_node(p)
            if k is not None:
                self.push_undo()
                self._drag = ("node", (self.selected, k,
                                       self._crossing_links([self.selected])))
                return
        hit = self._hit_shape(p)
        if tool == "select":
            if hit is None:
                # по пустому месту тянется РАМКА: обвести квартал и подвинуть его целиком —
                # обычное дело, а по одной фигуре это тридцать перетаскиваний
                if not self._shift(ev):
                    self._select([])
                self._drag = ("box", (p, p))
                self._box = (p, p)
                self._refresh_shapes()
                self.draw()
                return
            if self._shift(ev):
                self._toggle_sel(hit)
                self._refresh_shapes()
                self.draw()
                return
            if hit not in self.sel:
                self._select([hit])
            self.push_undo()
            self._drag = ("move", (p, [(i, copy.deepcopy(self.doc.shapes[i]))
                                       for i in self.sel],
                                   self._crossing_links(self.sel)))
        else:
            self._select([] if hit is None else [hit])
        self._refresh_shapes()
        self.draw()

    def on_motion(self, ev):
        if self.mode3d.get():
            if self._drag and self._drag[0] == "box3d":
                self._box3d = (self._box3d[0], self._box3d[1], ev.x, ev.y)
                self.draw_3d()
                return
            if self._drag and self._drag[0] in ("vision", "ruler", "building_rot",
                                                "crossing_rot", "move", "node"):
                p = self.pick_ground(ev.x, ev.y)
                if p is not None:
                    kind = self._drag[0]
                    if kind == "vision":
                        self._update_vision(np.array(p, dtype=np.float32))
                    elif kind == "ruler":
                        self._ruler[1] = list(p)
                        self.draw_3d()
                    elif kind == "node":
                        self._apply_node(self._drag[1], np.asarray(p, dtype=np.float32))
                        self.draw_3d()
                    elif kind == "move":
                        self._apply_move(self._drag[1], np.asarray(p, dtype=np.float32))
                        self.draw_3d()
                    else:
                        self._rotate_placed(kind, np.asarray(p, dtype=np.float32))
                return
            if not self._cam_drag:
                return
            mode, data = self._cam_drag
            if mode == "pan":
                self.cam.drag_to(data, ev.x, ev.y, (self.W, self.H), self._ground_h())
            else:
                x0, y0, yaw0, bias0 = data
                self.cam.yaw = yaw0 + (ev.x - x0) * 0.35
                self.cam.bias = float(np.clip(bias0 - (ev.y - y0) * 0.25, -30.0, 30.0))
                self.cam.clamp()
            self.draw_3d()
            return
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
            self._apply_node(data, p)
            self.draw()
        elif mode == "box":
            self._box = (data[0], p)
            self.draw()
        elif mode == "move":
            self._apply_move(data, p)
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

    @staticmethod
    def _vector_occluders(tm):
        """Помехи вектора -> две сетки окна: «держит луч» и «сколько метров терпит».

        Отметка идёт по ЛЮБОМУ касанию клетки, а не по доле покрытия: у доли есть порог
        существования (дом мельче примерно 7x7 м при клетке 30 м не даёт ни одной), а вся затея
        с вектором была ради того, чтобы такой дом существовал. Считается один раз на окно и
        живёт вместе с ним, поэтому движение мыши бесплатно."""
        Gx, Gy = tm.grid.shape
        blocks = np.zeros((Gx, Gy), dtype=bool)
        lim = np.full((Gx, Gy), 1e9, dtype=np.float32)
        cx = (np.arange(Gx) + 0.5) * tm.cell
        cy = (np.arange(Gy) + 0.5) * tm.cell
        for _bnum, polys, see, _dem in tm.vterr.parts:
            for pl in polys:
                px = [q[0] for q in pl]
                py = [q[1] for q in pl]
                i0 = max(0, int(min(px) / tm.cell) - 1)
                i1 = min(Gx, int(max(px) / tm.cell) + 2)
                j0 = max(0, int(min(py) / tm.cell) - 1)
                j1 = min(Gy, int(max(py) / tm.cell) + 2)
                if i0 >= i1 or j0 >= j1:
                    continue                       # фигура вне окна
                X, Y = np.meshgrid(cx[i0:i1], cy[j0:j1], indexing="ij")
                m = vectormap.points_in_poly(X, Y, pl)
                if not m.any():
                    # мелкая фигура может не накрыть ни одного ЦЕНТРА клетки, а существовать
                    # обязана: отмечаем клетку, в которую попал её центр
                    mx = (min(px) + max(px)) * 0.5
                    my = (min(py) + max(py)) * 0.5
                    gi = int(np.clip(mx / tm.cell, 0, Gx - 1))
                    gj = int(np.clip(my / tm.cell, 0, Gy - 1))
                    blocks[gi, gj] = True
                    lim[gi, gj] = min(float(lim[gi, gj]), see)
                    continue
                sub_b = blocks[i0:i1, j0:j1]
                sub_l = lim[i0:i1, j0:j1]
                sub_b |= m
                np.minimum(sub_l, np.where(m, see, 1e9), out=sub_l)
        tm.f_blocks_vec = blocks
        tm.f_see_vec = lim
        return tm

    def _vision_terrain(self, p, radius_m, cell):
        """Кусок местности вокруг точки, собранный НАСТОЯЩИМ движком боя.

        По всей карте это делать нельзя: на театре сборка стоит полсекунды и повторяется после
        каждой правки. Лучу дальше радиуса ничего не нужно, поэтому берём окно с запасом и
        держим его, пока курсор не подошёл к краю, — тогда протяжка мыши бесплатна."""
        pad = radius_m + 600.0
        W, H = self.doc.size_m
        x0 = max(0.0, math.floor((p[0] - pad) / cell) * cell)
        y0 = max(0.0, math.floor((p[1] - pad) / cell) * cell)
        x1 = min(W, math.ceil((p[0] + pad) / cell) * cell)
        y1 = min(H, math.ceil((p[1] + pad) / cell) * cell)
        key = (self.doc.version, self.doc.hversion, round(cell, 3), x0, y0, x1, y1)
        got = self._vis_tm
        if got is not None and got[0] == key:
            return got[1], (x0, y0)
        # запас в 600 м: пока точка внутри прежнего окна с этим запасом, пересобирать нечего
        if got is not None and got[0][:3] == key[:3]:
            gx0, gy0, gx1, gy1 = got[0][3:]
            if (gx0 <= p[0] - radius_m and p[0] + radius_m <= gx1
                    and gy0 <= p[1] - radius_m and p[1] + radius_m <= gy1):
                return got[1], (gx0, gy0)
        surf = vectormap.surface_window(self.doc.vec, cell, x0, y0, x1 - x0, y1 - y0)
        fields = terrain.fields_from_surface(surf)
        hm = self.doc.vec.get("height")
        if hm is not None and np.any(hm["h"]):
            gx = (np.arange(surf.shape[0]) + 0.5) * cell + x0
            gy = (np.arange(surf.shape[1]) + 0.5) * cell + y0
            X, Y = np.meshgrid(gx, gy, indexing="ij")
            fields["height"] = vectormap.sample_height(hm, X, Y) / P.M_PER_UNIT
        tm = terrain.from_fields(surf, fields, cell / P.M_PER_UNIT)
        # тем же вектором, что и бой: просмотр прострелов не должен показывать своё
        tm.attach_vector(self.doc.vec, P.M_PER_UNIT, origin_m=(x0, y0))
        self._vector_occluders(tm)        # раскладываем помехи по клеткам окна один раз
        self._vis_tm = (key, tm)
        return tm, (x0, y0)

    def _update_vision(self, p):
        t0 = time.perf_counter()
        try:
            cell = float(self.cell_var.get())
        except (tk.TclError, ValueError, AttributeError):
            cell = self.doc.cell_m
        radius = float(self.vision_r.get())
        tm, origin = self._vision_terrain(p, radius, cell)
        local = np.array([p[0] - origin[0], p[1] - origin[1]], dtype=np.float32)
        field = viewshed(tm, local, radius, P.M_PER_UNIT)
        self._vision = (p, field, cell, origin)
        r_cells = float(self.vision_r.get()) / cell
        in_circle = max(1.0, math.pi * r_cells * r_cells)
        seen = float((field > 0.02).sum()) / in_circle
        self.status.config(text=f"просматривается {min(seen, 1.0) * 100:.0f}% круга радиусом "
                                f"{self.vision_r.get():.0f} м · счёт "
                                f"{(time.perf_counter() - t0) * 1000:.0f} мс")
        self.draw()

    def _finish_box(self, ev):
        """Рамка отпущена: берём всё, что она задела. Совсем мелкую рамку считаем промахом по
        пустому месту — иначе случайное дрожание руки при щелчке снимает выделение не туда."""
        p0 = self._box[0]
        p = self._world(ev)
        self._box = None
        if abs(p[0] - p0[0]) < self._tol_m() and abs(p[1] - p0[1]) < self._tol_m():
            self._select([])
        else:
            self._select(self._in_box(p0[0], p0[1], p[0], p[1]), add=self._shift(ev))
            self.status.config(text="выделено фигур: %d  ·  тащить — двигать, Del — удалить, "
                                    "Ctrl+C/Ctrl+V — копировать" % len(self.sel))
        self._refresh_shapes()
        self.draw()

    CROSS_TOL_M = 60.0         # тот же допуск, что у crossing_gaps: ближе этого — та же переправа

    def _crossing_links(self, moving):
        """К какой ПАРЕ линий привязана каждая переправа, которую заденет протяжка.

        Мост стоит там, где дорога пересекает реку, но фигурой он самостоятельный: точка и всё.
        Поэтому подвинутая дорога оставляла мост на прежнем месте — посреди поля, — а на новом
        пересечении пересчёт ставил ещё один. Получалось два моста, ни один не там, где нужно.

        Теперь перед протяжкой запоминаем, на пересечении каких линий сидит переправа, и во
        время движения пересчитываем её точку по этой паре. Пересчитывать надо именно так, а не
        двигать мост на то же смещение: когда едет одна дорога, пересечение ползёт ВДОЛЬ реки,
        и сдвиг у него совсем другой.

        Помним НОМЕРАМИ: во время протяжки список фигур не меняется, а имён у фигур нет. Пары
        берём только те, где хоть одна линия движется, — остальные пересечения никуда не денутся.
        Сама двигаемая переправа пропускается: её тащат руками, и спорить с рукой незачем."""
        moving = set(moving)
        roads = [i for i, sh in enumerate(self.doc.shapes)
                 if sh["kind"] == "line" and sh.get("type") == "road"]
        waters = [i for i, sh in enumerate(self.doc.shapes)
                  if sh["kind"] == "line" and sh.get("type") == "water"]
        links = []
        for ci, sh in enumerate(self.doc.shapes):
            if sh["kind"] != "crossing" or ci in moving:
                continue
            q = sh["point"]
            best = None
            for ri in roads:
                for wi in waters:
                    if ri not in moving and wi not in moving:
                        continue
                    for hp in vectormap.line_hits(self.doc.shapes[ri]["points"],
                                                  self.doc.shapes[wi]["points"]):
                        d = math.hypot(hp[0] - q[0], hp[1] - q[1])
                        if d < self.CROSS_TOL_M and (best is None or d < best[0]):
                            best = (d, ri, wi)
            if best is not None:
                links.append((ci, best[1], best[2]))
        return links

    def _reanchor_crossings(self, links):
        """Поставить переправы на их пересечения заново — после того, как линии поехали.

        Из нескольких пересечений пары берём БЛИЖАЙШЕЕ к прежнему месту моста: дорога может
        пересекать излучину дважды, и мост должен остаться на своём броде, а не прыгнуть на
        соседний. Во время протяжки точка едет понемногу, поэтому «ближайшее к прежнему» —
        это и есть «своё»."""
        for ci, ri, wi in links:
            if not (0 <= ci < len(self.doc.shapes) and 0 <= ri < len(self.doc.shapes)
                    and 0 <= wi < len(self.doc.shapes)):
                continue
            sh = self.doc.shapes[ci]
            if sh["kind"] != "crossing":
                continue
            hits = vectormap.line_hits(self.doc.shapes[ri]["points"],
                                       self.doc.shapes[wi]["points"])
            if not hits:
                continue                   # линии разошлись — мост оставляем где был
            q = sh["point"]
            hp = min(hits, key=lambda h: (h[0] - q[0]) ** 2 + (h[1] - q[1]) ** 2)
            sh["point"] = [round(float(hp[0]), 1), round(float(hp[1]), 1)]

    def _touch(self, rect):
        """Пометить округу к пересчёту кусков, объединяя с уже помеченным.

        Во время протяжки _changed НЕ зовут — он дорогой, — и округа оставалась пустой. А пустая
        означает «поменялось всё»: замер показал, что одно движение мыши выбрасывало ВСЕ готовые
        куски (39 из 39), и кадр протяжки стоил 304 мс против 53 спокойного. Теперь помечаем
        ровно то, что задели."""
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        cur = self._dirty_rect
        if cur is None:
            self._dirty_rect = (x0, y0, x1, y1)
        else:
            self._dirty_rect = (min(cur[0], x0), min(cur[1], y0),
                                max(cur[2], x1), max(cur[3], y1))

    def _apply_node(self, data, p):
        """Узел встал туда, куда показывает курсор. Общее для плана и объёма: в объёме p — это
        точка ЗЕМЛИ под курсором, и разница только в том, чем её получили."""
        si, k = data[0], data[1]
        was = self._shape_rect(self.doc.shapes[si])
        self.doc.shapes[si]["points"][k] = [round(p[0], 1), round(p[1], 1)]
        if len(data) > 2:
            self._reanchor_crossings(data[2])
        self._touch(was)
        self._touch(self._shape_rect(self.doc.shapes[si]))
        self.doc.bump()

    def _apply_move(self, data, p):
        """Перенос всей выделенной пачки на смещение курсора В МИРЕ.

        Смещение берётся по МИРОВЫМ точкам, а не по экранным пикселям, и в объёме это
        единственный правильный способ: при наклонённой камере и на склоне одинаковый сдвиг по
        экрану — это совершенно разный сдвиг по земле, и фигура уползала бы из-под курсора.
        Ровно та болезнь, от которой камеру в своё время перевели на обратную проекцию."""
        p0, group = data[0], data[1]
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        for i, _orig in group:
            self._touch(self._shape_rect(self.doc.shapes[i]))     # где фигура была
        for i, orig in group:
            sh = self.doc.shapes[i]
            if orig["kind"] in ("polygon", "line"):
                sh["points"] = [[round(q[0] + dx, 1), round(q[1] + dy, 1)]
                                for q in orig["points"]]
            elif orig["kind"] == "building":
                r = list(orig["rect_m"])
                sh["rect_m"] = [round(r[0] + dx, 1), round(r[1] + dy, 1), r[2], r[3], r[4]]
            elif orig["kind"] == "crossing":
                sh["point"] = [round(orig["point"][0] + dx, 1),
                               round(orig["point"][1] + dy, 1)]
        if len(data) > 2:
            self._reanchor_crossings(data[2])
        for i, _orig in group:
            self._touch(self._shape_rect(self.doc.shapes[i]))     # и куда переехала
        self.doc.bump()

    @staticmethod
    def _nearest_edge(pts, q, closed, tol):
        """Ближайшее РЕБРО ломаной к точке q: (номер начала ребра, доля вдоль него).

        Работает в любых координатах — в метрах на плане и в пикселях в объёме, — потому что
        считает одно и то же: проекцию точки на отрезок."""
        best, bd = None, float(tol) * float(tol)
        m = len(pts)
        if m < 2:
            return None
        for k in range(m if closed else m - 1):
            ax, ay = pts[k]
            bx, by = pts[(k + 1) % m]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            if L2 < 1e-9:
                continue
            t = ((q[0] - ax) * vx + (q[1] - ay) * vy) / L2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            dx, dy = ax + t * vx - q[0], ay + t * vy - q[1]
            d = dx * dx + dy * dy
            if d < bd:
                best, bd = (k, t), d
        return best

    def _insert_node(self, k, t):
        """Вставить узел на ребре k в доле t от его начала. Точка садится ровно НА ГРАНИЦУ:
        она считается по мировым концам ребра, а не по тому, куда пришёлся курсор, — иначе
        новый узел сдвигал бы контур в тот же миг, когда его ставят."""
        sh = self.doc.shapes[self.selected]
        pts = sh["points"]
        a, b = pts[k], pts[(k + 1) % len(pts)]
        q = [round(a[0] + (b[0] - a[0]) * t, 1), round(a[1] + (b[1] - a[1]) * t, 1)]
        self.push_undo()
        pts.insert(k + 1, q)
        self._changed()
        self.status.config(text="узел добавлен на границе")

    def _hit_node_3d(self, ev, px=9.0):
        """Узел под курсором В ОБЪЁМЕ — по ЭКРАНУ, а не по земле.

        Сначала было по земле, тем же путём, что и попадание в фигуру: точку под курсором даёт
        pick_ground. Замер показал, почему так нельзя: узел у бровки склона отстоял от
        возвращённой точки на СТО МЕТРОВ. Луч не ошибся — он честно упёрся в бугор, который
        закрывает узел от камеры. Для фигуры это верное поведение (щёлкнул по земле — получи
        то, что на ней лежит), а для ручки узла нет: ручка нарисована на экране поверх всего,
        видна поверх бугра, и попадание в неё обязано считаться там же, где она нарисована.

        Заодно исчезает вопрос о допуске: ручка на экране всегда одного размера, поэтому и
        мерка в пикселях, а не в метрах, которые у горизонта и под ногами разные."""
        if self.selected is None:
            return None
        sh = self.doc.shapes[self.selected]
        if sh["kind"] not in ("polygon", "line"):
            return None
        scr, vis = self._project_ground(sh["points"])
        best, bd = None, float(px) * float(px)
        for k, (x, y) in enumerate(scr):
            if not vis[k]:
                continue                       # узел за спиной камеры: проекция врёт
            d = (x - ev.x) ** 2 + (y - ev.y) ** 2
            if d < bd:
                best, bd = k, d
        return best

    def _shape_screen_bbox(self, sh):
        """Габарит фигуры НА ЭКРАНЕ или None, если её не видно.

        Рамка в объёме считается по экрану, а не по земле, и это единственный честный способ:
        прямоугольник на экране захватывает по земле трапецию, уходящую к горизонту, — что в
        неё попало, на глаз не предскажешь. По экрану же обводят ровно то, что видят.
        Габаритом, а не точным пересечением, — как и на плане: рамкой обводят, чтобы захватить."""
        if sh["kind"] in ("polygon", "line"):
            pts = sh["points"]
        elif sh["kind"] == "building":
            cx, cy, bw, bh, ang = sh["rect_m"]
            pts = vectormap._rect_points(cx, cy, bw, bh, ang)
        elif sh["kind"] == "crossing":
            c, length, width_m, ang = vectormap.crossing_geom(self.doc.vec, sh)
            pts = vectormap._rect_points(c[0], c[1], length, max(width_m, 4.0), ang)
        else:
            return None
        scr, vis = self._project_ground(pts)
        seen = [scr[i] for i in range(len(scr)) if vis[i]]
        if not seen:
            return None                    # вся фигура за спиной камеры
        xs = [q[0] for q in seen]
        ys = [q[1] for q in seen]
        return min(xs), min(ys), max(xs), max(ys)

    def _finish_box_3d(self, add):
        """Выделить всё, чей габарит на экране задела рамка."""
        x0, y0, x1, y1 = self._box3d
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        self._box3d = None
        if x1 - x0 < 3 and y1 - y0 < 3:
            return                         # это был щелчок, а не рамка
        got = []
        for i, sh in enumerate(self.doc.shapes):
            b = self._shape_screen_bbox(sh)
            if b is None or b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                continue
            got.append(i)
        self._select(got, add=add)
        self._refresh_shapes()
        self.status.config(text="обведено фигур: %d" % len(self.sel))

    def _press_pick_3d(self, ev):
        """Выбор, перенос и правка узлов в ОБЪЁМЕ. Возвращает True, если кнопку забрали себе.

        Попадание считается в МИРОВЫХ координатах — тем же кодом, что на плане, — а мировую
        точку под курсором даёт pick_ground: луч бьёт по рельефу, поэтому на склоне не
        промахивается. Снаружи пришлось передать только допуск: в объёме он от камеры.

        Рамки по пустому месту здесь НЕТ, и это решение, а не недоделка: прямоугольник на экране
        захватывает по земле трапецию, уходящую к горизонту, и что в неё попало — на глаз не
        предскажешь. По пустому месту снимаем выделение и отдаём кнопку камере: тащить землю —
        самое ожидаемое действие в пустоте."""
        # Узел ищем ДО pick_ground и по экрану: ручка видна поверх местности, и щелчок по ней
        # должен браться даже там, где под курсором вообще нет земли (небо за гребнем)
        if self.tool.get() == "nodes":
            k = self._hit_node_3d(ev)
            if k is not None:
                self.push_undo()
                self._drag = ("node", (self.selected, k,
                                       self._crossing_links([self.selected])))
                return True
        p = self.pick_ground(ev.x, ev.y)
        if p is None:
            return False
        p = np.asarray(p, dtype=np.float32)
        hit = self._hit_shape(p, tol=self._tol_m_3d(p))
        if hit is None:
            # Shift по пустому месту тянет РАМКУ. Простая ЛКМ по пустому оставлена камере:
            # тащить землю — самое частое движение в объёме, и отбирать у него кнопку нельзя.
            # Shift здесь же и означает «добавить к выделенному», так что рамка добавляет.
            if self._shift(ev):
                self._box3d = (ev.x, ev.y, ev.x, ev.y)
                self._drag = ("box3d", None)
                return True
            if self.sel:
                self._select([])
                self._refresh_shapes()
                self.draw_3d()
            return False
        if self._shift(ev):
            self._toggle_sel(hit)
        elif self.tool.get() == "select":
            if hit not in self.sel:
                self._select([hit])
            self.push_undo()
            self._drag = ("move", (p, [(i, copy.deepcopy(self.doc.shapes[i]))
                                       for i in self.sel],
                                   self._crossing_links(self.sel)))
        else:
            self._select([hit])
        self._refresh_shapes()
        self.draw_3d()
        return True

    def on_release(self, ev):
        if self.mode3d.get():
            self._cam_drag = None
            if self._drag and self._drag[0] == "vision":
                self._drag = None
                self._vision = None        # как на плане: показ живёт, пока держишь кнопку
            elif self._drag and self._drag[0] == "ruler":
                self._drag = None
                if not self.ruler_keep.get():
                    self._ruler = None     # замер сделан — незачем ему висеть поверх местности
            elif self._drag and self._drag[0] in ("building_rot", "crossing_rot"):
                self._drag = None
            elif self._drag and self._drag[0] == "box3d":
                self._drag = None
                self._finish_box_3d(add=True)
            elif self._drag and self._drag[0] in ("move", "node"):
                # _changed обязателен: он пересобирает поля, проверяет пересечения на переправы
                # и метит округу к пересчёту. Без него правка осталась бы только на картинке.
                # Округу передаём НАКОПЛЕННУЮ за протяжку (_touch): без неё _changed пометил бы
                # «поменялось всё», и на отпускании пересчитывалась бы вся карта разом — во
                # время протяжки-то куски не трогали.
                self._drag = None
                self._changed(dirty=self._dirty_rect)
                self._refresh_points()
            self.draw_3d()                 # отпустили — перерисовать в полном разрешении
            return
        if self._drag and self._drag[0] == "vision":
            self._vision = None
            self._drag = None
            self.draw()
            return
        if self._drag and self._drag[0] == "ruler":
            self._drag = None
            if not self.ruler_keep.get():
                self._ruler = None
            self.draw()
            return
        if self._drag and self._drag[0] == "box":
            self._drag = None
            return self._finish_box(ev)
        if self._drag and self._drag[0] in ("move", "node", "building", "building_rot",
                                            "crossing_rot", "marker"):
            self._changed(dirty=self._dirty_rect)
        self._drag = None
        self._refresh_points()

    def on_double(self, ev):
        if self._draft:
            self.finish_draft()

    def on_right(self, ev):
        if self.mode3d.get():
            if self._draft:
                self.finish_draft()      # как на плане: ПКМ замыкает начатое, а не крутит камеру
                return
            if self.tool.get() == "nodes" and self._right_node_3d(ev):
                return
            self._cam_drag = ("rot", (ev.x, ev.y, self.cam.yaw, self.cam.bias))
            return
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
            # Не попали в узел, но попали в ГРАНИЦУ — ставим там новый. Иначе изогнуть готовый
            # контур было нечем: узлы можно было только двигать и удалять, а добавить — никак,
            # и приходилось перерисовывать фигуру целиком
            if self.selected is not None:
                sh = self.doc.shapes[self.selected]
                if sh["kind"] in ("polygon", "line"):
                    hit = self._nearest_edge(sh["points"], p, sh["kind"] == "polygon",
                                             max(self._tol_m(9), sh.get("width_m", 8) / 2))
                    if hit is not None:
                        self._insert_node(*hit)
                        self.draw()
                        return
        hit = self._hit_shape(p)
        if hit is not None:
            self._select([hit])
            self._refresh_shapes()
            self.draw()

    def _right_node_3d(self, ev):
        """ПКМ по узлам В ОБЪЁМЕ: по ручке — удалить, по границе — добавить. Возвращает True,
        если кнопку забрали у камеры.

        Считаем по ЭКРАНУ, как и попадание в ручку: и ручки, и обводка нарисованы поверх
        местности, поэтому за бугром они видны, а луч по земле туда не дойдёт."""
        if self.selected is None:
            return False
        sh = self.doc.shapes[self.selected]
        if sh["kind"] not in ("polygon", "line"):
            return False
        k = self._hit_node_3d(ev)
        if k is not None:
            if len(sh["points"]) > (3 if sh["kind"] == "polygon" else 2):
                self.push_undo()
                sh["points"].pop(k)
                self._changed()
                self.draw_3d()
            else:
                self.status.config(text="узлов и так минимум — удалите фигуру целиком")
            return True
        # Целимся в ту линию, которая НАРИСОВАНА, то есть в дроблёную по земле. По исходным
        # вершинам не выйдет: между ними экранный отрезок идёт напрямик, а граница — по склону,
        # и на бугре они расходятся на десятки пикселей.
        closed = sh["kind"] == "polygon"
        dense, mark = self._drape(sh["points"], closed)
        scr, vis = self._project_ground(dense)
        # точки за спиной камеры уводим в бесконечность: там проекция врёт
        seen = [scr[i] if vis[i] else (1e9, 1e9) for i in range(len(scr))]
        hit = self._nearest_edge(seen, (ev.x, ev.y), False, 10.0)
        if hit is None:
            return False
        i, t = hit
        k0, t0 = mark[i]
        k1, t1 = mark[min(i + 1, len(mark) - 1)]
        # доля внутри ИСХОДНОГО ребра; если дробление перешагнуло на следующее — берём его конец
        frac = t0 + (t1 - t0) * t if k1 == k0 else min(1.0, t0 + (1.0 - t0) * t)
        self._insert_node(k0, frac)
        self.draw_3d()
        return True

    def on_hover(self, ev):
        if self.mode3d.get():
            p = self.pick_ground(ev.x, ev.y)
            if p is None:
                self.status.config(text="небо")
                return
            cell = self._height_cell()
            h = self.doc.height_m(cell)
            z = 0.0 if h is None else float(
                h[int(np.clip(p[0] / cell, 0, h.shape[0] - 1)),
                  int(np.clip(p[1] / cell, 0, h.shape[1] - 1))])
            self.status.config(text="%7.0f x %7.0f м   высота %+6.1f м%s"
                                    % (p[0], p[1], z,
                                       "   ·   точек в фигуре: %d, ПКМ или Enter замыкает"
                                       % len(self._draft) if self._draft else ""))
            self._hover_track(p)
            return
        x, y = self._world(ev)
        W_m, H_m = self.doc.size_m
        where = "" if (0 <= x <= W_m and 0 <= y <= H_m) else "  вне карты"
        self.status.config(text=f"{x:7.0f} x {y:7.0f} м   "
                                f"({x / P.M_PER_UNIT:.1f}, {y / P.M_PER_UNIT:.1f} ед){where}")
        self._hover_track((x, y))

    def _hover_track(self, p):
        """Курсор поехал — если фигура начата, перерисовать кадр с нитью до курсора.

        Без этого рисование выглядит мёртвым: поставил точку и до следующего щелчка не видишь
        ничего. Перерисовываем не чаще тридцати раз в секунду: кадр объёма стоит 12 мс, а событий
        движения мыши приходит вдвое больше."""
        self._hover = (float(p[0]), float(p[1]))
        if self._draft and self._hover_job is None:
            self._hover_job = self.after(33, self._hover_draw)

    def _hover_draw(self):
        self._hover_job = None
        if self._draft and self.winfo_exists():
            self.draw()

    # --- правка

    def _changed(self, height_only=False, dirty=None, recheck=True):
        # Замер тут НЕ запускается. Раньше он шёл после каждой правки, и на большой карте это
        # выглядело как зависание: сетка 666x666 плюс сотни лучей на каждый штрих. Теперь — F5.
        #
        # А вот переправы проверяются именно здесь, и намеренно: это ОБЩАЯ воронка всех правок
        # геометрии. Проверка в ней покрывает не только сегодняшние пути (перенос, узлы, поворот,
        # вставка, зеркало, скругление), но и те, что появятся позже, — новый путь получает её
        # даром, а не забывается, как забылись все, кроме рисования. Цена — 6.6 мс на театре.
        #
        # recheck=False у отмены, возврата и удаления: первые два обязаны быть точно обратимы, а
        # созданная внутри них фигура обратимость ломает; удаление же переправу создать не может.
        if recheck and not height_only and self._sync_crossings():
            dirty = None       # мост мог встать вне задетого куска — пересчитываем всю округу
        if height_only:
            self.doc.bump_height()       # в живом стиле картинки местности не трогаем
            # ...а в топостиле трогаем: горизонтали рисуются по высоте. Поэтому и здесь
            # говорим, ЧТО именно поменялось: без этого печать холма сбрасывала все
            # куски карты разом и топостиль наводился одиннадцать секунд вместо одной.
            self._dirty_rect = dirty
        else:
            self.doc.bump()
            # что именно поменялось — чтобы пересчитать только задетые куски местности, а не всю
            # округу: без этого после каждой нарисованной фигуры карта на полсекунды грубеет
            self._dirty_rect = dirty
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
        kind = self.draft_kind(tool)
        relief = tool in self.RELIEF_TOOLS
        if kind == "polygon" and len(pts) >= 3:
            self.push_undo()
            sh = {"kind": "polygon", "type": "relief" if relief else t, "points": pts}
            if relief:
                return self._stamp_relief(sh)
            self.doc.shapes.append(sh)
            self._done_shape(sh)
            return
        elif kind == "line" and len(pts) >= 2:
            self.push_undo()
            width = float(self.relief_width.get() if relief else self.width_m.get())
            sh = {"kind": "line", "type": "relief" if relief else t,
                  "width_m": width, "points": pts}
            if relief:
                return self._stamp_relief(sh)
            if self.dash_on.get():
                sh["dash_m"] = [float(self.dash_len.get()), float(self.dash_gap.get())]
            self.doc.shapes.append(sh)
            self._sync_crossings(only=[sh])
            self._done_shape(sh)
            return
        else:
            self.status.config(text="точек мало: полигону нужно 3, линии 2")
            return

    def _done_shape(self, sh):
        """Фигура дорисована: сбрасываем черновик и пересобираем только то, чего она коснулась."""
        self._draft = None
        self._select([len(self.doc.shapes) - 1])
        x0, y0, x1, y1 = vectormap._shape_bounds(sh)
        pad = 2.0 * float(sh.get("width_m", 0.0)) + 60.0
        self._changed(dirty=(x0 - pad, y0 - pad, x1 + pad, y1 + pad))

    def _stamp_relief(self, sh):
        """Рельеф не фигура, а ШТАМП: форма вдавливается в карту высот и перестаёт существовать
        отдельно. Иначе на одну высоту два источника правды — фигуры и растр, — и правка одного
        молча расходится с другим.

        Склон задаётся в метрах и свой у каждого штампа. Раньше он выводился из размера карты
        (сглаживание радиусом в восьмую стороны), поэтому одна и та же гряда давала разный холм
        на 2.5 км и на 10 км, а обрыв нарисовать было нечем."""
        sh["h_m"] = float(self.relief_h.get())
        slope = float(self.relief_slope.get())
        vectormap.stamp(self.doc.vec, [sh], slope_m=slope,
                        absolute=bool(self.relief_abs.get()))
        # Задетая округа: сам штамп плюс запас на его склон (размытие расходится за радиус) и на
        # сглаживание высоты перед горизонталями. Лучше пересчитать лишний квадрат, чем оставить
        # соседний со старой линией.
        b = vectormap._shape_bounds(sh)
        # три прохода скользящего среднего радиусом 0.55*slope расходятся на 1.65*slope; сверху
        # запас на сглаживание высоты перед горизонталями (45 м) и на клетку карты высот
        pad = 2.0 * slope + 120.0
        self._draft = None
        self._select([])
        hm = self.doc.vec["height"]["h"]
        self.status.config(
            text="рельеф вдавлен: %s %+.0f м, склон %.0f м · карта высот %.0f..%.0f м"
                 % ("выровнено до" if self.relief_abs.get() else "поднято на",
                    sh["h_m"], float(self.relief_slope.get()), hm.min(), hm.max()))
        self._changed(height_only=True,
                      dirty=(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))

    def _place_building(self, p):
        """Поставить дом. Общее для плана и объёма: в объёме точка приходит лучом по рельефу,
        дальше всё то же самое."""
        self.push_undo()
        w, h = self._house_size()
        self.doc.shapes.append({"kind": "building",
                                "rect_m": [round(float(p[0]), 1), round(float(p[1]), 1),
                                           w, h, 0.0],
                                "capacity": 1})
        self._select([len(self.doc.shapes) - 1])
        # протяжка только доворачивает: размер задан числами, и сбивать его движением мыши
        # незачем — дома на карте отличаются именно размером
        self._drag = ("building_rot", (p, self.selected))
        self._changed(dirty=self._shape_rect(self.doc.shapes[-1]))
        return None

    def _place_crossing(self, p):
        """Поставить переправу — или довернуть уже стоящую, если попали в неё."""
        hit = self._hit_shape(p)
        if hit is not None and self.doc.shapes[hit]["kind"] == "crossing":
            self.push_undo()                           # повторная протяжка доворачивает мост
            self._select([hit])
            self._drag = ("crossing_rot", (self.doc.shapes[hit]["point"], hit))
            return None
        self.push_undo()
        sh = {"kind": "crossing", "point": [round(float(p[0]), 1), round(float(p[1]), 1)]}
        # Брод ставится только руками. Автопостановка ищет пересечения ДОРОГИ с рекой, а там
        # нужен мост: дорога, упирающаяся в брод, — это не переправа, а место, где дорога тонет.
        if self.bridge_ford.get():
            sh["ford"] = True
        if not self.bridge_auto.get():
            # ручные размеры: длина — насколько мост перекрывает реку, ширина — проезд
            sh["length_m"] = float(self.bridge_len.get())
            sh["width_m_road"] = float(self.bridge_wid.get())
        # Поставили мост там, где раньше снесли, — значит восстановили: снимаем отметку, иначе
        # она осталась бы гасить будущие проверки в этом месте.
        self.doc.vec["blown"] = [q for q in self.doc.vec.get("blown", [])
                                 if math.hypot(q[0] - sh["point"][0], q[1] - sh["point"][1]) > 60.0]
        self.doc.shapes.append(sh)
        self._select([len(self.doc.shapes) - 1])
        self._drag = ("crossing_rot", (p, self.selected))
        self._changed(dirty=self._shape_rect(sh))
        return None

    @staticmethod
    def _shape_rect(sh, pad=80.0):
        x0, y0, x1, y1 = vectormap._shape_bounds(sh)
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    def _sync_crossings(self, only=None):
        """Переправы там, где дорога пересекла реку. Зовётся после ЛЮБОЙ правки геометрии.

        Без переправы дорога через реку НЕ работает: у клетки один тип, вода перекрывает дорогу,
        и берег остаётся берегом. На вид при этом всё в порядке — дорога входит в реку и выходит
        с той стороны, — поэтому промах не замечается, пока карта не окажется разрезанной
        пополам. Генератор ставит мосты сам с самого начала; руками про них забывали.

        РАНЬШЕ ПРОВЕРКА СТОЯЛА В ОДНОМ МЕСТЕ — на завершении линии. То есть мост появлялся,
        только если пересечение возникало прямо в момент рисования, а любая последующая правка
        его уже не создавала: дорогу подвинули (в том числе рамкой, пачкой), поправили узлы реки,
        вставили копией, отразили карту — пересечение есть, переправы нет. Отказ ровно тот же и
        такой же тихий. Правка пачкой сделала его вероятнее: двигать дорогу стало дёшево.

        only — сузить до конкретных фигур (только что нарисованная линия). Без него проверяется
        вся карта: на театре в 497 фигур это 6.6 мс — дешевле одной перерисовки, поэтому можно
        не экономить и звать на каждую правку.

        Мост остаётся обычной фигурой: его можно подвинуть, довернуть, задать длину или удалить —
        взорванный мост тоже нужен, и выражается он именно отсутствием переправы. Снесённые
        вручную не воскресают: их точки лежат в doc["blown"], и crossing_gaps считает их
        решённым вопросом."""
        if only is not None:
            only = [sh for sh in only if sh.get("type") in ("road", "water")]
            if not only:
                return 0
        gaps = vectormap.crossing_gaps(self.doc.vec, only=only)
        for p in gaps:
            self.doc.shapes.append({"kind": "crossing", "point": p})
        if gaps:
            self.status.config(
                text="поставлено переправ: %d — дорога через реку без моста не работает"
                     % len(gaps))
        return len(gaps)

    def cancel_draft(self):
        self._draft = None
        self._hover = None
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
        """Удаляем ВСЕ выделенные, с конца: иначе после первого же удаления остальные индексы
        съезжают и стирается не то."""
        if not self.sel:
            return
        self.push_undo()
        n = len(self.sel)
        blown = self.doc.vec.setdefault("blown", [])
        for i in sorted(self.sel, reverse=True):
            sh = self.doc.shapes.pop(i)
            # Удалить переправу — это «мост взорван», осмысленное решение, а не оплошность.
            # Точку помним: без неё пересчёт при ближайшей же правке поставит мост обратно и
            # молча отменит решение — ровно та ложь редактора, которой быть не должно.
            if sh["kind"] == "crossing":
                blown.append([float(sh["point"][0]), float(sh["point"][1])])
        self._select([])
        self._changed(recheck=False)
        self.status.config(text="удалено фигур: %d  ·  Ctrl+Z вернёт" % n)

    def copy_selected(self):
        if not self.sel:
            return
        self.clipboard = [copy.deepcopy(self.doc.shapes[i]) for i in self.sel]
        self.status.config(text="скопировано фигур: %d  ·  Ctrl+V вставит со сдвигом"
                                % len(self.clipboard))

    def paste_clipboard(self, dx=None, dy=None):
        """Вставка СО СДВИГОМ, а не на то же место: иначе копия ложится точно поверх исходной, и
        понять, вставилось что-то или нет, можно только по счётчику фигур."""
        if not self.clipboard:
            return
        step = float(dx if dx is not None else max(20.0, self._tol_m() * 3))
        stepy = float(dy if dy is not None else -step)
        self.push_undo()
        first = len(self.doc.shapes)
        for sh in self.clipboard:
            q = copy.deepcopy(sh)
            if q["kind"] in ("polygon", "line"):
                q["points"] = [[round(a + step, 1), round(b + stepy, 1)] for a, b in q["points"]]
            elif q["kind"] == "building":
                r = list(q["rect_m"])
                q["rect_m"] = [round(r[0] + step, 1), round(r[1] + stepy, 1), r[2], r[3], r[4]]
            elif q["kind"] == "crossing":
                q["point"] = [round(q["point"][0] + step, 1), round(q["point"][1] + stepy, 1)]
            self.doc.shapes.append(q)
        self._select(list(range(first, len(self.doc.shapes))))
        self._changed()
        self.status.config(text="вставлено фигур: %d  ·  тащить — поставить на место"
                                % len(self.clipboard))

    def duplicate_selected(self):
        self.copy_selected()
        self.paste_clipboard()

    def select_all(self):
        self._select(list(range(len(self.doc.shapes))))
        self._refresh_shapes()
        self.draw()
        self.status.config(text="выделено всё: %d фигур" % len(self.sel))

    def do_undo(self):
        if self.undo:
            self.redo.append(self.doc.snapshot())
            self.doc.restore(self.undo.pop())
            self._select([])
            self._changed(recheck=False)
            self._refresh_points()

    def do_redo(self):
        if self.redo:
            self.undo.append(self.doc.snapshot())
            self.doc.restore(self.redo.pop())
            self._select([])
            self._changed(recheck=False)
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
        # Взорванные мосты отражаем вместе с картой: иначе они остались бы на прежнем берегу и
        # гасили переправу не там, где её сносили.
        self.doc.vec["blown"] = [[x, round(H - y, 1)]
                                 for x, y in self.doc.vec.get("blown", [])]
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
        if self.doc.shapes:
            self._graph_main()          # кэшируется по версии карты, считать заново не придётся
            n, e, c, main = getattr(self, "_graph_stats", (0, 0, 0, 0))
            lines += ["", f"дороги: {n} узлов, {e} участков"]
            if c > 1 and n:
                off = n - main
                lines.append(f"  сеть из {c} частей: в стороне {off} узлов из {n}")
                if off > max(2, n * 0.1):
                    lines.append("  это много — проверьте переправы и разрывы дорог")
        if len(self.sel) > 1:
            lines += ["", f"выделено фигур: {len(self.sel)}",
                      "  тащить — двигать все, Del — удалить,",
                      "  Ctrl+C / Ctrl+V — копировать"]
        elif self.selected is not None and self.selected < len(self.doc.shapes):
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

    def lost_buildings(self, surface, cell):
        """Номера строений, которых в собранной сетке НЕТ ни одной клеткой.

        Клетка становится зданием по доле покрытия, а у мелкого дома доля мала: замерено, что
        ниже примерно 7x7 м строение при клетке 30 м не даёт ни одной клетки.

        ЧТО ЭТО ЗНАЧИТ ТЕПЕРЬ — не то, что раньше. Пока местность считалась клетками, такой дом
        для боя не существовал вовсе: пули шли насквозь. С VectorTerrain он и обзор перекрывает,
        и укрывает того, кто в нём сидит, — помеха и материал берутся по фигуре. Но СВОЕЙ КЛЕТКИ
        у него по-прежнему нет, а значит нет и компоненты застройки: занять его отделением
        нельзя, и в поиске укрытия по сетке он не участвует.

        Поэтому предупреждение осталось, но говорит другое. Раньше оно значило «дом пропал»,
        теперь — «дом есть, но гарнизона не держит»."""
        bid = vectormap._types()["building"]
        out = []
        for i, sh in enumerate(self.doc.shapes):
            if sh["kind"] != "building":
                continue
            cx, cy, bw, bh, ang = sh["rect_m"]
            pts = vectormap._rect_points(cx, cy, bw, bh, ang)
            xs = [q[0] for q in pts]
            ys = [q[1] for q in pts]
            x0 = int(np.clip(min(xs) / cell, 0, surface.shape[0] - 1))
            x1 = int(np.clip(max(xs) / cell, 0, surface.shape[0] - 1))
            y0 = int(np.clip(min(ys) / cell, 0, surface.shape[1] - 1))
            y1 = int(np.clip(max(ys) / cell, 0, surface.shape[1] - 1))
            if not (surface[x0:x1 + 1, y0:y1 + 1] == bid).any():
                out.append(i)
        return out

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
        m = measure(surface, cell, n_pairs=400, vec=self.doc.vec)
        lost = self.lost_buildings(surface, cell)
        m["lost_buildings"] = lost
        self.metrics = m
        self.status.config(
            text=f"сетка {surface.shape[0]}x{surface.shape[1]} по {cell:.0f} м · "
                 f"лес {m['frac'][1] * 100:.0f}% застр {m['frac'][2] * 100:.0f}% "
                 f"дор {m['frac'][4] * 100:.0f}% · видимость {m['vis'] * 100:.0f}% · "
                 f"строений {m['comps']} · "
                 + (f"без своей клетки строений: {len(lost)} (укрывают и перекрывают обзор, "
                    f"но гарнизон не держат) · " if lost else "")
                 + ("годна" if not m["bad"] else "вырождена: " + "; ".join(m["bad"])))

    # --- файлы

    def back_to_menu(self):
        """Выход к списку карт. Несохранённую правку спрашиваем: карта живёт в памяти до
        «сохранить и собрать», и уйти отсюда молча значит потерять всё нарисованное."""
        if (self.doc.version, self.doc.hversion) != self._saved_at and not messagebox.askyesno(
                "выйти без сохранения?",
                "Карта изменена после последнего сохранения. Выйти в меню и потерять правки?"):
            return
        self.on_close()

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
        self._saved_at = (self.doc.version, self.doc.hversion)
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
    restore_timer = _timer_resolution(1)
    root = tk.Tk()
    App(root, doc)
    try:
        root.mainloop()
    finally:
        if restore_timer is not None:
            restore_timer()


if __name__ == "__main__":
    main()
