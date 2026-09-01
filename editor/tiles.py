"""Кусочная местность: карта режется на тайлы, у каждого уровня своя клетка.

Зачем. Карту 10x10 км нельзя показать целиком в мелкой клетке: в пяти метрах это четыре
миллиона точек, секунды на пересборку и сотни мегабайт. Но и не нужно — вдали клетка всё равно
меньше пикселя. Поэтому карта режется на квадраты, у каждого уровня клетка вдвое крупнее
предыдущего, и рядом с камерой берутся мелкие, вдали — крупные.

Что тайл считает, а что нет. Тайл — это только сетка ТИПОВ (лес, дорога, вода, застройка),
посчитанная из вектора для своего окна. Высоту он не трогает: сглаживание рельефа работает на
всю карту сразу, и посчитанное по куску не совпало бы с посчитанным по целой — на стыке кусков
земля бы ломалась. Высота одна на всех, общим полем, и каждый уровень читает её с
интерполяцией. Отсюда же главное следствие: щелей между уровнями не бывает в принципе, потому
что соседние уровни берут высоту из одного и того же места.

То же и с горизонталями топостиля: они посчитаны ЛИНИЯМИ по всей карте один раз (paint.Contours)
и в тайл только рисуются. Уровень подробности решает лишь, каждую ли линию показывать или
каждую вторую-четвёртую, и кратность берётся степенью двойки — поэтому дальний вид показывает
подмножество ближнего, а не другой набор линий.

Счёт идёт в отдельном потоке. Тайл 640 м в клетке 5 м стоит 17 мс, а на театре и все 46 — это
пропущенные кадры, если считать в главном потоке. Пока тайл не готов, на его месте рисуется
кусок родительского, более грубого: карта не мигает дырами, а просто становится подробнее.
"""
import copy
import heapq
import math
import os
import threading

import numpy as np

import paint as painter
import vectormap
import view3d

TILE_CELLS = 128           # клеток по стороне тайла — одинаково на всех уровнях
BASE_CELL = 5.0            # метров в клетке самого мелкого уровня: дом различим
TARGET_PX = 340.0          # тайл крупнее этого на экране — спускаемся на уровень мельче
MIN_LEVEL = -4             # докуда разрешено дробить НИЖЕ базового. Нулевой уровень — тайл в
#                            640 м на 512 точек, то есть 1.25 м на точку, и это было дно: у
#                            земли, где на карте видно полтораста метров, одна точка растягивалась
#                            на полтора десятка пикселей экрана — та самая блочность на дорогах и
#                            горизонталях. Дробить ниже нуля ничего не стоит: cell = 5 * 2**level
#                            на отрицательных считается так же, ключи и подмена предком работают
#                            на РАЗНОСТЯХ уровней, а сколько тайлов в кадре — решает всё тот же
#                            TARGET_PX, а не глубина. На -4 тайл 40 м даёт 0.078 м на точку,
#                            вшестнадцатеро подробнее прежнего дна. Глубже без нужды не пускаем:
#                            каждый уровень — это ещё один пересчёт тайлов на подлёте
MAX_TILES = 224            # предохранитель: больше этого в кадр не берём. Поднят вместе с
#                            MIN_LEVEL: когда дробить стало можно глубже, прежних 96 перестало
#                            хватать, и предохранитель срабатывал ШТАТНО, а не в крайнем случае —
#                            на театре до трети кусков оставались грубее, чем положено по их
#                            размеру на экране, то есть в кадре были пятна мыла. Замерено:
#                            при 96 таких 32%, при 160 — 3%, при 224 — 0%, а больше 176 кусков
#                            не набирается ни на одной проверенной точке обзора
MARGIN = 4                 # клеток запаса вокруг тайла: столько нужно мипмапам. С одной
#                            клеткой уменьшённые копии картинки на краю замешивают пустоту, и
#                            стыки тайлов проступают тонкой сеткой поверх местности
PAINT_PX = 512             # точек на сторону тайла в векторной картинке: на нижнем уровне это
#                            1.25 м на точку — вчетверо подробнее клетки в 5 м и без лесенки
PAINT_MARGIN = 16          # точек запаса вокруг векторной картинки, то же назначение
WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
#                            Считаем куски НЕСКОЛЬКИМИ потоками: картинка куска почти целиком
#                            внутри PIL и numpy, а те на время работы отпускают замок питона,
#                            поэтому потоки идут по-настоящему параллельно. Одним потоком
#                            наведение театра занимало 6 секунд.
CACHE_BYTES = 320 << 20    # столько памяти под готовые куски всех стилей: картинка куска
#                            888 КБ, то есть около 360 штук. Дальше вытесняется давнее.
#                            Удвоено вместе с MAX_TILES: 176 кусков одного кадра — это уже
#                            156 МБ, и при прежних 160 МБ в памяти не оставалось НИЧЕГО, кроме
#                            текущего кадра. Любой поворот выбрасывал только что посчитанное


class TileGrid:
    """Уровни, выбор тайлов под камеру и фоновый счёт."""

    def __init__(self, vec, size_m, base_cell=BASE_CELL, tile_cells=TILE_CELLS, mode="vector"):
        self.size_m = (float(size_m[0]), float(size_m[1]))
        # vector — фигуры как есть; cells — клетки, что читает бой; topo — топографический лист
        self.mode = mode
        self._hm = None                  # снимок карты высот: топостилю нужны горизонтали
        self.step_m = 5.0                # сечение рельефа, одно на всю карту
        self.base_cell = float(base_cell)
        self.tile_cells = int(tile_cells)
        self.min_level = int(MIN_LEVEL)
        self.levels = 1
        while self.span(self.levels - 1) < max(self.size_m):
            self.levels += 1

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._cache = {}                 # (стиль,L,tx,ty) -> картинка куска
        self._used = {}                  # тот же ключ -> когда последний раз спрашивали
        self._clock = 0
        self._gen = {}                   # тот же ключ -> счёт пересчётов этого куска.
        #                            Нужен снаружи: видеокарта держит картинки тайлов по
        #                            ключу, и без номера пересчёта она оставляла бы
        #                            прежнюю — нарисованная фигура не появлялась бы до
        #                            смены стиля, когда ключ менялся целиком
        self._queue = []                 # ключи в очереди, ближние первыми
        self._shapes = []
        self.version = (-1, -1)
        self.built = 0                   # сколько тайлов посчитано за сеанс (для замера)
        self._stop = False
        self.set_source(vec, 0)
        self._workers = [threading.Thread(target=self._work, daemon=True)
                         for _ in range(WORKERS)]
        for w in self._workers:
            w.start()

    # --- геометрия уровней

    def cell(self, level):
        return self.base_cell * (2 ** level)

    def span(self, level):
        return self.cell(level) * self.tile_cells

    def rect(self, key):
        """Левый нижний угол и сторона тайла в метрах."""
        level, tx, ty = key
        sp = self.span(level)
        return tx * sp, ty * sp, sp

    def margin_m(self, level, mode=None):
        """Запас вокруг тайла в метрах — он же сдвиг картинки относительно своего квадрата."""
        if (mode or self.mode) in ("vector", "topo"):
            return self.span(level) * PAINT_MARGIN / PAINT_PX
        return MARGIN * self.cell(level)

    # Ключ в хранилище — со СТИЛЕМ: (стиль, уровень, x, y). Иначе смена стиля выбрасывала все
    # готовые куски, и на их месте оставался единственный посчитанный — верхний, вся карта в
    # 512 точек. На театре это 20 м на точку: картинка на секунды становилась грубым растром,
    # хотя сами фигуры никуда не девались. Теперь стили лежат рядом, и возврат мгновенный.
    def _fk(self, key, mode=None):
        return (mode or self.mode,) + tuple(key)

    def set_mode(self, mode):
        """Векторная картинка, клетки боя или тополист. Готовые куски прежнего стиля НЕ
        выбрасываем: они пригодятся и на возврат, и как подмена, пока считается новый стиль."""
        if mode == self.mode:
            return
        with self._lock:
            self.mode = mode
            self._queue.clear()

    # --- источник

    def set_source(self, vec, version, dirty=None):
        """Новая версия источника: ПАРА (версия фигур, версия высоты) — одна на все стили.

        Пара, а не одно число, потому что высота меняет только тополист: горизонтали рисуются
        по ней, а цвет местности — нет. Правка рельефа сбрасывает поэтому лишь куски топостиля.
        И пара ОДНА для всех стилей: пока версия зависела от текущего стиля, переключение стиля
        само выглядело как смена карты и выбрасывало все готовые куски разом.

        Фигуры копируем: их правят в главном потоке, а читает рабочий,
        и читать чужой список во время правки нельзя.

        dirty — прямоугольник (x0, y0, x1, y1) в метрах, который поменялся. Пересчитываем только
        задетые куски: нарисованный лесок трогает два-три квадрата из шестнадцати, а сбрасывать
        все — это полсекунды, в которые местность на глазах грубеет и потом снова наводится.
        None означает «неизвестно что поменялось» (отмена, перенос фигуры) — тогда всё заново."""
        with self._lock:
            if version == self.version:
                return
            was = self.version
            self.version = version
            # поменялась только высота — цвет местности от неё не зависит, трогаем один тополист
            only_topo = (isinstance(was, tuple) and isinstance(version, tuple)
                         and was[0] == version[0])
            self._shapes = copy.deepcopy(vec["shapes"])
            hm = vec.get("height")
            # Массив высот НЕ копируем: штамп заменяет его целиком, а не правит на месте,
            # поэтому ссылка на прежний остаётся целой, пока её читает рабочий поток.
            self._hm = None if hm is None else {"cell_m": hm["cell_m"], "h": hm["h"]}
            if self._hm is not None and np.any(self._hm["h"]):
                rng = float(self._hm["h"].max() - self._hm["h"].min())
                # Сечение считается по ВСЕЙ карте, а не по куску: иначе у соседних кусков разный
                # шаг горизонталей, и на стыке они не сходятся — читается как рябь.
                self.step_m = painter.contour_step(rng)
            x0, y0, x1, y1 = dirty if dirty else (None, None, None, None)
            for fk in list(self._cache):
                if only_topo and fk[0] != "topo":
                    continue
                if dirty is not None:
                    kx, ky, sp = self.rect(fk[1:])
                    m = self.margin_m(fk[1], fk[0])
                    if (kx - m > x1 or kx + sp + m < x0
                            or ky - m > y1 or ky + sp + m < y0):
                        continue
                del self._cache[fk]
            self._queue.clear()

    # --- выбор тайлов под камеру

    def select(self, cam, size):
        """Листья дерева тайлов под текущую камеру, ближние первыми.

        Спускаемся от самого грубого уровня: пока тайл на экране крупнее TARGET_PX, дробим его
        на четыре. Мерка — экранный размер, а не расстояние: она сама учитывает и наклон, и
        удаление, и ширину окна.

        Отдельно — отсев того, что вне кадра. Без него при подлёте к земле дробится вся округа,
        включая то, что за спиной: девяносто тайлов вместо пятнадцати, и все считаются."""
        w, h = float(size[0]), float(size[1])
        cp = math.cos(math.radians(cam.pitch))
        sp_ = math.sin(math.radians(cam.pitch))
        cy = math.cos(math.radians(cam.yaw))
        sy = math.sin(math.radians(cam.yaw))
        ez = cam.dist * sp_

        k = h / (2 * w)                  # тангенс половины вертикального угла (фокус = ширина)
        m = 1.25                         # поле допуска: лучше лишний тайл, чем дыра в кадре

        def visible(key):
            """(глубина центра, виден ли) — отсев по плоскостям пирамиды видимости.

            Считаем в координатах КАМЕРЫ, без деления на глубину. Делить нельзя: у тайла,
            сидящего на камере, часть углов позади, деление там врёт, и такой тайл либо
            пропадает, либо — что хуже — считается ближайшим и дробится в мелкую клетку. На
            театре из-за этого в мелкую клетку уходило восемьдесят тайлов вместо десяти,
            половина из них за спиной."""
            x0, y0, sp = self.rect(key)
            xr_, yc_, zc_ = [], [], []
            for cxm, cym in ((x0, y0), (x0 + sp, y0), (x0, y0 + sp), (x0 + sp, y0 + sp)):
                dx, dy = cxm - cam.tx, cym - cam.ty
                xr = dx * cy - dy * sy
                yr = dx * sy + dy * cy
                xr_.append(xr)
                yc_.append(yr * sp_)
                zc_.append(yr * cp + cam.dist)
            if all(z < 1.0 for z in zc_):
                return 0.0, False
            if all(x > 0.5 * m * z for x, z in zip(xr_, zc_)):
                return 0.0, False
            if all(x < -0.5 * m * z for x, z in zip(xr_, zc_)):
                return 0.0, False
            if all(y > k * m * z for y, z in zip(yc_, zc_)):
                return 0.0, False
            if all(y < -k * m * z for y, z in zip(yc_, zc_)):
                return 0.0, False
            return sum(zc_) / 4.0, True

        # Дробим ПО ОЧЕРЕДИ, начиная с самого крупного на экране, а не как придётся из стопки.
        # Порядок важен из-за предохранителя MAX_TILES: когда он срабатывает, спуск обрывается,
        # и обрывается он на том, до чего очередь не дошла. При обходе стопкой это выходило
        # произвольно — замер на театре показывал БЛИЖНИЙ кусок уровня -2 при мельчайшем -3 в
        # кадре, то есть подробность доставалась боковому, а земля под носом оставалась грубой.
        # С очередью предохранитель срезает наименее заметное, а ближнее получает своё первым.
        seq = 0

        def offer(pile, key):
            """Положить тайл в очередь, если он вообще виден. Экранный размер — он же
            приоритет."""
            nonlocal seq
            x0, y0, sp = self.rect(key)
            if x0 >= self.size_m[0] or y0 >= self.size_m[1]:
                return                                    # тайл целиком за краем карты
            zc, seen = visible(key)
            if not seen:
                return                                    # тайл вне кадра
            seq += 1                                      # разводит равные, порядок устойчив
            heapq.heappush(pile, (-(sp * w / max(zc, 1.0)), seq, key, zc))

        top = self.levels - 1
        n = int(math.ceil(max(self.size_m) / self.span(top)))
        pile = []
        for tx in range(n):
            for ty in range(n):
                offer(pile, (top, tx, ty))
        out = []
        while pile:
            _, _, key, zc = heapq.heappop(pile)
            level, tx, ty = key
            x0, y0, sp = self.rect(key)
            if (level > self.min_level and sp * w / max(zc, 1.0) > TARGET_PX
                    and len(out) + len(pile) < MAX_TILES):
                for i in (0, 1):
                    for j in (0, 1):
                        offer(pile, (level - 1, tx * 2 + i, ty * 2 + j))
            else:
                d = math.sqrt((x0 + sp / 2 - cam.tx) ** 2 + (y0 + sp / 2 - cam.ty) ** 2 + ez * ez)
                out.append((d, key))
        out.sort()
        return [k for _, k in out]

    # --- готовность

    def get(self, key, mode=None):
        with self._lock:
            return self._touch(self._fk(key, mode))

    def get_gen(self, key, mode=None):
        """Картинка тайла вместе с номером пересчёта, одним замком: между двумя
        отдельными запросами рабочий поток успевает подменить картинку, и номер
        достался бы не от неё."""
        fk = self._fk(key, mode)
        with self._lock:
            return self._touch(fk), self._gen.get(fk, 0)

    def _touch(self, fk):
        """Достать и пометить как только что нужный — по этой метке идёт вытеснение."""
        got = self._cache.get(fk)
        if got is not None:
            self._used[fk] = self._clock
            self._clock += 1
        return got

    def ready_ancestor(self, key, mode=None):
        """Ближайший готовый предок — им закрываем место, пока считается свой тайл."""
        level, tx, ty = key
        while level < self.levels - 1:
            level, tx, ty = level + 1, tx // 2, ty // 2
            got, gen = self.get_gen((level, tx, ty), mode)
            if got is not None:
                return (level, tx, ty), got, gen
        return None, None, 0

    def other_style(self, key):
        """Тот же квадрат, посчитанный в ДРУГОМ стиле. Подмена на время переключения: картинка
        не та по цвету, зато той же подробности — это честнее, чем растянутый верхний кусок,
        который грубее в двадцать раз."""
        for mode in ("vector", "topo", "cells"):
            if mode == self.mode:
                continue
            got, gen = self.get_gen(key, mode)
            if got is not None:
                return mode, got, gen
        return None, None, 0

    def request(self, keys):
        """Поставить в очередь недостающие. Порядок сохраняется: ближние считаются первыми."""
        with self._lock:
            want = [self._fk(k) for k in keys]
            want = [k for k in want if k not in self._cache and k not in self._queue]
            if want:
                keep = set(self._fk(k) for k in keys)
                self._queue = want + [k for k in self._queue if k in keep]
                self._wake.notify_all()          # разбудить всех: очередь на несколько потоков

    def build_now(self, key):
        """Посчитать тайл прямо здесь. Нужен для самого грубого уровня при открытии карты:
        без него первый кадр пустой."""
        fk = self._fk(key)
        surf = self._render(fk)
        with self._lock:
            self._store(fk, surf)
        return surf

    def _store(self, fk, surf):
        """Положить картинку и вытеснить давнее, если памяти набралось выше предела. Без предела
        хранилище росло бы вечно: на театре одних только мелких квадратов 256 на стиль."""
        self._cache[fk] = surf
        self._gen[fk] = self._gen.get(fk, 0) + 1
        self._used[fk] = self._clock
        self._clock += 1
        self.built += 1
        total = sum(v.nbytes for v in self._cache.values())
        if total <= CACHE_BYTES:
            return
        for old in sorted(self._cache, key=lambda k: self._used.get(k, 0)):
            if total <= CACHE_BYTES:
                break
            total -= self._cache[old].nbytes
            del self._cache[old]
            self._used.pop(old, None)

    # --- рабочий поток

    def _render(self, fk):
        """Картинка тайла: (py, px, 3) uint8, строка 0 — низ квадрата. Ключ приходит СО СТИЛЕМ:
        пока кусок ждал очереди, стиль могли переключить, и считать надо тот, что заказан.

        Оба вида отдают именно картинку, а не сетку типов: видеокарте нужна текстура, и
        переводить одно в другое где-то ещё значит держать этот перевод в двух местах."""
        mode, key = fk[0], fk[1:]
        level = key[0]
        x0, y0, sp = self.rect(key)
        m = self.margin_m(level, mode)
        with self._lock:
            shapes = self._shapes
        if mode in ("vector", "topo"):
            n = PAINT_PX + 2 * PAINT_MARGIN
            # отсекаем фигуры вне куска: на театре их за две сотни, и гонять все ради квадрата
            # в полкилометра — впустую. Тот же отбор делает и оконная растеризация
            x1, y1 = x0 + sp + m, y0 + sp + m
            near = [sh for sh in shapes
                    if not (lambda b: b[2] < x0 - m or b[0] > x1 or b[3] < y0 - m or b[1] > y1)(
                        vectormap._shape_bounds(sh))]
            lines, mult = None, 1
            if mode == "topo" and self._hm is not None:
                # Горизонтали считаются ЛИНИЯМИ по всей карте разом и запоминаются:
                # тайл только выбирает, какие из них попали в его квадрат. Пока каждый
                # тайл считал их сам по своей выборке высот, у соседних кусков выходили
                # разные линии, а при отъезде камеры они перерисовывались заново.
                lines = painter.contour_set(self._hm["h"], self._hm["cell_m"],
                                            self.step_m)
                # прореживание — от МАСШТАБА уровня, а не от местности под куском:
                # иначе у соседей разные наборы линий и на стыке они не сходятся
                mult = lines.mult_for((sp + 2 * m) / n)
            return painter.paint(near, x0 - m, y0 - m, sp + 2 * m, sp + 2 * m, n, n,
                                 style=mode, lines=lines, mult=mult)
        c = self.cell(level)
        surf = vectormap.surface_window({"shapes": shapes}, c,
                                        x0 - m, y0 - m, sp + 2 * m, sp + 2 * m)
        return np.ascontiguousarray(view3d.surface_rgb(surf).transpose(1, 0, 2).astype(np.uint8))

    def _work(self):
        while True:
            with self._lock:
                while not self._queue and not self._stop:
                    self._wake.wait()
                if self._stop:
                    return
                fk = self._queue.pop(0)
                version = self.version
            try:
                surf = self._render(fk)
            except Exception as ex:                        # не роняем поток из-за одного тайла
                print("тайл %s не посчитался: %s" % (fk, ex))
                continue
            with self._lock:
                if version == self.version:                # карту не успели переправить
                    self._store(fk, surf)

    def stop(self):
        with self._lock:
            self._stop = True
            self._wake.notify_all()

    # --- для показа

    def stats(self):
        with self._lock:
            return {"уровней": "%d..%d" % (self.min_level, self.levels - 1),
                    "мельче всего": "%.2f м на точку" % (self.span(self.min_level) / PAINT_PX),
                    "в памяти": len(self._cache),
                    "в очереди": len(self._queue), "посчитано": self.built,
                    "картинка": self.mode}
