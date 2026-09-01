"""
Местность v1 для WarGameEnv: сетка тайлов поверх непрерывного поля.

Даёт три механики (см. terrain.json):
  * cover      — множитель снижения урона/подавления по цели в клетке;
  * blocks_los — клетка перекрывает ЛИНИЮ ОГНЯ (нельзя стрелять сквозь лес/здание);
  * speed      — множитель скорости движения в клетке.

Карта генерится процедурно на каждый reset (из np_random) — несколько лесных пятен
и зданий, чтобы агент не заучивал одну карту.
"""
import json
import math
import os

import numpy as np


def _load_cfg():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

_CFG = _load_cfg()

# Шаг прореживания кандидатов-ориентиров. Кромка — одномерная кривая, шаг вдоль неё
# даёт 1/3; глубина — двумерная область, решётка даёт 1/9.
EDGE_STRIDE = 3
DEEP_STRIDE = 3


FIELDS = ("cover", "speed_foot", "speed_veh", "see_limit", "see_limit_dem", "blocks",
          "impassable", "height")

# Высота глаз над землёй, в ИГРОВЫХ единицах (2 м при 15 м/ед). Ею определяется, укроет ли
# гребень: наблюдатель и цель смотрят не с земли, а с высоты роста, и складка глубиной в метр
# перестаёт быть укрытием.
EYE_UNITS = 2.0 / 15.0

# Уклон. Подъём стоит времени, крутизна не берётся вовсе. Числа взяты по нормативам движения
# пехоты и техники: пеший теряет около трети скорости на подъёме в 20%, техника — половину;
# предел для техники ~27 градусов (0.5), для пехоты почти вдвое круче.
SLOPE_SLOWDOWN_FOOT = 1.8
SLOPE_SLOWDOWN_VEH = 3.2
MAX_GRADE_FOOT = 0.90
MAX_GRADE_VEH = 0.50


def fields_from_surface(grid):
    """Поля свойств из сетки типов. Таблица «тип -> число» раскрывается в число НА КЛЕТКУ.

    Зачем так, если из типа и так всё выводится: клетка перестаёт быть меткой и становится
    набором величин. Тогда «редкий лес», «сад», «бурьян» — это не новые типы с правкой таблиц,
    а другие числа в тех же полях, и векторная карта может класть их напрямую, минуя тип.
    Пока источник — тип, значения совпадают с прежними один в один."""
    (types, cover_by_id, blocks_by_id, speed_by_id, see_through_by_id, impassable_by_id,
     veh_speed_by_id, see_through_demolish_by_id) = _type_tables()
    g = grid.astype(np.int32)
    return {"cover": cover_by_id[g], "speed_foot": speed_by_id[g], "speed_veh": veh_speed_by_id[g],
            "see_limit": see_through_by_id[g], "see_limit_dem": see_through_demolish_by_id[g],
            "blocks": blocks_by_id[g], "impassable": impassable_by_id[g],
            # плоская карта: рельефа нет, и все проверки высоты обязаны схлопнуться в ноль —
            # старые карты должны вести себя ровно как раньше
            "height": np.zeros(g.shape, dtype=np.float32)}


class TerrainMap:
    """Поля свойств на прямоугольной сетке Gx×Gy + запросы cover/speed и рейкаст линии огня.

    Сетка тут — не описание местности, а посчитанная таблица: что в точке, сколько укрытия,
    как быстро идётся, сколько метров этого материала терпит луч. Откуда она взялась — из типов
    (старые карты) или из вектора — TerrainMap не касается.

    Поле `grid` остаётся типом поверхности: по нему ищутся кромки-ориентиры и рисуется карта."""

    def __init__(self, grid, cell_size, cover_by_id, blocks_by_id, speed_by_id, see_through_by_id,
                 impassable_by_id, building_comp, building_id, building_centers, building_capacity=None,
                 veh_speed_by_id=None, see_through_demolish_by_id=None, forest_id=None, fields=None):
        self.grid = grid                      # (Gx, Gy) int id, индексируется [gx, gy]
        self.Gx, self.Gy = grid.shape
        self.G = self.Gx                      # обратная совместимость (квадратные карты)
        self.width_m = self.Gx * cell_size    # размер поля в игровых единицах по X
        self.height_m = self.Gy * cell_size   # и по Y
        self.cell = cell_size
        self.cover_by_id = cover_by_id        # np.array по id
        self.blocks_by_id = blocks_by_id
        self.speed_by_id = speed_by_id
        self.veh_speed_by_id = veh_speed_by_id if veh_speed_by_id is not None else speed_by_id
        self.see_through_by_id = see_through_by_id  # сколько МЕТРОВ луч проходит сквозь тип до блока
        self.see_through_demolish_by_id = (see_through_demolish_by_id
            if see_through_demolish_by_id is not None else see_through_by_id)  # для фугаса (demolish)
        self.impassable_by_id = impassable_by_id    # непроходимо ВСЕМ наземным (вода)
        self.forest_id = forest_id if forest_id is not None else -1  # id типа «лес» (для штрафа точности)
        self.building_comp = building_comp    # (Gx,Gy) id связного здания (0 = не здание)
        self.building_id = building_id        # id типа «здание»
        self.building_centers = building_centers    # comp -> (wx, wy) центр здания (для отрисовки)
        self.building_capacity = building_capacity or {}  # comp -> вместимость (отрядов); нет записи = 1
        self._edge_cache = None               # маска кромок, считается лениво
        self._deep_cache = {}                 # маски глубины массива по (тип, клеток)

        # ПОЛЯ — то, по чему реально считается бой. Точечные запросы читают их, а не таблицы
        # по id: так карта может прийти из вектора с непрерывными значениями, а код не заметит.
        f = fields if fields is not None else fields_from_surface(grid)
        self.f_cover = np.asarray(f["cover"], dtype=np.float32)
        self.f_speed_foot = np.asarray(f["speed_foot"], dtype=np.float32)
        self.f_speed_veh = np.asarray(f["speed_veh"], dtype=np.float32)
        self.f_see = np.asarray(f["see_limit"], dtype=np.float32)
        self.f_see_dem = np.asarray(f["see_limit_dem"], dtype=np.float32)
        self.f_blocks = np.asarray(f["blocks"], dtype=bool)
        self.f_impassable = np.asarray(f["impassable"], dtype=bool)
        self.f_height = np.asarray(f.get("height", np.zeros(grid.shape)), dtype=np.float32)
        self.has_relief = bool(np.any(self.f_height))   # плоская карта -> профиль не считаем
        # маски для поиска ближайшего: считаются один раз, а не предикатом на каждую клетку
        self._inv_cell = 1.0 / float(cell_size)
        self._road_mask = (self.f_speed_foot > 1.0) | (self.f_speed_veh > 1.0)
        self._cover_mask = self.f_cover > 0
        self._is_building = self.grid == building_id

    def component_at(self, pos):
        """id здания под позицией (0 = не в здании) — для правила «своё здание прозрачно»."""
        gx, gy = self._cell(pos)
        return int(self.building_comp[gx, gy])

    def capacity_of(self, comp):
        """Вместимость (сколько отрядов может стоять) компонента здания. По умолчанию 1
        (процедурные карты — маленькие дискретные здания); импортированные карты задают явно
        по плотности реальной застройки внутри слипшегося компонента."""
        return self.building_capacity.get(comp, 1)

    def passable(self, pos, is_vehicle):
        """Проходима ли клетка: вода непроходима всем; технике (is_vehicle) здания тоже непроходимы."""
        gx, gy = self._cell(pos)
        if self.f_impassable[gx, gy]:
            return False
        if is_vehicle and self._is_building[gx, gy]:
            return False
        return True

    def _cell(self, pos):
        """Точка -> клетка. Самая горячая функция всего боя: её зовёт каждый шаг луча линии
        огня, а лучей за шаг десятки тысяч. Здесь СПЕЦИАЛЬНО нет numpy: np.clip на скаляре стоит
        около трёх микросекунд, и на 1.9 млн вызовов за сто двадцать шагов это была треть всего
        времени боя. Обычная арифметика Python делает то же самое в разы дешевле."""
        gx = int(pos[0] // self.cell)
        gy = int(pos[1] // self.cell)
        if gx < 0:
            gx = 0
        elif gx >= self.Gx:
            gx = self.Gx - 1
        if gy < 0:
            gy = 0
        elif gy >= self.Gy:
            gy = self.Gy - 1
        return gx, gy

    def _id_at(self, pos):
        gx, gy = self._cell(pos)
        return int(self.grid[gx, gy])

    def height_at(self, pos):
        """Высота земли в точке, в игровых единицах."""
        gx, gy = self._cell(pos)
        return float(self.f_height[gx, gy])

    def slope_factor(self, p0, p1, is_vehicle=False):
        """Во сколько раз медленнее идти из p0 в p1 из-за уклона; 0.0 — не пройти.

        Считается по ходу движения, а не по клетке: одна и та же крутизна вниз и вверх — разные
        вещи. Спуск не ускоряем: на тактическом темпе выигрыша нет, а осторожность на спуске
        съедает его целиком.
        """
        if not self.has_relief:
            return 1.0
        d = np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32)
        run = float(np.linalg.norm(d))
        if run < 1e-6:
            return 1.0
        rise = self.height_at(p1) - self.height_at(p0)
        grade = rise / run
        if grade > (MAX_GRADE_VEH if is_vehicle else MAX_GRADE_FOOT):
            return 0.0
        if grade <= 0:
            return 1.0
        k = SLOPE_SLOWDOWN_VEH if is_vehicle else SLOPE_SLOWDOWN_FOOT
        return float(1.0 / (1.0 + k * grade))

    def cover_at(self, pos):
        gx, gy = self._cell(pos)
        return float(self.f_cover[gx, gy])

    def speed_at(self, pos, is_vehicle=False):
        gx, gy = self._cell(pos)
        return float(self.f_speed_veh[gx, gy] if is_vehicle else self.f_speed_foot[gx, gy])

    def _nearest_matching_point(self, pos, radius, mask):
        """Ближайшая (в метрах) точка клетки, попавшей в маску, в радиусе. None, если нет.

        Считается пачкой, а не двойным циклом: окно радиуса — это сотни клеток, и перебор их
        по одной в питоне стоил около трети миллисекунды на вызов, а зовётся это на каждый шаг
        каждого юнита. Порядок обхода сохранён (сначала по x, потом по y), поэтому при равных
        расстояниях выбирается ровно та же клетка, что и раньше."""
        gx, gy = self._cell(pos)
        rc = int(radius / self.cell)
        x0, x1 = max(0, gx - rc), min(self.Gx, gx + rc + 1)
        y0, y1 = max(0, gy - rc), min(self.Gy, gy + rc + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        sub = mask[x0:x1, y0:y1]
        if not sub.any():
            return None
        wx = (np.arange(x0, x1, dtype=np.float32) + 0.5) * self.cell - pos[0]
        wy = (np.arange(y0, y1, dtype=np.float32) + 0.5) * self.cell - pos[1]
        dd = wx[:, None] ** 2 + wy[None, :] ** 2
        dd = np.where(sub & (dd > 1e-6), dd, np.inf)
        k = int(np.argmin(dd))
        if not np.isfinite(dd.flat[k]):
            return None
        i, j = divmod(k, y1 - y0)
        return ((x0 + i + 0.5) * self.cell, (y0 + j + 0.5) * self.cell)

    def nearest_road_point(self, pos, radius=25.0):
        """Мировые координаты (не вектор!) ближайшей клетки дороги в радиусе, или None.
        Возвращает точку (не направление), чтобы вызывающий мог держать её как ФИКСИРОВАННУЮ
        цель (см. _role_move) — иначе «ближайшая точка» скачет каждый шаг и юнит дёргается."""
        return self._nearest_matching_point(pos, radius, self._road_mask)

    def nearest_cover_point(self, pos, radius=18.0):
        """Мировые координаты ближайшей клетки с укрытием (cover>0) в радиусе, или None.
        Точка (не направление) — для фиксации цели, см. nearest_road_point."""
        return self._nearest_matching_point(pos, radius, self._cover_mask)

    def nearest_cover(self, pos, radius=18.0):
        """Единичный вектор к ближайшей клетке с укрытием (cover>0) в радиусе, или None."""
        best = self._nearest_matching_point(pos, radius, self._cover_mask)
        if best is None:
            return None
        v = np.array([best[0] - pos[0], best[1] - pos[1]], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    def _edge_mask(self):
        """Клетки, у которых есть сосед ДРУГОГО типа, — то есть кромки.

        Кромка, а не середина: разметка живого игрока легла именно так (медиана расстояния до
        края леса 0 м против 30 м у случайной точки внутри массива), и то же вышло при наведении
        огня — центр массива тактически пуст, туда не видно и там никого нет."""
        if getattr(self, "_edge_cache", None) is None:
            g = self.grid
            e = np.zeros(g.shape, dtype=bool)
            e[:-1, :] |= g[:-1, :] != g[1:, :]
            e[1:, :] |= g[1:, :] != g[:-1, :]
            e[:, :-1] |= g[:, :-1] != g[:, 1:]
            e[:, 1:] |= g[:, 1:] != g[:, :-1]
            # ПРОРЕЖИВАЕМ ВТРОЕ. Соседние клетки кромки — это один и тот же выбор: они в 30 м
            # друг от друга, а «занято другим подразделением» и так отсекает всё в радиусе трёх
            # клеток. Густота кандидатов только зашумляла отбор и удорожала поиск.
            # (x+y)%LM_EDGE_STRIDE вдоль связной кривой оставляет РОВНО каждую третью и
            # равномерно — в отличие от случайного прореживания.
            xs, ys = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]), indexing="ij")
            self._edge_cache = e & ((xs + ys) % EDGE_STRIDE == 0)
        return self._edge_cache

    def _deep_mask(self, tid, k):
        """Клетки типа tid, удалённые от любой границы не меньше чем на k клеток.

        Смысл отдельный от кромки: с 90 м леса линию огня уже не проложить, поэтому глубина
        массива — это укрытие ОТ НАБЛЮДЕНИЯ. Там не стреляют, там накапливаются незамеченными,
        прячут резерв и проходят скрытно. Кромка — наоборот, огневая позиция.

        Эрозия сдвигами, без scipy: k маленькое (2-3), а зависимость тянуть незачем."""
        key = (tid, k)
        if key not in self._deep_cache:
            m = (self.grid == tid)
            for _ in range(max(k - 1, 0)):
                a = m.copy()
                a[:-1, :] &= m[1:, :]
                a[1:, :] &= m[:-1, :]
                a[:, :-1] &= m[:, 1:]
                a[:, 1:] &= m[:, :-1]
                a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = False   # край карты — тоже граница
                m = a
            # ПРОРЕЖИВАЕМ ВДЕВЯТЕРО: глубина массива — двумерная область, и решёткой с шагом
            # DEEP_STRIDE от неё остаётся 1/9. Шаг 3 клетки = 90 м, что как раз соответствует
            # непрозрачности леса: две соседние «глубокие» точки различимы тактически, а
            # соседние клетки — нет.
            xs, ys = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
            m = m & (xs % DEEP_STRIDE == 0) & (ys % DEEP_STRIDE == 0)
            if len(self._deep_cache) > 8:
                self._deep_cache.clear()
            self._deep_cache[key] = m
        return self._deep_cache[key]

    def nearest_edge_point(self, pos, tid, radius, want=None, min_cos=0.15, exclude=(),
                           keep_out=0.0, deep_cells=0):
        """Ближайшая кромочная клетка типа tid в радиусе, лежащая В СЕКТОРЕ вокруг want.

        want — единичный вектор желаемого направления (из приказа: слева/фронтально/справа).
        min_cos отсекает всё, что вне сектора. exclude — уже занятые кем-то точки: без этого
        два подразделения выберут одну опушку и взвод сложится в кучу.
        Возвращает мировые координаты или None."""
        gx, gy = self._cell(pos)
        rc = int(radius / self.cell)
        # deep_cells>0 — ищем не кромку, а ГЛУБИНУ массива (укрытие от наблюдения)
        edge = self._deep_mask(tid, deep_cells) if deep_cells > 0 else self._edge_mask()
        best, bestd = None, 1e18
        for x in range(max(0, gx - rc), min(self.Gx, gx + rc + 1)):
            for y in range(max(0, gy - rc), min(self.Gy, gy + rc + 1)):
                if int(self.grid[x, y]) != tid or not edge[x, y]:
                    continue
                wx = (x + 0.5) * self.cell
                wy = (y + 0.5) * self.cell
                dx, dy = wx - pos[0], wy - pos[1]
                dd = dx * dx + dy * dy
                if dd < keep_out * keep_out or dd >= bestd:
                    continue
                if want is not None and dd > 1e-9:
                    n = dd ** 0.5
                    if (dx * want[0] + dy * want[1]) / n < min_cos:
                        continue                      # не в том секторе
                if any((wx - ex) ** 2 + (wy - ey) ** 2 < (3.0 * self.cell) ** 2
                       for ex, ey in exclude):
                    continue                          # эту кромку уже заняли
                bestd, best = dd, (wx, wy)
        return best

    def firing_position(self, pos, watch_pt, radius=22.0, max_range=30.0, stride=2,
                        transparent=()):
        """Позиция в радиусе от pos, откуда ЕСТЬ линия огня на watch_pt в пределах max_range;
        при прочих равных предпочитает укрытие и близость к pos.

        Нужна для задачи ПРИКРЫТЬ: командир указывает КОГО прикрывать, а конкретную огневую
        позицию (лесополосу, окраину посёлка, гребень) ищут правила — потому что это вычислимо,
        а не вопрос суждения. stride прореживает перебор: без него на каждую клетку радиуса
        приходится трассировка луча, и это заметно дорого.
        """
        gx, gy = self._cell(pos)
        rc = int(radius / self.cell)
        best, best_score = None, -1e18
        for x in range(max(0, gx - rc), min(self.Gx, gx + rc + 1), stride):
            for y in range(max(0, gy - rc), min(self.Gy, gy + rc + 1), stride):
                if self.f_impassable[x, y]:
                    continue
                wx = (x + 0.5) * self.cell
                wy = (y + 0.5) * self.cell
                p = (wx, wy)
                d_watch = np.hypot(wx - watch_pt[0], wy - watch_pt[1])
                if d_watch > max_range or d_watch < 1e-6:
                    continue
                if self.blocked(p, watch_pt, transparent):
                    continue                       # нет линии огня — позиция бесполезна
                    # transparent ОБЯЗАТЕЛЕН: если наблюдаемый сидит в здании, оно перекрывает
                    # луч в любую точку, годных позиций нет ВООБЩЕ, и поиск запускается каждый
                    # шаг заново. Снаружи это выглядит как беготня кругами вокруг дома.
                d_self = np.hypot(wx - pos[0], wy - pos[1])
                score = 2.0 * float(self.f_cover[x, y]) - 0.05 * d_self
                if score > best_score:
                    best_score, best = score, p
        return best

    N_SENSE_DIRS = 8
    SENSE_RADIUS = 12.0  # ед.: сопоставимо с дальностью хода за 1 шаг с запасом

    _SENSE_ANGLES = None  # кэш np.arange(N_SENSE_DIRS) углов, чтобы не пересчитывать каждый вызов

    def sense_obstacles(self, pos, is_vehicle):
        """Дистанция до ближайшего НЕПРОХОДИМОГО препятствия (здание технике, вода всем) по
        N_SENSE_DIRS направлениям вокруг pos, нормированная на SENSE_RADIUS (1.0 = чисто, 0 = вплотную).
        Даёт агенту «чувство препятствия впереди» — без этого он слепо упирается в стены.
        Векторизовано (без Python-цикла по шагам луча) — иначе на 20 юнитов/шаг это заметно дорого."""
        if TerrainMap._SENSE_ANGLES is None:
            k = np.arange(self.N_SENSE_DIRS)
            TerrainMap._SENSE_ANGLES = 2.0 * np.pi * k / self.N_SENSE_DIRS
        step = self.cell * 0.5
        nsteps = max(1, int(self.SENSE_RADIUS / step))
        s = np.arange(1, nsteps + 1, dtype=np.float32)             # (nsteps,)
        dx = np.cos(TerrainMap._SENSE_ANGLES) * step                # (N_DIRS,)
        dy = np.sin(TerrainMap._SENSE_ANGLES) * step
        px = pos[0] + np.outer(dx, s)                                # (N_DIRS, nsteps)
        py = pos[1] + np.outer(dy, s)
        # Допуск обязателен: cos(270 град) = -1.8e-16, а не ровно 0. У юнита, прижатого к x=0
        # (в _move позиция клипится к границе), луч «вниз» получает микроскопически отрицательный
        # x и считается вышедшим за карту, а зеркальный луч «вверх» — cos(90) = +6.1e-17 — нет.
        # Синие и красные у края видели разное; тест зеркалирования это поймал.
        eps = 1e-6
        oob = ((px < -eps) | (px > self.width_m + eps)
               | (py < -eps) | (py > self.height_m + eps))
        gx = np.clip((px // self.cell).astype(np.int32), 0, self.Gx - 1)
        gy = np.clip((py // self.cell).astype(np.int32), 0, self.Gy - 1)
        blocked = self.f_impassable[gx, gy] | oob
        if is_vehicle:
            blocked = blocked | self._is_building[gx, gy]
        any_blocked = blocked.any(axis=1)
        first = blocked.argmax(axis=1).astype(np.float32)            # индекс первого True (0 если нет)
        return np.where(any_blocked, first / nsteps, 1.0).astype(np.float32)

    def forest_crossed_m(self, p0, p1):
        """Суммарная толщина ЛЕСА (метров) вдоль луча p0->p1 — НЕ блокирует сама по себе (это для
        blocked()/see_through_m), а даёт МЕРУ для штрафа точности: тонкая лесополоса не глушит
        выстрел совсем, но должна мешать прицелу, а не быть нулевым фактором."""
        d = np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32)
        dist = float(np.linalg.norm(d))
        if dist < 1e-6 or self.forest_id < 0:
            return 0.0
        c0 = self._cell(p0)
        c1 = self._cell(p1)
        steps = int(dist / (self.cell * 0.5)) + 1
        step_len = dist / steps
        total = 0.0
        for s in range(1, steps):
            p = p0 + d * (s / steps)
            c = self._cell(p)
            if c == c0 or c == c1:
                continue
            if int(self.grid[c[0], c[1]]) == self.forest_id:
                total += step_len
        return total

    def blocked(self, p0, p1, transparent=(), demolish=False):
        """Перекрыта ли линия огня между p0 и p1? Здание блокирует сразу (кроме своих/целевых
        компонент из transparent). Лес — накопительно: блокирует, если суммарная толщина леса
        вдоль луча превышает see_through_m (обычное оружие) или see_through_demolish_m (фугас —
        видит/бьёт сквозь лес ДАЛЬШЕ, но НЕ бесконечно — было багом "видит лес насквозь").
        Рельеф перекрывает луч, если земля поднимается выше линии от глаз к глазам.

        Считается пачкой, а не шагами в цикле: луч — самый горячий путь всего боя (треть
        времени шага). Результат тот же, потому что ответ здесь булев: раз накопленная толщина
        превысила порог хоть где-то, порядок обхода не важен, и сумму можно взять сразу по
        всему лучу.
        """
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        d = p1 - p0
        # ВАЖНО: длина и координаты вдоль луча считаются ровно как раньше — одинарная точность
        # для разности, двойная для хода по лучу. Считать «эквивалентно, но иначе» тут нельзя:
        # на границе клетки луч уходит в соседнюю, и линия огня меняется. Проверка боями это
        # поймала сразу.
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return False
        c0 = self._cell(p0)
        c1 = self._cell(p1)
        steps = int(dist / (self.cell * 0.5)) + 1
        if steps < 2:
            return False
        step_len = dist / steps

        t = np.arange(1, steps, dtype=np.float64) / steps
        ix = np.floor((p0[0] + d[0] * t) / self.cell).astype(np.int32)
        iy = np.floor((p0[1] + d[1] * t) / self.cell).astype(np.int32)
        np.clip(ix, 0, self.Gx - 1, out=ix)
        np.clip(iy, 0, self.Gy - 1, out=iy)

        # клетки самого стрелка и самой цели не считаются — своё укрытие не мешает
        keep = ~(((ix == c0[0]) & (iy == c0[1])) | ((ix == c1[0]) & (iy == c1[1])))
        if not keep.any():
            return False
        ix = ix[keep]
        iy = iy[keep]

        if self.has_relief:
            h0 = self.f_height[c0[0], c0[1]] + EYE_UNITS
            h1 = self.f_height[c1[0], c1[1]] + EYE_UNITS
            line = h0 + (h1 - h0) * t[keep].astype(np.float32)
            if (self.f_height[ix, iy] > line).any():
                return True

        blocks = self.f_blocks[ix, iy]
        if not blocks.any():
            return False
        if transparent:
            # ВАЖНО: ноль означает «не здание», и он частенько попадает в transparent — среда
            # кладёт туда компоненту под стрелком, а тот обычно не в доме. Без проверки на ноль
            # прозрачными становились ВСЕ не-здания, включая лес, и линия огня открывалась
            # через весь лес. Бои это поймали сразу, случайные лучи — нет: там transparent пуст.
            comp = self.building_comp[ix, iy]
            see_through = (comp != 0) & np.isin(comp, np.fromiter(transparent, dtype=np.int32))
            blocks = blocks & ~see_through
            if not blocks.any():
                return False

        lim = (self.f_see_dem if demolish else self.f_see)[ix, iy][blocks]
        if (lim <= 0).any():                       # здание: гасит луч сразу
            return True
        # остальное копится по материалам: столько метров этого материала луч терпит
        for v in np.unique(lim):
            if float(np.count_nonzero(lim == v)) * step_len > v:
                return True
        return False


def _type_tables():
    """Таблицы свойств по id тайла из terrain.json (общие для генератора и загрузчика)."""
    types = _CFG["types"]
    max_id = max(t["id"] for t in types.values())
    cover_by_id = np.zeros(max_id + 1, dtype=np.float32)
    blocks_by_id = np.zeros(max_id + 1, dtype=bool)
    speed_by_id = np.ones(max_id + 1, dtype=np.float32)
    veh_speed_by_id = np.ones(max_id + 1, dtype=np.float32)
    see_through_by_id = np.zeros(max_id + 1, dtype=np.float32)
    see_through_demolish_by_id = np.zeros(max_id + 1, dtype=np.float32)
    impassable_by_id = np.zeros(max_id + 1, dtype=bool)
    for t in types.values():
        cover_by_id[t["id"]] = t["cover"]
        blocks_by_id[t["id"]] = t["blocks_los"]
        speed_by_id[t["id"]] = t["speed"]
        veh_speed_by_id[t["id"]] = t.get("speed_vehicle", t["speed"])  # дороги: техника быстрее пеших
        see_through_by_id[t["id"]] = t.get("see_through_m", 0)
        # фугас (demolish) видит/бьёт сквозь лес дальше обычного, но НЕ бесконечно (было — баг:
        # техника "видела лес насквозь" на большой карте); по умолчанию = обычная see_through_m.
        see_through_demolish_by_id[t["id"]] = t.get("see_through_demolish_m", t.get("see_through_m", 0))
        impassable_by_id[t["id"]] = t.get("impassable", False)
    return (types, cover_by_id, blocks_by_id, speed_by_id, see_through_by_id, impassable_by_id,
           veh_speed_by_id, see_through_demolish_by_id)


BLOCK_CELLS = 2   # сторона квартала в клетках; при клетке 30 м это 60 м — размер строения


def _building_components(grid, bid):
    """Строения как ОТДЕЛЬНЫЕ КВАРТАЛЫ, а не сплошные связные компоненты.

    Раньше здесь была обычная заливка связности, и на импортированных картах целое село
    слипалось в ОДИН компонент — замер на настоящих вырезках дал 390 x 270 м, семь гектаров.
    Последствия были все сразу:
      * вместимость компонента 1 отряд, то есть одно отделение занимало село целиком и никто
        больше войти не мог;
      * укрытие 0.65 на всей площади, а снаружи в село не видно вовсе — бесплатная крепость
        тому, кто добежал первым;
      * правило «своё здание прозрачно для своего огня» (писавшееся под домик 30 x 30) давало
        сквозную простреливаемость на все 390 м;
      * пересечь такой «дом» занимало десять минут игрового времени, и отряд всё это время
        формально находился в здании.
    Поэтому режем сетку на квадраты BLOCK_CELLS x BLOCK_CELLS и ищем связность ВНУТРИ квадрата:
    так каждый квартал — самостоятельное строение со своей вместимостью и своей прозрачностью,
    а бой за село превращается в бой за отдельные дома. Два несмежных строения, попавших в один
    квадрат, остаются разными компонентами — связность считается честно.
    """
    Gx, Gy = grid.shape
    building_comp = np.zeros((Gx, Gy), dtype=np.int32)
    cid = 0
    for x in range(Gx):
        for y in range(Gy):
            if grid[x, y] != bid or building_comp[x, y] != 0:
                continue
            bx, by = x // BLOCK_CELLS, y // BLOCK_CELLS      # квартал, которому принадлежит клетка
            cid += 1
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if building_comp[cx, cy] != 0:
                    continue
                building_comp[cx, cy] = cid
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < Gx and 0 <= ny < Gy):
                        continue
                    if grid[nx, ny] != bid or building_comp[nx, ny] != 0:
                        continue
                    if (nx // BLOCK_CELLS, ny // BLOCK_CELLS) != (bx, by):
                        continue                            # соседний квартал — это другое строение
                    stack.append((nx, ny))
    return building_comp, cid


def _finalize(grid, cell, bid, building_capacity=None, fields=None, building_comp=None):
    (types, cover_by_id, blocks_by_id, speed_by_id, see_through_by_id, impassable_by_id,
     veh_speed_by_id, see_through_demolish_by_id) = _type_tables()
    if building_comp is None:
        building_comp, cid = _building_components(grid, bid)
    else:
        building_comp = np.asarray(building_comp, dtype=np.int32)
        cid = int(building_comp.max())
    building_centers = {}
    for c in range(1, cid + 1):
        xs, ys = np.where(building_comp == c)
        if len(xs) == 0:
            continue    # компонент без клеток: у векторных карт так бывает, когда один дом
            #             целиком накрыт другим — перекрытые клетки достаются верхнему.
            #             Раньше это роняло сборку карты на argmin пустой последовательности.
        mx, my = xs.mean(), ys.mean()
        # ВАЖНО: для несимметричных/разбросанных компонент (реальные слипшиеся кластеры зданий
        # на импортированных картах) геометрический центроид может упасть В РАЗРЫВ формы — вне
        # клеток самого компонента. Берём ближайшую РЕАЛЬНУЮ клетку компонента, а не сырое среднее.
        k = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
        building_centers[c] = ((xs[k] + 0.5) * cell, (ys[k] + 0.5) * cell)
    return TerrainMap(grid, cell, cover_by_id, blocks_by_id, speed_by_id, see_through_by_id,
                      impassable_by_id, building_comp, bid, building_centers, building_capacity,
                      veh_speed_by_id, see_through_demolish_by_id, types["forest"]["id"], fields)


def make_map(rng, arena):
    """Сгенерировать процедурную TerrainMap (квадратное поле arena×arena, для тренировочной среды)."""
    types, *_ = _type_tables()
    cell = float(_CFG["cell_size"])
    G = int(round(arena / cell))
    grid = np.zeros((G, G), dtype=np.int8)  # 0 = open
    gen = _CFG["generate"]
    fid = types["forest"]["id"]
    bid = types["building"]["id"]

    # ЛЕС ДИСКАМИ, но мелкими. Ориентир — статистика 14 годных вырезок настоящих карт:
    # лес 21%, линия огня на 300-900 м открыта 52%. Было radius 7-18 (то есть 105-270 м при
    # 15 м/ед): пятно перекрывало любой задевший его луч, видимость падала до 31%.
    #
    # Пробовал лесополосами и сбором в массивы — вышло ХУЖЕ по обоим показателям сразу
    # (полосы под общим углом дали 30% видимости, сбор в массивы схлопнул лес до 4%, потому
    # что полосы длиннее кластера и ложились друг на друга). Оставляю диски: они дают нужные
    # цифры, а форму настоящей местности всё равно честнее брать из настоящих карт —
    # см. maps/platoon_pool.json.
    xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    nf = int(rng.integers(gen["forest_patches"][0], gen["forest_patches"][1] + 1))
    for _ in range(nf):
        cx = rng.integers(0, G); cy = rng.integers(0, G)
        r = max(1, int(round(rng.uniform(*gen["forest_radius"]) / cell)))
        grid[(xs - cx) ** 2 + (ys - cy) ** 2 <= r * r] = fid

    # здания (квадратные блоки со стороной building_size_m метров)
    nb = int(rng.integers(gen["buildings"][0], gen["buildings"][1] + 1))
    for _ in range(nb):
        s = max(1, int(round(rng.uniform(gen["building_size_m"][0], gen["building_size_m"][1]) / cell)))
        bx = int(rng.integers(0, G - s + 1)); by = int(rng.integers(0, G - s + 1))
        grid[bx:bx + s, by:by + s] = bid

    return _finalize(grid, cell, bid)


def from_grid(grid, cell_size, building_capacity=None):
    """Построить TerrainMap из готовой (Gx,Gy) сетки id тайлов (для авторских/загруженных сценариев).
    building_capacity: {comp_id: вместимость} — для импортированных карт с плотной застройкой,
    где несколько реальных зданий слились в один компонент сетки (см. maps/segment_reference.py)."""
    types, *_ = _type_tables()
    bid = types["building"]["id"]
    return _finalize(grid.astype(np.int8), float(cell_size), bid, building_capacity)


def from_fields(surface, fields, cell_size, building_capacity=None, building_comp=None):
    """TerrainMap из ГОТОВЫХ полей — путь векторной карты (см. vectormap.py).

    surface нужен по-прежнему: по нему ищутся кромки-ориентиры («опушка», «край застройки»,
    «узел дороги») и рисуется карта. Числа же берутся из полей и могут быть какими угодно —
    редкий лес, разбитая дорога, каменный дом отличаются значениями, а не новым типом.
    building_comp можно передать готовым: у векторной карты дома — объекты, и искать связные
    пятна клеток не нужно (это самая дорогая часть построения карты)."""
    types, *_ = _type_tables()
    bid = types["building"]["id"]
    return _finalize(np.asarray(surface, dtype=np.int8), float(cell_size), bid,
                     building_capacity, fields, building_comp)


def load_ascii_map(path, cell_size, legend=None):
    """Загрузить карту из текстового файла: строки = ряды по Y (сверху=большой Y), символы -> тип
    тайла по legend (по умолчанию '.'=open '^'=forest '#'=building '~'=water). Не привязан к нашей
    процедурной генерации — способ импортировать/авторизовать любую раскладку."""
    types, *_ = _type_tables()
    legend = legend or {".": "open", "^": "forest", "#": "building", "~": "water"}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip("\n") != ""]
    Gy = len(lines); Gx = max(len(ln) for ln in lines)
    grid = np.zeros((Gx, Gy), dtype=np.int8)
    for row_i, line in enumerate(lines):
        y = Gy - 1 - row_i  # первая строка файла = верх карты = больший Y
        for x, ch in enumerate(line):
            tname = legend.get(ch, "open")
            grid[x, y] = types[tname]["id"]
    return from_grid(grid, cell_size)
