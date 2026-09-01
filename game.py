"""Играбельная оболочка: посмотреть механики вживую и получить ЧЕЛОВЕЧЕСКИЙ эталон.

Два режима:
    py -3.12 game.py --mode play    ты командуешь синими, красными — скриптовый взводный
    py -3.12 game.py --mode watch   оба взводных скриптовые; можно встрять, взяв элемент под себя

Зачем это не просто «поиграть»:
  * человек, модель и скрипт играют ОДНИМ интерфейсом (задача -> объект -> заход), поэтому
    сравнимы напрямую — появляется человеческий эталон для bench.py;
  * если играя всё время тянешься к сырому «идти в точку» (правая кнопка) вместо задач из
    словаря — это готовый список того, чего словарю не хватает. Никакой замер этого не покажет;
  * подавление, потеря стволов с потерями, прижатие — видно глазами, а не по числам.

Управление:
    ЛКМ по юниту      выбрать его элемент + карточка юнита
    1..9              задача выбранному элементу (список слева внизу)
    ЛКМ по объекту    указать объект приказа (зона / свой элемент / элемент противника)
    Q / W / E         заход: слева / фронтально / справа
    ПКМ               СЫРОЙ приказ «идти в точку» (вне словаря — намеренно)
    L (удерживать)    что просматривается из-под курсора (линия видимости, как в SD2)
    ПРОБЕЛ            пауза,  .  шаг при паузе,  + / -  скорость
    TAB               линии огня вкл/выкл
    R                 перезапуск боя,  ESC  выход
"""
import argparse
import os
import sys

import numpy as np
import pygame

import play          # нужен целиком: карточка рисует _nato_icon
import wargame_env
from wargame_env import AXES, CALLSIGNS, ELEMENT_NAMES, TASKS, TASK_OBJECT
from play import (BLUE, RED, SCALE, SIZE, draw_fire_lines, draw_terrain,
                  draw_unit, draw_zones, to_screen)

# Интерфейс — НИЖНЯЯ ПОЛУПРОЗРАЧНАЯ ПОЛОСА поверх поля, как в Wargame: Red Dragon. Не сбоку:
# карта остаётся квадратной и координаты to_screen/from_screen не надо пересчитывать.
BAR = 232
W_TOTAL = SIZE
BAR_Y = SIZE - BAR
UI_BG = (22, 28, 22)          # тёмная олива, как панели в Wargame: Red Dragon
UI_BOX = (38, 48, 36)         # фон коробки ствола
UI_EDGE = (104, 128, 84)      # рамка
UI_AMBER = (214, 198, 132)    # тан — основной текст WRD
UI_TEXT = (188, 200, 168)
UI_DIM = (120, 136, 108)
UI_OK = (150, 220, 130)       # ГОТОВ
UI_WARN = (232, 170, 70)
UI_BAD = (226, 84, 66)        # ПРИЖАТ / расчёт выбит
UI_STRENGTH = (110, 220, 226) # голубые блоки численности — как в WRD


def weapon_icon(screen, name, x, y, w, h, col):
    """Силуэт ствола. Рисуем линиями, а не картинками: файлов ресурсов у проекта нет,
    а различать стволы в карточке надо с одного взгляда, как в WRD."""
    cy = y + h // 2
    n = name.upper()
    if "РПГ" in n:                                   # труба с конической БЧ
        pygame.draw.line(screen, col, (x + 4, cy), (x + w - 10, cy), 4)
        pygame.draw.polygon(screen, col, [(x + w - 10, cy - 5), (x + w - 1, cy), (x + w - 10, cy + 5)])
        pygame.draw.line(screen, col, (x + w // 2, cy), (x + w // 2 - 4, cy + 7), 2)
    elif "ГП" in n:                                  # подствольник: ствол + короткая толстая труба
        pygame.draw.line(screen, col, (x + 3, cy - 4), (x + w - 3, cy - 4), 2)
        pygame.draw.rect(screen, col, (x + w // 2 - 6, cy + 1, 14, 6))
    elif "ПКМ" in n or "РПК" in n:                   # пулемёт: ствол, короб, сошки
        pygame.draw.line(screen, col, (x + 2, cy - 2), (x + w - 3, cy - 2), 3)
        pygame.draw.rect(screen, col, (x + 7, cy, 11, 8))
        pygame.draw.line(screen, col, (x + w - 8, cy - 1), (x + w - 12, cy + 8), 2)
        pygame.draw.line(screen, col, (x + w - 8, cy - 1), (x + w - 4, cy + 8), 2)
    else:                                            # автомат: ствол, приклад, магазин
        pygame.draw.line(screen, col, (x + 3, cy - 2), (x + w - 3, cy - 2), 3)
        pygame.draw.line(screen, col, (x + 3, cy - 2), (x, cy + 4), 3)
        pygame.draw.polygon(screen, col, [(x + w // 2 - 3, cy), (x + w // 2 + 4, cy),
                                          (x + w // 2 + 2, cy + 9), (x + w // 2 - 1, cy + 9)])


def strength_blocks(screen, x, y, alive, total, bw=9, bh=13, gap=3):
    """Численность ОТДЕЛЬНЫМИ БЛОКАМИ по бойцам, а не сплошной полоской — как в WRD.
    Так сразу видно, сколько человек выбито, а через min_crew — какие стволы уже потеряны."""
    for k in range(total):
        r = (x + k * (bw + gap), y, bw, bh)
        pygame.draw.rect(screen, UI_STRENGTH if k < alive else (48, 56, 46), r)
        pygame.draw.rect(screen, UI_EDGE, r, 1)
    return x + total * (bw + gap)


def from_screen(mx, my):
    return np.array([(mx - play.VIEW_X) / play.SCALE,
                     (play.VIEW_H - my) / play.SCALE], dtype=np.float32)


def unit_at(env, p, radius=4.5):
    """Ближайший живой юнит к точке (в игровых единицах), или -1."""
    best, bd = -1, radius
    for i in range(env.n):
        if not env.alive[i]:
            continue
        d = float(np.linalg.norm(env.pos[i] - p))
        if d < bd:
            best, bd = i, d
    return best


def _who(sel):
    """Человекочитаемое перечисление выбранных: позывной, если один, иначе счёт."""
    return CALLSIGNS[0][sorted(sel)[0]] if len(sel) == 1 else f"{len(sel)} подразделений"


def element_of(env, i):
    """Какому элементу принадлежит слот и на какой он стороне."""
    for side in (0, 1):
        for e in range(env.n_elements):
            if i in env._element_slots(side, e):
                return side, e
    return None, None


ORDER_COLORS = {
    "move":    (120, 200, 255),   # ОВЛАДЕТЬ / ОБОЙТИ — куда наступать
    "support": (140, 240, 160),   # ПРИКРЫТЬ / ПОДДЕРЖАТЬ — кого обеспечивать
    "fire":    (255, 160, 120),   # СКОВАТЬ / ПОДАВИТЬ — по кому работать
    "hold":    (190, 190, 190),   # ЗАКРЕПИТЬСЯ / РЕЗЕРВ
    "back":    (255, 200, 90),    # ОТОЙТИ
    "free":    (255, 120, 220),   # сырой приказ «идти в точку» — намеренно другого цвета
}
_TASK_GROUP = {"ОВЛАДЕТЬ": "move", "ОБОЙТИ": "move", "ПРИКРЫТЬ": "support",
               "ПОДДЕРЖАТЬ": "support", "СКОВАТЬ": "fire", "ПОДАВИТЬ": "fire",
               "ОГОНЬ": "fire", "ОТОЙТИ": "back", "ЗАКРЕПИТЬСЯ": "hold", "РЕЗЕРВ": "hold"}

# Что подразделение РЕАЛЬНО делает по каждой задаче — иначе по одному названию не догадаться,
# чем ОБОЙТИ отличается от ОВЛАДЕТЬ, а СКОВАТЬ от ПОДАВИТЬ.
TASK_HELP = {
    "ОВЛАДЕТЬ":    "выдвинуться и удерживать",
    "ОБОЙТИ":      "то же, но скрытно через укрытия",
    "ЗАКРЕПИТЬСЯ": "занять укрытие рядом и стоять",
    "ПРИКРЫТЬ":    "встать так, чтобы простреливать его направление",
    "ПОДДЕРЖАТЬ":  "идти следом, эшелонированно позади",
    "СКОВАТЬ":     "держать под угрозой с предельной дистанции",
    "ПОДАВИТЬ":    "сблизиться на дальность огня и бить",
    "ОГОНЬ":       "вызвать миномёты по его району (запас ограничен)",
    "ОТОЙТИ":      "выйти из-под огня к своему краю",
    "РЕЗЕРВ":      "отойти в глубину и ждать ввода в бой",
}

# Словарь задач живёт в среде, а описания и цвета — здесь, и они разъезжаются молча: добавили
# ОГОНЬ в TASKS — интерфейс упал с KeyError при открытии справки. Проверяем на импорте, чтобы
# это вылезало сразу и в консоли, а не в бою.
_missing = [t for t in TASKS if t not in TASK_HELP or t not in _TASK_GROUP]
assert not _missing, f"в game.py нет описания/цвета для задач: {_missing} — дополни TASK_HELP и _TASK_GROUP"


_SMALL_FONT = None


def _small():
    """Мелкий шрифт для подписей поверх карты. Лениво, потому что на импорте модуля
    pygame.font ещё не инициализирован."""
    global _SMALL_FONT
    if _SMALL_FONT is None:
        _SMALL_FONT = pygame.font.SysFont("consolas", 14)
    return _SMALL_FONT


BOOM_SHOW_STEPS = 3     # сколько шагов держать вспышку разрыва на экране


def draw_mortars(screen, env, side):
    """Вызванный миномётный огонь: куда летит (с обратным отсчётом) и где только что легло.

    Метка обязательна: точка фиксируется в момент вызова, и за 30 секунд цель успевает уйти —
    игрок должен видеть, куда именно заказано, а не держать это в голове. Вспышка обязательна
    отдельно: без неё в момент разрыва метка просто пропадает, и САМОГО события не видно."""
    if not hasattr(env, "_mortar_q"):
        return
    col = ORDER_COLORS["fire"] if side == 0 else (215, 120, 110)
    r = max(3, int(wargame_env.MORTAR_R * play.SCALE))
    pending = env._mortar_q[side][env._mortar_q[side][:, 0] >= 0]
    if len(pending):                                    # налёт заказан, но ещё не начался/идёт
        c = to_screen(env._mortar_aim[side])
        # круг по РАССЕИВАНИЮ, а не по радиусу одной мины: игрок должен видеть область, куда
        # реально ляжет залп, иначе метка обещает точность, которой нет
        rr = max(5, int(wargame_env.MORTAR_SPREAD * 2.0 * play.SCALE))
        pygame.draw.circle(screen, col, c, rr, 2)
        pygame.draw.line(screen, col, (c[0] - 9, c[1]), (c[0] + 9, c[1]), 2)
        pygame.draw.line(screen, col, (c[0], c[1] - 9), (c[0], c[1] + 9), 2)
        left = max(int(pending[:, 0].min()) - env.steps, 0)
        screen.blit(_small().render(f"{left * wargame_env.SECONDS_PER_STEP:.0f}с  x{len(pending)}",
                                    True, col), (c[0] + rr + 4, c[1] - 8))
    for bx, by, bs in env._mortar_boom[side]:
        age = env.steps - bs
        if bs < 0 or not (0 <= age <= BOOM_SHOW_STEPS):
            continue
        c = to_screen(np.array([bx, by], dtype=np.float32))
        k = 1.0 - age / (BOOM_SHOW_STEPS + 1)
        pygame.draw.circle(screen, (255, 236, 180), c, max(2, int(r * (0.5 + 0.5 * k))), 3)
        for ang in range(0, 360, 45):
            v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            p2 = (int(c[0] + v[0] * r * 1.3 * k), int(c[1] + v[1] * r * 1.3 * k))
            pygame.draw.line(screen, (255, 210, 120), c, p2, 2)


def dashed(screen, col, a, b, dash=9, gap=7, width=2):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    n = float(np.hypot(*d))
    if n < 1e-6:
        return
    d /= n
    t = 0.0
    while t < n:
        p1 = a + d * t
        p2 = a + d * min(t + dash, n)
        pygame.draw.line(screen, col, p1, p2, width)
        t += dash + gap


def draw_orders(screen, env, sel, side=0, dim=False):
    """Куда и зачем идёт каждый элемент. Без этого приказ виден только текстом, и понять,
    что происходит на карте, невозможно — особенно «идти в точку», у которого цели нет в словаре."""
    for e in range(env.n_elements):
        live = env._element_alive(side, e)
        if not live:
            continue
        c = env._element_center(side, e)
        src = to_screen(c)
        t, o, x = (int(v) for v in env._orders[side, e])
        task = TASKS[t]
        grp = _TASK_GROUP.get(task, "hold")
        col = ORDER_COLORS[grp]
        if dim or (sel and e not in sel and side == 0):
            col = tuple(int(v * 0.45) for v in col)
        wide = 3 if (sel and e in sel and side == 0) else 2

        kind = TASK_OBJECT[task]
        tgt = None
        if kind == "zone" and env.n_zones:
            tgt = env.zones[min(o, env.n_zones - 1)]
        elif kind == "friend":
            tgt = env._element_center(side, min(o, env.n_elements - 1))
        elif kind == "enemy":
            tgt = env._element_center(1 - side, min(o, env.n_elements - 1))

        if tgt is not None:
            dst = to_screen(tgt)
            if task in ("ОВЛАДЕТЬ", "ОБОЙТИ") and x != 1:
                # показываем ТОЧКУ ЗАХОДА: видно, что элемент пойдёт не в лоб, а с фланга
                wp = env._axis_waypoint(live[0], tgt, x)
                mid = to_screen(wp)
                dashed(screen, col, src, mid, width=wide)
                dashed(screen, col, mid, dst, width=wide)
                pygame.draw.circle(screen, col, mid, 4, 1)
            else:
                dashed(screen, col, src, dst, width=wide)
            pygame.draw.circle(screen, col, dst, 8 if grp == "move" else 6, 2)
            if grp == "fire":                       # перечёркнутый круг = работать по этим
                pygame.draw.line(screen, col, (dst[0] - 8, dst[1] - 8), (dst[0] + 8, dst[1] + 8), 1)
        elif task == "ОТОЙТИ":
            back = -1 if side == 0 else 1
            dashed(screen, col, src, (src[0], src[1] - back * 55), width=wide)
        elif task == "РЕЗЕРВ":
            # РЕЗЕРВ отводит в глубину, поэтому рисуем КУДА именно — иначе он визуально
            # неотличим от ЗАКРЕПИТЬСЯ, а это разные приказы
            depth = wargame_env.RESERVE_DEPTH_M / wargame_env.M_PER_UNIT
            line = depth if side == 0 else wargame_env.ARENA - depth
            dst = to_screen(np.array([float(env.pos[live[0], 0]), line], dtype=np.float32))
            dashed(screen, col, src, dst, width=wide)
            pygame.draw.circle(screen, col, dst, 13, 1)
        else:                                       # ЗАКРЕПИТЬСЯ — стоим здесь
            pygame.draw.circle(screen, col, src, 13, 1)

    draw_mortars(screen, env, side)

    # сырой приказ «идти в точку» — отдельным цветом, он вне словаря
    if side == 0:
        for i in range(env.n_side):
            fp = env._free_point[i]
            if env.alive[i] and not np.isnan(fp[0]):
                a, b = to_screen(env.pos[i]), to_screen(fp)
                dashed(screen, ORDER_COLORS["free"], a, b, dash=5, gap=5, width=2)
                pygame.draw.line(screen, ORDER_COLORS["free"], (b[0] - 6, b[1] - 6), (b[0] + 6, b[1] + 6), 2)
                pygame.draw.line(screen, ORDER_COLORS["free"], (b[0] - 6, b[1] + 6), (b[0] + 6, b[1] - 6), 2)


RANGE_RINGS_M = (100, 250, 500, 1000, 1500, 2000, 2500)


class LosOverlay:
    """Инструмент видимости: что просматривается из точки + кольца дальностей.

    Считается РАДИАЛЬНОЙ РАЗВЁРТКОЙ (лучи из точки), а не перебором клеток с трассировкой в
    каждую: раньше на клетку приходился отдельный вызов blocked(), и панель ощутимо подтормаживала.
    Теперь всё считается numpy разом.

    Рисуем ЗАТЕМНЕНИЕ непросматриваемого, а не подсветку простреливаемого: так читается сильно
    лучше — видно «тени» за домами и лесом, как и должно выглядеть поле зрения."""

    def __init__(self):
        self.origin = None
        self.surf = None

    @staticmethod
    def sweep(env, origin, n_rays=540):
        """Для каждого луча — расстояние, на котором обзор перекрыт. Логика та же, что в
        TerrainMap.blocked: лес копится накопительно, здание рубит сразу."""
        m = env.map
        step = m.cell * 0.5
        nsteps = int(wargame_env.ARENA * 1.5 / step)
        ang = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
        s = (np.arange(1, nsteps + 1) * step).astype(np.float32)
        px = origin[0] + np.cos(ang)[:, None] * s[None, :]
        py = origin[1] + np.sin(ang)[:, None] * s[None, :]
        oob = (px < 0) | (px > m.width_m) | (py < 0) | (py > m.height_m)
        gx = np.clip((px / m.cell).astype(np.int32), 0, m.Gx - 1)
        gy = np.clip((py / m.cell).astype(np.int32), 0, m.Gy - 1)
        tid = m.grid[gx, gy]
        blk = m.blocks_by_id[tid]
        thr = m.see_through_by_id[tid]
        cum = np.cumsum(np.where(blk, step, 0.0), axis=1)      # накопленная толщина преграды
        cut = oob | (blk & (cum > thr))
        first = np.argmax(cut, axis=1)
        first[~cut.any(axis=1)] = nsteps - 1
        return ang, s[first]

    def get(self, env, p):
        key = (round(float(p[0]), 1), round(float(p[1]), 1))
        if key == self.origin and self.surf is not None:
            return self.surf
        ang, dist = self.sweep(env, p)
        surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
        surf.fill((6, 8, 14, 150))                              # тень на всё поле
        pts = [to_screen((p[0] + np.cos(a) * d, p[1] + np.sin(a) * d)) for a, d in zip(ang, dist)]
        pygame.draw.polygon(surf, (0, 0, 0, 0), pts)            # просматриваемое — вырезаем
        pygame.draw.polygon(surf, (250, 235, 170, 90), pts, 2)  # кромка обзора
        self.origin, self.surf = key, surf
        return surf


def draw_range_rings(screen, font, p):
    """Кольца дальностей вокруг точки — линейка для оценки, кто куда достаёт."""
    cx, cy = to_screen(p)
    for k, m_ in enumerate(RANGE_RINGS_M):
        r = int(m_ / wargame_env.M_PER_UNIT * SCALE)
        if r < 8 or r > SIZE * 1.2:
            continue
        col = (236, 200, 110) if m_ in (500, 1000) else (150, 170, 140)
        pygame.draw.circle(screen, col, (cx, cy), r, 2 if m_ in (500, 1000) else 1)
        lab = font.render(f"{m_} м" if m_ < 1000 else f"{m_ / 1000:g} км", True, col)
        screen.blit(lab, (cx + int(r * 0.7071) + 3, cy - int(r * 0.7071) - 14))
    pygame.draw.circle(screen, (250, 235, 170), (cx, cy), 4)
    pygame.draw.line(screen, (250, 235, 170), (cx - 9, cy), (cx + 9, cy), 1)
    pygame.draw.line(screen, (250, 235, 170), (cx, cy - 9), (cx, cy + 9), 1)


def unit_card(env, i):
    """Строки карточки юнита: то, что человеку нужно, чтобы принять решение."""
    live = env.live_weapons(i)
    lost = [env.weapons[i][k]["name"] for k in range(len(env.weapons[i])) if k not in live]
    side, e = element_of(env, i)
    t, o, x = (int(v) for v in env._orders[side, e])
    kind = TASK_OBJECT[TASKS[t]]
    if kind == "zone":
        obj = f" об.{min(o, max(env.n_zones - 1, 0)) + 1}"
    elif kind in ("friend", "enemy"):
        obj = f" {ELEMENT_NAMES[min(o, env.n_elements - 1)]}" + (" пр." if kind == "enemy" else "")
    else:
        obj = ""
    s = float(env.suppr[i])
    state = "ПРИЖАТ" if s >= wargame_env.SUPPR_PINNED else ("подавлен" if s > 0.4 else "спокоен")
    return [
        f"{env.types[i]}  [{'наши' if env.friendly[i] else 'противник'}]",
        f"элемент: {ELEMENT_NAMES[e]}",
        f"HP {env.hp[i]:.0f}/{env.max_hp[i]:.0f}   личный состав {env.crew_alive(i)}/{int(env.crew[i])}",
        f"стволы: {', '.join(env.weapons[i][k]['name'] for k in live) or '—'}",
        f"потеряны: {', '.join(lost) or '—'}",
        f"подавление {s:.2f}  ({state})",
        f"укрытие {env.map.cover_at(env.pos[i]):.2f}",
        f"приказ: {TASKS[t]}{obj}" + (f", {AXES[x].lower()}" if TASKS[t] in ("ОВЛАДЕТЬ", "ОБОЙТИ") else ""),
    ]


def list_replays(limit=14):
    """Записи, свежие сверху. Имя устроено как режим_дата_время_seed, поэтому читается как есть."""
    import glob as _g
    d = os.path.join(wargame_env.project_dir(), "replays")
    files = sorted(_g.glob(os.path.join(d, "*.npz")), key=os.path.getmtime, reverse=True)
    return files[:limit]


def draw_menu(screen, font, big, page, files, hover):
    """Меню по ESC. Возвращает список (прямоугольник, действие) — обработка в вызывающем цикле,
    чтобы отрисовка не знала о состоянии игры."""
    veil = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    veil.fill((8, 10, 12, 190))
    screen.blit(veil, (0, 0))

    if page == "main":
        items = [("continue", "ПРОДОЛЖИТЬ"), ("replays", "РЕПЛЕИ"),
                 ("restart", "НОВЫЙ БОЙ"), ("quit", "ВЫХОД")]
        w, rh = 380, 46
    else:
        items = [("back", "< НАЗАД")] + [(f"file:{p}", os.path.basename(p)[:-4]) for p in files]
        if not files:
            items.append(("none", "записей пока нет — сыграй бой, он сохранится сам"))
        w, rh = 640, 34

    h = 84 + len(items) * (rh + 6)
    x, y = (SIZE - w) // 2, max(40, (SIZE - h) // 2)
    pygame.draw.rect(screen, UI_BG, (x, y, w, h))
    pygame.draw.rect(screen, UI_AMBER, (x, y, w, h), 2)
    title = "МЕНЮ" if page == "main" else "РЕПЛЕИ  (свежие сверху)"
    screen.blit(big.render(title, True, UI_AMBER), (x + 20, y + 18))

    rects = []
    ry = y + 60
    for key, label in items:
        r = pygame.Rect(x + 16, ry, w - 32, rh)
        if key == hover and key not in ("none",):
            pygame.draw.rect(screen, (40, 58, 40), r)
            pygame.draw.rect(screen, UI_AMBER, r, 1)
        col = UI_DIM if key == "none" else (UI_AMBER if key == hover else UI_TEXT)
        f = big if page == "main" else font
        screen.blit(f.render(label, True, col), (r.x + 14, r.y + (rh - f.get_height()) // 2))
        rects.append((r, key))
        ry += rh + 6
    screen.blit(font.render("ESC — закрыть меню", True, UI_DIM), (x + 20, y + h - 26))
    return rects


def _draw_bar(screen, env, font, big, sel_elems, elem_rows, mx, my,
              head, msg, show_help, help_btn, pending_task, axis=1):
    """Нижняя панель. ВЫНЕСЕНА из main(), потому что её рисует и живая игра, и реплей: держать
    две копии — гарантированный способ развести их со временем."""
    # Раскладка: слева подразделения, в центре карточка ВЫБРАННОГО, справа ВСЕ его стволы.
    # Список задач больше не занимает колонку постоянно — он шпаргалка, и висит только пока
    # держишь H или пока ждём объект для приказа. Освободившееся место отдано под стволы:
    # раньше коробки шириной 172 начинались с x=700 и влезали ТОЛЬКО ДВЕ из четырёх —
    # РПГ и ГП-25 у стрелкового отделения просто не показывались.
    C1, C2, C3 = 14, 276, 516
    BOXW, BOXG = 132, 6

    def txt(s_, x, yy, col=UI_TEXT, f=font):
        screen.blit(f.render(s_, True, col), (x, yy)); return yy + f.get_height() + 2

    def gauge(x, yy, w, h, frac, col, back=(40, 46, 42)):
        pygame.draw.rect(screen, back, (x, yy, w, h))
        pygame.draw.rect(screen, col, (x, yy, int(w * max(0.0, min(1.0, frac))), h))
        pygame.draw.rect(screen, UI_EDGE, (x, yy, w, h), 1)

    # --- шапка над панелью
    screen.blit(big.render(head, True, UI_AMBER), (C1, BAR_Y - 26))
    own = env._zone_owner() if env.n_zones else []
    zs = "  ".join(f"O{k+1}:{'наш' if own[k]>0 else ('пр.' if own[k]<0 else '-')}" for k in range(len(own)))
    screen.blit(font.render(zs, True, UI_TEXT), (C1 + 300, BAR_Y - 24))
    screen.blit(font.render(msg, True, UI_AMBER), (C1 + 500, BAR_Y - 24))

    # --- колонка 1: подразделения
    y = BAR_Y + 10
    y = txt("ПОДРАЗДЕЛЕНИЯ", C1, y, UI_AMBER, big)
    elem_rows.clear()
    for e in range(env.n_elements):
        live = env._element_alive(0, e)
        t = int(env._orders[0, e, 0])
        pin = any(env.suppr[i] >= wargame_env.SUPPR_PINNED for i in live) if live else False
        elem_rows[e] = pygame.Rect(C1 - 6, y - 2, 250, 34)
        if e in sel_elems:
            pygame.draw.rect(screen, (34, 52, 46), elem_rows[e])
            pygame.draw.rect(screen, UI_AMBER, elem_rows[e], 1)
        col = UI_AMBER if e in sel_elems else ((255, 150, 60) if pin else UI_TEXT)
        screen.blit(font.render(CALLSIGNS[0][e], True, col), (C1, y))
        screen.blit(font.render(TASKS[t], True, UI_TEXT if e in sel_elems else UI_DIM), (C1 + 108, y))
        hp = float(np.mean([env.hp[i] / env.max_hp[i] for i in live])) if live else 0.0
        sp = float(np.mean([env.suppr[i] for i in live])) / wargame_env.SUPPR_PINNED if live else 0.0
        gauge(C1, y + 18, 110, 6, hp, (96, 200, 104))
        gauge(C1 + 118, y + 18, 110, 6, min(sp, 1.0), (236, 150, 60) if sp < 1 else (240, 70, 55))
        y += 36

    # --- что показываем: наведённое важнее выбранного, иначе выбранное (ВСЕГДА, не по наведению)
    i_show = unit_at(env, from_screen(mx, my)) if my < BAR_Y else -1
    if i_show < 0 and sel_elems:
        liv = env._element_alive(0, sorted(sel_elems)[0])
        i_show = liv[0] if liv else -1

    if i_show >= 0:
        side_i, e_i = element_of(env, i_show)
        card = pygame.Rect(C2, BAR_Y + 8, 228, BAR - 26)
        pygame.draw.rect(screen, UI_BOX, card)
        pygame.draw.rect(screen, UI_EDGE, card, 1)
        cx0, yy = C2 + 12, BAR_Y + 16
        play._nato_icon(screen, env.types[i_show], cx0 + 15, yy + 12, 14, 10,
                        BLUE if side_i == 0 else RED)
        screen.blit(big.render(CALLSIGNS[side_i][e_i], True, UI_AMBER), (cx0 + 40, yy + 2))
        screen.blit(font.render(ELEMENT_NAMES[e_i], True, UI_DIM), (cx0 + 40, yy + 22))
        yy += 46
        crew, cmax = env.crew_alive(i_show), int(env.crew[i_show])
        screen.blit(font.render(f"состав {crew}/{cmax}", True, UI_TEXT), (cx0, yy)); yy += 19
        strength_blocks(screen, cx0, yy, crew, cmax, bw=8, gap=3); yy += 22
        sv = float(env.suppr[i_show])
        if sv >= wargame_env.SUPPR_PINNED:
            st, sc = "ПРИЖАТ", UI_BAD
        elif sv > 0.4:
            st, sc = "ПОДАВЛЕН", UI_WARN
        else:
            st, sc = "СПОКОЕН", UI_OK
        screen.blit(big.render(st, True, sc), (cx0, yy)); yy += 24
        gauge(cx0, yy, 196, 6, sv / wargame_env.SUPPR_PINNED, sc); yy += 14
        seen = int(sum(1 for k in range(env.n)
                       if env.friendly[k] != env.friendly[i_show] and env.alive[k]
                       and env._acq[i_show, k] >= 1.0))
        yy = txt(f"укрытие {env.map.cover_at(env.pos[i_show]):.2f}   целей: {seen}", cx0, yy, UI_DIM)
        txt("В ДВИЖЕНИИ" if env._moved[i_show] else f"на месте {int(env._settle[i_show])}",
            cx0, yy, UI_WARN if env._moved[i_show] else UI_DIM)

        # --- ВСЕ стволы, каждый своей коробкой
        liveW = env.live_weapons(i_show)
        for wi, w in enumerate(env.weapons[i_show]):
            bx = C3 + wi * (BOXW + BOXG)
            if bx + BOXW > SIZE - 6:
                break
            if wi not in liveW:
                stw, col = "РАСЧЁТ ВЫБИТ", UI_BAD
            elif env.reload[i_show][wi] > 0:
                stw = f"{int(env.reload[i_show][wi] * wargame_env.SECONDS_PER_STEP)}с"
                col = UI_DIM
            elif w["deploy"] and env._settle[i_show] < w["deploy"]:
                stw, col = "РАЗВЁРТ.", UI_WARN
            else:
                stw, col = "ГОТОВ", UI_OK
            box = pygame.Rect(bx, BAR_Y + 8, BOXW, BAR - 26)
            pygame.draw.rect(screen, UI_BOX if wi in liveW else (34, 30, 30), box)
            pygame.draw.rect(screen, col if wi in liveW else UI_BAD, box, 1)
            weapon_icon(screen, w["name"], bx + 8, BAR_Y + 12, 62, 26,
                        col if wi in liveW else (110, 70, 70))
            screen.blit(font.render(stw, True, col), (bx + 74, BAR_Y + 18))
            screen.blit(big.render(w["name"], True,
                                   UI_AMBER if wi in liveW else (140, 100, 100)), (bx + 8, BAR_Y + 44))
            pygame.draw.line(screen, UI_EDGE, (bx + 6, BAR_Y + 68), (bx + BOXW - 6, BAR_Y + 68), 1)
            # БОЕКОМПЛЕКТ показываем В ПАТРОНАХ «текущий/полный», как в WRD: внутренние
            # «выстрелы» игроку ничего не говорят, а патроны — понятная величина.
            now_r = int(env.ammo[i_show][wi] * w["per_shot"])
            full_r = int(w["rounds"])
            acol = UI_OK if now_r > full_r * 0.4 else (UI_WARN if now_r > full_r * 0.15 else UI_BAD)
            screen.blit(font.render("боекомплект", True, UI_DIM), (bx + 8, BAR_Y + 74))
            av = font.render(f"{now_r}/{full_r}", True, acol if wi in liveW else (130, 100, 100))
            screen.blit(av, (bx + BOXW - 8 - av.get_width(), BAR_Y + 74))
            ry = BAR_Y + 74 + font.get_height() + 1
            for val, lab in ((f"{w['rng'] * wargame_env.M_PER_UNIT:.0f} м", "дальность"),
                             (f"{w['dmg']:.0f}", "урон"),
                             (f"{w['cd'] * wargame_env.SECONDS_PER_STEP:.0f} с", "перезаряд"),
                             (f"{w['min_crew']}", "расчёт от")):
                screen.blit(font.render(lab, True, UI_DIM), (bx + 8, ry))
                v = font.render(val, True, UI_TEXT if wi in liveW else (130, 100, 100))
                screen.blit(v, (bx + BOXW - 8 - v.get_width(), ry))
                ry += font.get_height() + 1
    else:
        # ничего не выбрано — вместо пустоты общая сводка по бою
        yy = txt("ОБСТАНОВКА", C2, BAR_Y + 10, UI_AMBER, big)
        yy = txt(f"наших подразделений: {sum(1 for e in range(env.n_elements) if env._element_alive(0, e))}"
                 f" из {env.n_elements}", C2, yy)
        yy = txt(f"у противника: {sum(1 for e in range(env.n_elements) if env._element_alive(1, e))}"
                 f" из {env.n_elements}", C2, yy)
        pin_n = int(sum(1 for i in range(env.n_side)
                        if env.alive[i] and env.suppr[i] >= wargame_env.SUPPR_PINNED))
        yy = txt(f"прижато наших: {pin_n}", C2, yy, UI_BAD if pin_n else UI_DIM)
        yy += 8
        txt("выбери подразделение: клик по значку,", C2, yy, UI_DIM)
        txt("по строке слева или рамкой по полю", C2, yy + 18, UI_DIM)

    # --- КИЛЛОГ в правом верхнем углу: кто, чем и по кому. Своих красим синим, чужих красным —
    # так с одного взгляда видно, в чью пользу идёт размен, без чтения самих строк.
    log = env.combat_log[-11:]
    if log:
        lw, lh = 330, 20 + len(log) * 17
        lx, ly = SIZE - lw - 10, help_btn.bottom + 10
        pane = pygame.Surface((lw, lh), pygame.SRCALPHA)
        pane.fill((*UI_BG, 190))
        screen.blit(pane, (lx, ly))
        pygame.draw.rect(screen, UI_EDGE, (lx, ly, lw, lh), 1)
        screen.blit(font.render("ДОНЕСЕНИЯ", True, UI_AMBER), (lx + 8, ly + 3))
        for k, ev in enumerate(reversed(log)):
            t = int(ev["step"] * wargame_env.SECONDS_PER_STEP)
            col = BLUE if ev["friendly_fire_side"] == 0 else RED
            fade = 1.0 if k < 4 else 0.62          # старые донесения приглушаем
            col = tuple(int(v * fade) for v in col)
            s_ = (f"{t//60:02d}:{t%60:02d} {ev['shooter']} ({ev['weapon']}) "
                  f"-> {ev['target']} -{ev['lost']}" + (" X" if ev["destroyed"] else ""))
            screen.blit(font.render(s_, True, col), (lx + 8, ly + 20 + k * 17))

    # --- КНОПКА СПРАВКИ в правом верхнем углу. Список задач больше не занимает место в панели:
    # это шпаргалка, нужная первые полчаса, а потом только мешающая.
    pygame.draw.rect(screen, UI_BG if not show_help else (44, 60, 42), help_btn)
    pygame.draw.rect(screen, UI_AMBER, help_btn, 2)
    bt = big.render("?  ПРИКАЗЫ", True, UI_AMBER)
    screen.blit(bt, (help_btn.centerx - bt.get_width() // 2,
                     help_btn.centery - bt.get_height() // 2))

    if show_help:
        ov = pygame.Rect(SIZE - 560, help_btn.bottom + 8, 548, 330)
        pane = pygame.Surface((ov.w, ov.h), pygame.SRCALPHA)
        pane.fill((*UI_BG, 248))
        screen.blit(pane, ov.topleft)
        pygame.draw.rect(screen, UI_AMBER, ov, 2)
        hy = ov.y + 10
        hy = txt("ЗАДАЧИ — выбери подразделение, затем цифру", ov.x + 12, hy, UI_AMBER, big)
        hy += 4
        for k, t in enumerate(TASKS):
            need = TASK_OBJECT[t]
            tag = {"zone": "объект", "friend": "своё подр.", "enemy": "подр. врага"}.get(need, "—")
            col = UI_AMBER if pending_task == k else UI_TEXT
            screen.blit(font.render(f"{k+1}", True, UI_AMBER), (ov.x + 12, hy))
            screen.blit(font.render(t, True, col), (ov.x + 32, hy))
            screen.blit(font.render(tag, True, UI_DIM), (ov.x + 140, hy))
            screen.blit(font.render(TASK_HELP[t], True, UI_DIM), (ov.x + 240, hy))
            hy += font.get_height() + 4
        hy += 6
        txt("заход Q/W/E — слева / фронтально / справа (для ОВЛАДЕТЬ и ОБОЙТИ)",
            ov.x + 12, hy, UI_DIM)

    # Строка подсказок — часть панели, поэтому живёт ЗДЕСЬ. Раньше она осталась в main() и
    # ссылалась на C1, уехавший сюда вместе с раскладкой: игра падала с NameError. Я это
    # пропустил, потому что все последующие проверки гоняли просмотр реплея, а не саму игру.
    screen.blit(font.render(
        f"заход {AXES[axis]} (Q/W/E)   ПКМ — в точку   L — видимость   "
        f"O/P — приказы   ПРОБЕЛ/+/-   R — заново", True, UI_DIM), (C1, SIZE - 20))



def replay_env(d):
    """Среда-«макет» по записи: нужна только чтобы отрисовщики могли спрашивать статы, местность
    и позывные. Шагов не делает — состояние в неё подставляется кадрами."""
    import terrain as _t
    env = wargame_env.WarGameEnv(action_mode="command", n_zones=int(d["n_zones"]), fog=bool(d["fog"]))
    env.reset(seed=int(d["seed"]))
    env.map = _t.from_grid(d["grid"], float(d["cell"]))
    env.zones = d["zones"]
    return env


def replay_events(d):
    """Донесения из записи обратно в вид, в котором их ждёт нижняя панель."""
    out = []
    for st, txt in zip(d["combat_step"], d["combat_txt"]):
        sh, wp, tg, lost, dead, sd = str(txt).split("|")
        out.append({"step": int(st), "shooter": sh, "weapon": wp, "target": tg,
                    "lost": int(lost), "destroyed": bool(int(dead)), "friendly_fire_side": int(sd)})
    return out


def replay_apply(env, d, idx, events):
    """Подставить записанный кадр в среду-макет."""
    env.pos[:] = d["pos"][idx]; env.hp[:] = d["hp"][idx]; env.alive[:] = d["alive"][idx]
    env.suppr[:] = d["suppr"][idx]; env._settle[:] = d["settle"][idx]
    env._moved[:] = d["moved"][idx]; env._orders[:] = d["orders"][idx]
    env._free_point[:] = d["free"][idx]; env._fire_point[:] = d["fire"][idx]
    env._acq[:] = d["acq"][idx]
    if "mortar" in d:                      # вызванный огонь: без него в реплее нет ни метки, ни вспышки
        m = np.asarray(d["mortar"][idx], dtype=np.float32)
        nq = 2 * wargame_env.MORTAR_ROUNDS * 3
        env._mortar_left[:] = m[0:2]
        env._mortar_aim[:] = m[2:6].reshape(2, 2)
        env._mortar_q[:] = m[6:6 + nq].reshape(2, wargame_env.MORTAR_ROUNDS, 3)
        env._mortar_boom[:] = m[6 + nq:6 + 2 * nq].reshape(2, wargame_env.MORTAR_ROUNDS, 3)
    flat = d["ammo"][idx]; off = 0
    for k in range(env.n):
        mlen = len(env.ammo[k]); env.ammo[k][:] = flat[off:off + mlen]; off += mlen
    env.steps = idx
    env.combat_log = [e for e in events if e["step"] <= idx]


def draw_field(screen, env, font, sel_elems=(), los_surface=None, rings_at=None):
    """ПОЛЕ БОЯ — единственная реализация на всех: живую игру, просмотр записи и рендер гифок.

    Раньше рендер гифок жил своим циклом в render_command.py и рисовал по-старому: без нижней
    панели, без приказов на карте, с легендой, где значились ПТУР, БМП и ОБТ, которых на
    взводном масштабе нет. Две копии одного отрисовщика неизбежно расходятся — сегодня пришлось
    добавлять вызов миномётов в оба места, а в следующий раз кто-то забудет."""
    screen.fill((14, 16, 20))
    draw_terrain(screen, env)
    draw_zones(screen, env, font)
    if los_surface is not None:
        screen.blit(los_surface, (0, 0))
    draw_fire_lines(screen, env)
    draw_orders(screen, env, set(sel_elems), side=0)
    draw_orders(screen, env, set(), side=1, dim=True)
    for e in sel_elems:
        for i in env._element_slots(0, e):
            if env.alive[i]:
                pygame.draw.circle(screen, (255, 255, 140), to_screen(env.pos[i]), 12, 2)
    for i in range(env.n):
        if env.alive[i]:
            draw_unit(screen, env, i, font)
    if rings_at is not None:
        draw_range_rings(screen, font, rings_at)


def render_replay_gif(path, out, fps=14, every=4):
    """Гифка ИЗ ЗАПИСИ, тем же отрисовщиком, что и живая игра."""
    import imageio
    import replay as rp
    d = rp.load(path)
    play.set_view(BAR_Y)
    pygame.init()
    screen = pygame.display.set_mode((W_TOTAL, SIZE))
    font = pygame.font.SysFont("consolas", 15)
    big = pygame.font.SysFont("consolas", 18, bold=True)
    env, events = replay_env(d), replay_events(d)
    n = len(d["pos"])
    frames = []
    for idx in range(0, n, max(every, 1)):
        replay_apply(env, d, idx, events)
        draw_field(screen, env, font)
        _draw_bar(screen, env, font, big, set(), {}, -1, -1,
                  head=f"{idx * wargame_env.SECONDS_PER_STEP / 60:5.1f} мин   кадр {idx}/{n-1}",
                  msg=str(d["source"]), show_help=False,
                  help_btn=pygame.Rect(0, 0, 0, 0), pending_task=None)
        frames.append(np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)))
    try:
        imageio.mimsave(out, frames, duration=1000.0 / fps, loop=0)
    except TypeError:
        imageio.mimsave(out, frames, fps=fps)
    pygame.quit()
    print(f"кадров: {len(frames)}  ->  {out}")


def run_replay(path, fps=30, standalone=True):
    """Просмотр записи. Отрисовка та же, что в живой игре — меняется только источник состояния:
    вместо шага симуляции подставляем записанный кадр. Поэтому карточка, приказы, линия видимости
    и донесения работают в реплее без единой отдельной строки кода."""
    import replay as rp
    d = rp.load(path)
    n_frames = len(d["pos"])
    play.set_view(BAR_Y)
    pygame.init()
    screen = pygame.display.set_mode((W_TOTAL, SIZE))
    pygame.display.set_caption(f"WarGame — реплей: {os.path.basename(path)}")
    font = pygame.font.SysFont("consolas", 15)
    big = pygame.font.SysFont("consolas", 18, bold=True)
    clock = pygame.time.Clock()
    los = LosOverlay()

    env, events = replay_env(d), replay_events(d)

    idx, paused, speed, sel_elems = 0, True, 1.0, set()
    acc = 0.0
    show_help = False
    help_btn = pygame.Rect(SIZE - 152, 10, 142, 32)
    elem_rows = {}
    while True:
        # Здесь НИЧЕГО не записываем: реплей уже проигрывается, записывать его заново незачем.
        for ev in pygame.event.get():
            # ESC возвращает В МЕНЮ, если просмотр запущен оттуда, и закрывает окно, если отдельно
            if ev.type == pygame.QUIT:
                if standalone:
                    pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if standalone:
                        pygame.quit()
                    return
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_RIGHT:
                    idx = min(idx + 1, n_frames - 1)
                if ev.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)
                if ev.key == pygame.K_HOME:
                    idx = 0
                if ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(speed * 1.6, 60)
                if ev.key == pygame.K_MINUS:
                    speed = max(speed / 1.6, 0.25)
                if ev.key == pygame.K_h:
                    show_help = not show_help
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if help_btn.collidepoint(ev.pos):
                    show_help = not show_help
                elif ev.pos[1] >= BAR_Y:
                    for e, rect in elem_rows.items():
                        if rect.collidepoint(ev.pos):
                            sel_elems = {e}
                else:
                    i = unit_at(env, from_screen(*ev.pos))
                    sel_elems = ({element_of(env, i)[1]} if i >= 0 and element_of(env, i)[0] == 0
                                 else set())

        if not paused and idx < n_frames - 1:
            acc += speed / max(fps, 1)
            step = int(acc)
            acc -= step
            idx = min(idx + step, n_frames - 1)

        replay_apply(env, d, idx, events)
        mx, my = pygame.mouse.get_pos()
        los_on = pygame.key.get_pressed()[pygame.K_l] and my < BAR_Y
        draw_field(screen, env, font, sel_elems,
                   los_surface=los.get(env, from_screen(mx, my)) if los_on else None,
                   rings_at=from_screen(mx, my) if los_on else None)

        _draw_bar(screen, env, font, big, sel_elems, elem_rows, mx, my,
                  head=f"РЕПЛЕЙ  кадр {idx}/{n_frames-1}   "
                       f"{idx*wargame_env.SECONDS_PER_STEP/60:5.1f} мин   "
                       + ("|| ПАУЗА" if paused else f">> x{speed:.1f}"),
                  msg=f"источник: {d['source']}   стрелки — покадрово, HOME — в начало",
                  show_help=show_help, help_btn=help_btn, pending_task=None)
        pygame.draw.rect(screen, UI_EDGE, (0, BAR_Y - 6, SIZE, 4), 1)
        pygame.draw.rect(screen, UI_AMBER, (0, BAR_Y - 6, int(SIZE * idx / max(n_frames - 1, 1)), 4))
        pygame.display.flip()
        clock.tick(fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["play", "watch"], default="play")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zones", type=int, default=3)
    ap.add_argument("--fixed-maps", action="store_true", help="реальные вырезки maps/platoon_crop_*")
    ap.add_argument("--map", default=None,
                    help="конкретная карта из maps/ по имени: и векторная (battle_21), и старая вырезка")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    fixed = []
    if args.map:
        # одна названная карта. Среда сама разберётся, что это: рядом лежат собранные поля —
        # берёт их, иначе читает старую сетку .npy
        fixed = [os.path.join(wargame_env.project_dir(), "maps", args.map)]
        print(f"карта: {args.map}")
    elif args.fixed_maps:
        import glob
        fixed = sorted(p[:-4] for p in glob.glob(
            os.path.join(wargame_env.project_dir(), "maps", "platoon_crop_*.npy")))
        print(f"вырезок: {len(fixed)}")

    env = wargame_env.WarGameEnv(action_mode="command", n_zones=args.zones, opponent="scripted",
                                 fixed_map_files=fixed, fixed_map_prob=1.0 if fixed else 0.0)

    play.set_view(BAR_Y)     # карта — только над панелью, иначе свои прячутся под интерфейсом
    pygame.init()
    screen = pygame.display.set_mode((W_TOTAL, SIZE))
    pygame.display.set_caption(f"WarGame — {'играешь за синих' if args.mode == 'play' else 'бой ботов'}")
    font = pygame.font.SysFont("consolas", 15)
    big = pygame.font.SysFont("consolas", 18, bold=True)
    clock = pygame.time.Clock()
    los = LosOverlay()

    seed = args.seed
    env.reset(seed=seed)
    env._opp_action = "scripted"

    sel_elems = set()        # ВЫБРАННЫЕ элементы — можно несколько (рамкой или Shift)
    drag_from = None         # начало рамки выделения
    # ЗАПИСЬ ИДЁТ ВСЕГДА: кадр весит копейки (46 КБ на двухчасовой бой), а потерять интересный
    # бой из-за того, что забыл включить запись, обидно. Сохраняется при конце боя, рестарте и выходе.
    rec_frames = [env.snapshot()]
    menu_open, menu_page, menu_rects = False, "main", []
    show_help = False        # справка по приказам — по кнопке в правом верхнем углу
    help_btn = pygame.Rect(SIZE - 152, 10, 142, 32)
    pending_task = None      # задача, ждущая объекта
    axis = 1
    paused = True
    # ШАГОВ В СЕКУНДУ, а не за кадр. Шаг = 10 с боя, поэтому 30 шагов/с (шаг за кадр при 30 fps)
    # прогоняли весь бой за 13 секунд реального времени — командовать было физически некогда.
    # 3 шага/с = ускорение x30 к реальному времени, бой ~2 минуты.
    speed = 3.0
    step_acc = 0.0
    elem_rows = {}           # прямоугольники строк элементов в панели — для выбора кликом
    step_once = False
    show_fire = True
    show_orders = True
    show_enemy_orders = False
    phys_left = 0
    msg = "ПРОБЕЛ — старт. ЛКМ по юниту — выбрать элемент."
    done = False

    while True:
        def _save_replay(why):
            if len(rec_frames) < 3:
                return None
            import replay as _rp
            path = _rp.auto_name(seed, tag="play" if args.mode == "play" else "watch")
            _rp.save_frames(path, env, rec_frames, f"{args.mode} ({why})", seed=seed)
            print(f"запись боя сохранена: {path}  ({len(rec_frames)} кадров)")
            print(f"  посмотреть:  py -3.12 replay.py play {path}")
            return path

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                _save_replay("выход"); pygame.quit(); return
            if menu_open and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for r, key in menu_rects:
                    if not r.collidepoint(ev.pos):
                        continue
                    if key == "continue":
                        menu_open = False
                    elif key == "replays":
                        menu_page = "replays"
                    elif key == "back":
                        menu_page = "main"
                    elif key == "restart":
                        _save_replay("рестарт"); rec_frames.clear()
                        seed += 1
                        env.reset(seed=seed); env._opp_action = "scripted"
                        rec_frames.append(env.snapshot())
                        sel_elems = set(); pending_task = None
                        done = False; phys_left = 0; menu_open = False
                        msg = f"новый бой, seed {seed}"
                    elif key == "quit":
                        _save_replay("выход"); pygame.quit(); return
                    elif key.startswith("file:"):
                        # просмотр прямо из меню: то же окно, по ESC вернёмся сюда
                        run_replay(key[5:], fps=args.fps, standalone=False)
                        play.set_view(BAR_Y)
                        pygame.display.set_caption(
                            f"WarGame — {'играешь за синих' if args.mode == 'play' else 'бой ботов'}")
                    break
                continue
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    # ESC = МЕНЮ, а не мгновенный выход: раньше случайное нажатие обрывало бой
                    menu_open = not menu_open
                    menu_page = "main"
                    if menu_open:
                        paused = True
                    continue
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_PERIOD:
                    step_once = True
                if ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(speed * 1.6, 30.0)
                if ev.key == pygame.K_MINUS:
                    speed = max(speed / 1.6, 0.5)
                if ev.key == pygame.K_TAB:
                    show_fire = not show_fire
                if ev.key == pygame.K_h:
                    show_help = not show_help
                if ev.key == pygame.K_o:
                    show_orders = not show_orders
                if ev.key == pygame.K_p:
                    show_enemy_orders = not show_enemy_orders
                    msg = f"приказы противника: {'видны' if show_enemy_orders else 'скрыты'}"
                if ev.key == pygame.K_r:
                    _save_replay("рестарт")
                    rec_frames.clear()
                    seed += 1
                    env.reset(seed=seed); env._opp_action = "scripted"
                    rec_frames.append(env.snapshot())
                    sel_elems = set(); pending_task = None; done = False; phys_left = 0
                    msg = f"новый бой, seed {seed}"
                if ev.key in (pygame.K_q, pygame.K_w, pygame.K_e):
                    axis = {pygame.K_q: 0, pygame.K_w: 1, pygame.K_e: 2}[ev.key]
                    msg = f"заход: {AXES[axis]}"
                if pygame.K_1 <= ev.key <= pygame.K_9 and sel_elems:
                    ti = ev.key - pygame.K_1
                    if ti < len(TASKS):
                        pending_task = ti
                        need = TASK_OBJECT[TASKS[ti]]
                        if need is None:
                            a = env._orders[0].copy()
                            for e in sel_elems:
                                a[e] = [ti, 0, axis]
                                for i in env._element_slots(0, e):
                                    env._free_point[i] = np.nan
                            env._set_orders(0, a)
                            pending_task = None
                            msg = f"{_who(sel_elems)}: {TASKS[ti]}"
                        else:
                            msg = f"{TASKS[ti]} — укажи объект ({'зону' if need == 'zone' else 'элемент'})"
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.pos[1] >= BAR_Y and ev.button == 1:
                for e, rect in elem_rows.items():      # строкой в панели; Shift — добавить
                    if rect.collidepoint(ev.pos):
                        if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                            sel_elems.symmetric_difference_update({e})
                        else:
                            sel_elems = {e}
                        pending_task = None
                        msg = (f"выбран {CALLSIGNS[0][sorted(sel_elems)[0]]}" if len(sel_elems)==1 else f"выбрано подразделений: {len(sel_elems)}")
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and help_btn.collidepoint(ev.pos):
                show_help = not show_help
                continue                       # клик по кнопке не должен уходить на карту
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and drag_from is not None:
                # РАМКА: берём все свои подразделения, чей хоть один живой юнит попал в прямоугольник
                x0, y0 = drag_from
                x1, y1 = ev.pos
                box = pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
                drag_from = None
                if box.w > 6 or box.h > 6:
                    picked = {e for e in range(env.n_elements)
                              for i in env._element_slots(0, e)
                              if env.alive[i] and box.collidepoint(to_screen(env.pos[i]))}
                    if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                        sel_elems |= picked
                    else:
                        sel_elems = picked
                    msg = f"рамкой выбрано подразделений: {len(sel_elems)}"
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.pos[1] < BAR_Y:
                p = from_screen(*ev.pos)
                if ev.button == 3 and sel_elems:
                    # СЫРОЙ приказ ОТМЕНЯЕТ текущую задачу, а не приостанавливает её: иначе после
                    # прихода в точку подразделение молча возвращалось к прежнему ОВЛАДЕТЬ и уходило
                    # обратно — игрок этого не приказывал и не ожидал.
                    a = env._orders[0].copy()
                    hold = TASKS.index("ЗАКРЕПИТЬСЯ")
                    for e in sel_elems:
                        a[e] = [hold, 0, 1]
                    env._set_orders(0, a)                          # сбрасывает и старые локи маршрута
                    for e in sel_elems:
                        for i in env._element_slots(0, e):
                            env._free_point[i] = p
                    msg = f"{_who(sel_elems)}: идти в точку (приказ отменён)"
                elif ev.button == 1:
                    if pending_task is not None and sel_elems:
                        need = TASK_OBJECT[TASKS[pending_task]]
                        obj = None
                        if need == "zone" and env.n_zones:
                            obj = int(np.argmin(np.linalg.norm(env.zones - p, axis=1)))
                        else:
                            i = unit_at(env, p)
                            if i >= 0:
                                s2, e2 = element_of(env, i)
                                if (need == "friend" and s2 == 0) or (need == "enemy" and s2 == 1):
                                    obj = e2
                        if obj is not None:
                            a = env._orders[0].copy()
                            for e in sel_elems:
                                a[e] = [pending_task, obj, axis]
                                for i in env._element_slots(0, e):
                                    env._free_point[i] = np.nan
                            env._set_orders(0, a)
                            msg = f"{_who(sel_elems)}: {TASKS[pending_task]}"
                            pending_task = None
                        else:
                            msg = "не тот объект для этой задачи"
                    else:
                        i = unit_at(env, p)
                        if i >= 0 and element_of(env, i)[0] == 0:
                            e2 = element_of(env, i)[1]
                            if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                                sel_elems.symmetric_difference_update({e2})
                            else:
                                sel_elems = {e2}
                            msg = (f"выбран {CALLSIGNS[0][sorted(sel_elems)[0]]}" if len(sel_elems)==1 else f"выбрано подразделений: {len(sel_elems)}")
                        else:
                            drag_from = ev.pos      # промах по юниту — тянем РАМКУ выделения
                            pending_task = None

        # ---- симуляция
        if not done and (not paused or step_once):
            if step_once:
                n = 1
            else:
                step_acc += speed / max(args.fps, 1)   # накапливаем дробные шаги по РЕАЛЬНОМУ времени
                n = int(step_acc)
                step_acc -= n
            for _ in range(n):
                if phys_left <= 0:                    # решение командиров раз в темп
                    if args.mode == "watch":
                        env._set_orders(0, env._scripted_commander(0))
                    env._set_orders(1, env._scripted_commander(1))
                    phys_left = wargame_env.COMMAND_TEMPO
                _, _, te, tr, info = env._physics_step(None)
                rec_frames.append(env.snapshot())
                phys_left -= 1
                if te or tr:
                    done = True
                    paused = True
                    f, en = info["friendly_alive"], info["enemy_alive"]
                    why = "время вышло" if tr and (f and en) else "противник разбит" if not en else "мы разбиты"
                    outcome = "ПОБЕДА" if f > en else ("ПОРАЖЕНИЕ" if en > f else "НИЧЬЯ")
                    msg = f"{outcome} ({why}): {f} : {en}.  R — новый бой"
                    _save_replay(outcome.lower())
                    break
            step_once = False

        # ---- отрисовка
        screen.fill((14, 16, 20))
        draw_terrain(screen, env)
        draw_zones(screen, env, font)
        mx, my = pygame.mouse.get_pos()
        los_on = pygame.key.get_pressed()[pygame.K_l] and my < BAR_Y
        if los_on:
            screen.blit(los.get(env, from_screen(mx, my)), (0, 0))
        if show_fire:
            draw_fire_lines(screen, env)
        if show_orders:
            draw_orders(screen, env, sel_elems, side=0)
            if show_enemy_orders:
                draw_orders(screen, env, set(), side=1, dim=True)
        for e in sel_elems:
            for i in env._element_slots(0, e):
                if env.alive[i]:
                    pygame.draw.circle(screen, (255, 255, 140), to_screen(env.pos[i]), 12, 2)
        for i in range(env.n):
            if env.alive[i]:
                draw_unit(screen, env, i, font)
                if env.suppr[i] >= wargame_env.SUPPR_PINNED:      # прижатых видно сразу
                    x, y = to_screen(env.pos[i])
                    screen.blit(font.render("!", True, (255, 210, 60)), (x + 7, y - 16))
        if los_on:                       # кольца дальностей — поверх юнитов, это линейка
            draw_range_rings(screen, font, from_screen(mx, my))
        if drag_from is not None:        # рамка выделения
            x0, y0 = drag_from
            box = pygame.Rect(min(x0, mx), min(y0, my), abs(mx - x0), abs(my - y0))
            sel_s = pygame.Surface((max(box.w, 1), max(box.h, 1)), pygame.SRCALPHA)
            sel_s.fill((255, 255, 140, 40))
            screen.blit(sel_s, box.topleft)
            pygame.draw.rect(screen, (255, 255, 140), box, 1)

        # ---- НИЖНЯЯ ПАНЕЛЬ в духе Wargame: Red Dragon (полупрозрачная полоса поверх поля)
        bar = pygame.Surface((SIZE, BAR), pygame.SRCALPHA)
        bar.fill((*UI_BG, 232))
        screen.blit(bar, (0, BAR_Y))
        pygame.draw.line(screen, UI_EDGE, (0, BAR_Y), (SIZE, BAR_Y), 2)
        mins = env.steps * wargame_env.SECONDS_PER_STEP / 60.0
        lim = wargame_env.MAX_STEPS * wargame_env.SECONDS_PER_STEP / 60.0
        _draw_bar(screen, env, font, big, sel_elems, elem_rows, mx, my,
                  head=f"{mins:5.1f} / {lim:.0f} мин   "
                       + ("|| ПАУЗА" if paused else f">> {speed:.1f} шаг/с"),
                  msg=msg, show_help=show_help, help_btn=help_btn,
                  pending_task=pending_task, axis=axis)

        if menu_open:
            menu_rects = draw_menu(screen, font, big, menu_page, list_replays(),
                                   next((k for r, k in menu_rects
                                         if r.collidepoint(pygame.mouse.get_pos())), None))
        if done:   # раньше окно просто замирало, и это выглядело как зависание
            box = pygame.Surface((560, 96)); box.set_alpha(230); box.fill((28, 30, 38))
            screen.blit(box, (SIZE // 2 - 280, SIZE // 2 - 48))
            pygame.draw.rect(screen, (255, 210, 90), (SIZE // 2 - 280, SIZE // 2 - 48, 560, 96), 2)
            t1 = big.render(msg, True, (255, 235, 160))
            t2 = font.render(f"бой длился {env.steps * wargame_env.SECONDS_PER_STEP / 60:.1f} мин "
                             f"(предел {wargame_env.MAX_STEPS * wargame_env.SECONDS_PER_STEP / 60:.0f})",
                             True, (200, 200, 210))
            screen.blit(t1, (SIZE // 2 - t1.get_width() // 2, SIZE // 2 - 32))
            screen.blit(t2, (SIZE // 2 - t2.get_width() // 2, SIZE // 2 + 4))
        pygame.display.flip()
        clock.tick(args.fps)


if __name__ == "__main__":
    main()
