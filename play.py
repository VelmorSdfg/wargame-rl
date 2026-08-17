"""
Показ боя обученного командира против скриптового бота.

Свои — синие, враги — красные. Форма по типу:
  круг = пехота, круг с точкой = ПТ-пехота, квадрат = средний танк,
  большой квадрат = тяжёлый танк, ромб = артиллерия.
Над юнитом — полоска HP; жёлтое кольцо = подавлен; тонкая линия = ведёт огонь по цели.

Запуск:
    python play.py models/best_model.zip
    python play.py models/best_model.zip --episodes 5 --fps 20
"""

import argparse
import os

import numpy as np
import pygame
from stable_baselines3 import PPO, SAC

import wargame_env
from wargame_env import ARENA

SIZE = 1080  # выше разрешение холста -> можно сделать иконку МЕНЬШЕ в метрах, не теряя пиксельную видимость
SCALE = SIZE / ARENA
# Высота ОБЛАСТИ КАРТЫ на экране. По умолчанию весь холст, но game.py уменьшает её под нижнюю
# панель: иначе свои подразделения (спавн на игровой y~12, экранный ~950) оказывались ПОД
# интерфейсом — их не было ни видно, ни кликнуть.
VIEW_H = SIZE
VIEW_X = 0            # сдвиг вправо, чтобы карта осталась по центру при уменьшении


def set_view(height):
    """Вписать поле в полосу заданной высоты, сохранив квадратность."""
    global VIEW_H, SCALE, VIEW_X
    VIEW_H = int(height)
    SCALE = VIEW_H / ARENA
    VIEW_X = (SIZE - VIEW_H) // 2
BLUE = (90, 150, 255)
RED = (255, 100, 90)

NAMES = {
    "infantry": "пехота — отделение, броню не берёт",
    "at_infantry": "ПТУР — бьёт технику ИЗДАЛЕКА, хрупкий",
    "ifv": "БМП — слабо бьёт ОБТ",
    "mbt": "ОБТ — бьёт всё",
}
UNIT_ORDER = ["infantry", "at_infantry", "ifv", "mbt"]


def load_any(path):
    algos = [PPO, SAC]
    try:
        from sb3_contrib import RecurrentPPO      # модели из train_recurrent.py
        algos.insert(0, RecurrentPPO)
    except ImportError:
        pass
    try:
        # MaskablePPO ПЕРВЫМ: им обучается взводный (train_platoon.py), и это самый частый случай.
        # Порядок важен — у него своя политика (MaskableActorCriticPolicy), и попытка загрузить
        # такой файл обычным PPO падает на неизвестном аргументе use_sde. Ошибка была не в
        # алгоритме, а в том, что его просто не было в списке: бенч не мог померить модель.
        from sb3_contrib import MaskablePPO
        algos.insert(0, MaskablePPO)
    except ImportError:
        pass
    if not (os.path.exists(path) or os.path.exists(path + ".zip")):
        raise FileNotFoundError(f"Файла модели нет: {path}\n"
                                f"(best_model.zip появляется только после первой оценки EvalCallback — "
                                f"на коротком прогоне его может не быть, бери *_final.zip)")
    errors = {}
    for algo in algos:
        try:
            return algo.load(path, device="cpu")
        except Exception as ex:                     # раньше ошибки молча глотались и причина терялась
            errors[algo.__name__] = f"{type(ex).__name__}: {ex}"
    detail = "\n".join(f"  {k}: {v}" for k, v in errors.items())
    raise RuntimeError(f"Не удалось загрузить модель {path} ни одним алгоритмом:\n{detail}")


def draw_zones(screen, env, font):
    """Точки-объекты: цвет = кто держит, кольцо = радиус захвата."""
    if not getattr(env, "n_zones", 0):
        return
    own = env._zone_owner()
    for z in range(env.n_zones):
        c = (90, 130, 210) if own[z] > 0 else ((205, 95, 90) if own[z] < 0 else (150, 150, 150))
        pygame.draw.circle(screen, c, to_screen(env.zones[z]), int(wargame_env.ZONE_RADIUS * SCALE), 2)
        x, y = to_screen(env.zones[z])
        screen.blit(font.render(f"O{z + 1}", True, c), (x - 8, y - 8))


def to_screen(p):
    return int(p[0] * SCALE) + VIEW_X, int(VIEW_H - p[1] * SCALE)


# Цвета разведены СИЛЬНО. Раньше открытое (52,58,40) и лес (30,55,32) отличались всего на 33
# единицы суммарно — глазом неразличимо, и казалось, что леса на карте нет вовсе, хотя механика
# работала (укрытие 0.5, скорость 0.6). Плюс лес рисуется с текстурой, см. draw_terrain.
TERRAIN_COLORS = {0: (92, 96, 62), 1: (26, 58, 30), 2: (78, 76, 84), 3: (28, 52, 88), 4: (140, 124, 74)}
# 0=открытое поле (оливковый), 1=лес (зелёный), 2=здание (серый), 3=вода (синий), 4=дорога (охра)
# занятое здание — бледный цвет команды: свои (синий), враг (красный), спорное
GARRISON_COLORS = {"f": (58, 78, 120), "e": (120, 66, 62), "b": (110, 80, 110)}
SQUAD_FRONT_M = 30.0   # условный реальный фронт подразделения; от него считается размер значка


def _garrison_by_building(env, m):
    """comp здания -> 'f'/'e'/'b' по тому, чья пехота внутри (для подсветки)."""
    occ = {}
    for i in range(env.n):
        if not env.alive[i]:
            continue
        comp = m.component_at(env.pos[i])
        if comp <= 0:
            continue
        s = "f" if env.friendly[i] else "e"
        occ[comp] = "b" if (comp in occ and occ[comp] != s) else occ.get(comp, s)
    return occ


def draw_terrain(screen, env):
    """Рисует тайлы местности; занятые пехотой здания подсвечиваются цветом команды (бледно)."""
    m = getattr(env, "map", None)
    if m is None:
        return
    cs = m.cell * SCALE
    occ = _garrison_by_building(env, m)
    for gx in range(m.Gx):
        for gy in range(m.Gy):
            tid = int(m.grid[gx, gy])
            c = TERRAIN_COLORS.get(tid)
            if c is None:
                continue
            if tid == 2:  # здание: если занято — красим в цвет команды
                side = occ.get(int(m.building_comp[gx, gy]))
                if side:
                    c = GARRISON_COLORS[side]
            # ВАЖНО: та же система координат, что у to_screen. Раньше здесь стоял SIZE напрямую,
            # и после переноса интерфейса вниз (карта ужалась до VIEW_H) местность осталась в
            # старом кадре, а юниты уехали в новый — картинка «поплыла».
            left = gx * cs + VIEW_X
            top = VIEW_H - (gy + 1) * cs
            pygame.draw.rect(screen, c, (left, top, cs + 1, cs + 1))
            if tid == 1 and cs >= 8:   # лес — крапом, чтобы читался как лес, а не «просто темнее»
                pygame.draw.circle(screen, (16, 40, 20), (int(left + cs * 0.32), int(top + cs * 0.34)),
                                   max(1, int(cs * 0.13)))
                pygame.draw.circle(screen, (16, 40, 20), (int(left + cs * 0.70), int(top + cs * 0.68)),
                                   max(1, int(cs * 0.11)))


def _nato_icon(screen, t, cx, cy, w, h, color):
    """Рисует значок рода войск (NATO APP-6) в центре рамки."""
    ix, iy = max(2, int(w * 0.75)), max(2, int(h * 0.67))  # отступ значка от рамки (пропорционально размеру)
    if t == "infantry":                       # пехота — крест X
        pygame.draw.line(screen, color, (cx - ix, cy - iy), (cx + ix, cy + iy), 2)
        pygame.draw.line(screen, color, (cx - ix, cy + iy), (cx + ix, cy - iy), 2)
    elif t == "at_infantry":                  # ПТ — уголок Λ вверх
        pygame.draw.lines(screen, color, False,
                          [(cx - ix, cy + iy), (cx, cy - iy), (cx + ix, cy + iy)], 2)
    elif t == "ifv":                          # БМП — мех-пехота: овал + крест внутри
        pygame.draw.ellipse(screen, color, (cx - ix, cy - iy // 2, 2 * ix, iy), 2)
        pygame.draw.line(screen, color, (cx - ix + 2, cy - iy // 2 + 1), (cx + ix - 2, cy + iy // 2 - 1), 2)
        pygame.draw.line(screen, color, (cx - ix + 2, cy + iy // 2 - 1), (cx + ix - 2, cy - iy // 2 + 1), 2)
    elif t == "mbt":                          # ОБТ — залитый овал (броня)
        pygame.draw.ellipse(screen, color, (cx - ix, cy - iy // 2, 2 * ix, iy), 0)


def icon_size():
    """Размер значка. Привязан к МАСШТАБУ карты, а не задан числом: подразделение занимает
    примерно фиксированный реальный фронт, поэтому на взводном поле (15 м/ед) значок крупный,
    а на ротном (70 м/ед) — мелкий, иначе двадцать значков сливаются в кашу. Нижняя граница —
    предел читаемости, верхняя — чтобы значок не заслонял местность."""
    w = int(np.clip(SQUAD_FRONT_M / wargame_env.M_PER_UNIT * SCALE / 2, 5, 16))
    return w, max(3, int(w * 0.72))


def draw_unit(screen, env, i, font=None):
    # Пехоту в здании раньше рисовали по ЦЕНТРУ здания, чтобы значок ложился на дом. Для домика
    # 30x30 это незаметно, но на настоящих картах застройка слипается в компоненты 390x270 м, и
    # значок уезжал от бойцов на пару сотен метров. Наглядный симптом: сам значок стоит на месте,
    # а линии огня других отрядов (они идут к РЕАЛЬНОЙ позиции) ползают туда-сюда — выглядит как
    # дёргающаяся метка цели. Поэтому центрируем ТОЛЬКО на мелких строениях, где это косметика.
    dpos = env.pos[i]
    m = getattr(env, "map", None)
    if m is not None and env.armor[i] == 0:
        comp = m.component_at(env.pos[i])
        if comp > 0 and comp in m.building_centers:
            c = m.building_centers[comp]
            if abs(c[0] - dpos[0]) <= m.cell and abs(c[1] - dpos[1]) <= m.cell:
                dpos = c
    x, y = to_screen(dpos)
    friendly = bool(env.friendly[i])
    color = BLUE if friendly else RED
    t = env.types[i]
    w, h = icon_size()

    # РАМКА: свои — прямоугольник, враг — ромб (ключевое различие в NATO-символике)
    if friendly:
        pygame.draw.rect(screen, (24, 26, 32), (x - w, y - h, 2 * w, 2 * h))
        pygame.draw.rect(screen, color, (x - w, y - h, 2 * w, 2 * h), 2)
    else:
        d = [(x, y - h - 3), (x + w + 3, y), (x, y + h + 3), (x - w - 3, y)]
        pygame.draw.polygon(screen, (24, 26, 32), d)
        pygame.draw.polygon(screen, color, d, 2)

    _nato_icon(screen, t, x, y, w, h, (245, 245, 245))

    bw = 2 * w
    # ПОЛОСКА МОРАЛИ (верхняя). Как в Steel Division 2 — ЗАПОЛНЯЕТСЯ под огнём, а не убывает:
    # пустая = спокоен, полная = прижат. Цвет темнеет к красному по мере набора.
    s = float(np.clip(env.suppr[i] / wargame_env.SUPPR_PINNED, 0.0, 1.0))
    pygame.draw.rect(screen, (48, 48, 52), (x - w, y - h - 13, bw, 4))
    if s > 0.02:
        col = (255, 225, 70) if s < 0.6 else ((255, 160, 40) if s < 1.0 else (255, 70, 50))
        pygame.draw.rect(screen, col, (x - w, y - h - 13, int(bw * s), 4))
    if env.suppr[i] >= wargame_env.SUPPR_PINNED:      # прижат — обвести, чтобы било в глаза
        pygame.draw.rect(screen, (255, 70, 50), (x - w - 1, y - h - 14, bw + 2, 6), 1)

    # ПОЛОСКА HP (нижняя): зелёная -> жёлтая -> красная. Для пехоты это ещё и личный состав,
    # а значит какие стволы уже потеряны (см. crew_alive/min_crew).
    frac = float(env.hp[i] / env.max_hp[i])
    hcol = (80, 220, 80) if frac > 0.6 else ((230, 200, 60) if frac > 0.3 else (230, 80, 60))
    pygame.draw.rect(screen, (48, 48, 52), (x - w, y - h - 7, bw, 4))
    pygame.draw.rect(screen, hcol, (x - w, y - h - 7, int(bw * frac), 4))


def draw_legend(screen, font):
    """Легенда: NATO-значки + что означают, и кто по цвету/рамке."""
    panel = pygame.Surface((330, 190)); panel.set_alpha(225); panel.fill((30, 32, 38))
    screen.blit(panel, (8, 30))
    screen.blit(font.render("свои = синий ПРЯМОУГОЛЬНИК", True, BLUE), (16, 36))
    screen.blit(font.render("враг = красный РОМБ", True, RED), (16, 56))
    for k, t in enumerate(UNIT_ORDER):
        yy = 82 + k * 22
        pygame.draw.rect(screen, (200, 200, 200), (18, yy, 30, 18), 1)
        _nato_icon(screen, t, 33, yy + 9, 13, 9, (230, 230, 230))
        screen.blit(font.render(NAMES[t], True, (210, 210, 210)), (58, yy))


def draw_fire_lines(screen, env):
    """Линии огня только к РЕАЛЬНЫМ целям стволов (учёт приоритета/vs_soft/LOF)."""
    for i in range(env.n):
        c = (60, 110, 200) if env.friendly[i] else (200, 70, 60)
        for k in env._weapon_targets(i):
            pygame.draw.line(screen, c, to_screen(env.pos[i]), to_screen(env.pos[k]), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--fixed-maps", action="store_true",
                        help="играть на РЕАЛЬНЫХ вырезках maps/tactical_crop_* вместо процедурной карты")
    parser.add_argument("--orders", action="store_true", help="модель обучена в режиме приказов")
    parser.add_argument("--zones", type=int, default=0, help="число точек-объектов (как при обучении)")
    parser.add_argument("--fog", action="store_true", help="туман войны (как при обучении)")
    args = parser.parse_args()

    fixed = []
    if args.fixed_maps:
        import glob
        fixed = sorted(p[:-4] for p in glob.glob(
            os.path.join(wargame_env.project_dir(), "maps", "tactical_crop_*.npy")))
        print(f"вырезок: {len(fixed)}")
    env = wargame_env.WarGameEnv(fixed_map_files=fixed, fixed_map_prob=1.0 if fixed else 0.0,
                                 action_mode="orders" if args.orders else "velocity",
                                 n_zones=args.zones, fog=args.fog)
    model = load_any(args.model_path)

    pygame.init()
    screen = pygame.display.set_mode((SIZE, SIZE))
    pygame.display.set_caption("WarGame — синие (агент) против красных (бот)")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        result = ""
        lstm_state = None            # у RecurrentPPO надо протаскивать состояние между шагами,
        first = True                 # иначе память не работает и модель ведёт себя как обычная MLP
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
            if type(model).__name__ == "RecurrentPPO":
                action, lstm_state = model.predict(obs, state=lstm_state,
                                                   episode_start=np.array([first]), deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            first = False
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            screen.fill((18, 20, 24))
            draw_terrain(screen, env)
            draw_zones(screen, env, font)
            draw_fire_lines(screen, env)
            for i in range(env.n):
                if env.alive[i]:
                    draw_unit(screen, env, i)
            hud = f"эпизод {ep + 1}/{args.episodes}  свои: {info['friendly_alive']}  враги: {info['enemy_alive']}  шаг {env.steps}"
            screen.blit(font.render(hud, True, (230, 230, 230)), (10, 8))
            pygame.display.flip()
            clock.tick(args.fps)

            if done:
                result = "ПОБЕДА" if info.get("win") else ("поражение" if info["friendly_alive"] == 0 else "ничья (таймаут)")
        # показать исход на паузе
        txt = font.render(f"Эпизод {ep + 1}: {result}", True, (255, 255, 120))
        screen.blit(txt, (SIZE // 2 - 100, SIZE // 2))
        pygame.display.flip()
        pygame.time.wait(1200)

    pygame.quit()


if __name__ == "__main__":
    main()
