"""Проверка зеркалирования наблюдения для self-play.

Это самое опасное место всей затеи: ошибка в зеркалировании НЕ падает и НЕ видна в логах —
политика просто молча учится на кривых данных. Поэтому проверяем инвариант численно.

Инвариант: obs(side=1) в мире W должен ПОБИТОВО совпадать с obs(side=0) в мире, где
  * стороны поменяны местами,
  * карта отражена по оси Y,
  * все позиции отражены по Y.
То есть «глазами красных» = «глазами синих в отражённом мире». Если это так — политика,
обученная за синих, видит за красных ровно тот же тип картинки и не переучивается на геометрию.

    py -3.12 test_mirror.py
"""
import numpy as np

import terrain
import wargame_env as W


def check_sense_permutation():
    """Отдельно и в лоб: перестановка лучей препятствий при отражении по Y."""
    rng = np.random.default_rng(0)
    m = terrain.make_map(rng, W.ARENA)
    grid = m.grid
    mirrored = terrain.from_grid(grid[:, ::-1].copy(), m.cell)

    worst = 0.0
    for _ in range(200):
        p = rng.uniform(5, W.ARENA - 5, size=2).astype(np.float32)
        pm = np.array([p[0], m.height_m - p[1]], dtype=np.float32)
        a = m.sense_obstacles(p, False)[W._SENSE_YFLIP]   # лучи в мире, переставленные
        b = mirrored.sense_obstacles(pm, False)           # лучи в отражённом мире
        worst = max(worst, float(np.abs(a - b).max()))
    ok = worst < 1e-5
    print(f"[{'OK ' if ok else 'FAIL'}] перестановка лучей _SENSE_YFLIP: макс. расхождение {worst:.2e}")
    return ok


def build_mirrored_env(src):
    """Мир src, отражённый по Y и с поменянными сторонами."""
    dst = W.WarGameEnv(n_zones=src.n_zones, fog=src.fog, action_mode=src.action_mode)
    dst.reset(seed=0)
    if src.action_mode == "command":
        # приказы тоже меняются местами: то, что у src приказано красным, у dst — своим
        # при переносе в отражённый мир направление тоже переворачивается (лево <-> право)
        a0, a1 = src._orders[1].copy(), src._orders[0].copy()
        for a in (a0, a1):
            a[:, 2] = len(W.AXES) - 1 - a[:, 2]
        dst._orders[0], dst._orders[1] = a0, a1
        dst._order_age[0], dst._order_age[1] = src._order_age[1].copy(), src._order_age[0].copy()
    dst.map = terrain.from_grid(src.map.grid[:, ::-1].copy(), src.map.cell)

    def flip(p):
        return np.array([p[0], W.ARENA - p[1]], dtype=np.float32)

    ns = src.n_side
    for j in range(ns):
        dst.pos[j] = flip(src.pos[ns + j])          # красные src становятся «своими» в dst
        dst.pos[ns + j] = flip(src.pos[j])
        dst.hp[j], dst.hp[ns + j] = src.hp[ns + j], src.hp[j]
        dst.suppr[j], dst.suppr[ns + j] = src.suppr[ns + j], src.suppr[j]
        dst.alive[j], dst.alive[ns + j] = src.alive[ns + j], src.alive[j]
        dst.reload[j] = src.reload[ns + j].copy()
        dst.reload[ns + j] = src.reload[j].copy()
    if src.n_zones:
        dst.zones = np.stack([flip(z) for z in src.zones]).astype(np.float32)
    # ВАЖНО: у _last_seen надо не только отразить координаты, но и ПЕРЕСТАВИТЬ юниты — слот j
    # в dst соответствует слоту ns+j в src (как и для _vis ниже). Забыть перестановку легко.
    dst._last_seen = np.stack([
        np.stack([flip(p) for p in np.concatenate([src._last_seen[1 - s][ns:],
                                                   src._last_seen[1 - s][:ns]])])
        for s in (0, 1)])
    dst._vis = np.stack([np.concatenate([src._vis[1 - s][ns:], src._vis[1 - s][:ns]]) for s in (0, 1)])
    # Состояние, появившееся позже (обнаружение, боекомплект, изготовка), тоже надо переносить —
    # иначе тест сравнивает обстрелянный мир с нетронутым и падает не по делу.
    perm = np.concatenate([np.arange(ns, 2 * ns), np.arange(ns)])   # своих и чужих местами
    dst._acq = src._acq[np.ix_(perm, perm)].copy()
    dst._settle = src._settle[perm].copy()
    dst._moved = src._moved[perm].copy()
    dst.ammo = [src.ammo[perm[k]].copy() for k in range(src.n)]
    # Миномёты — состояние ПО СТОРОНАМ, значит при обмене сторон меняются и они. Точка вызова
    # вдобавок отражается по Y, как всё остальное на карте.
    dst._mortar_left[0], dst._mortar_left[1] = src._mortar_left[1], src._mortar_left[0]
    dst._mortar_aim[0] = flip(src._mortar_aim[1])
    dst._mortar_aim[1] = flip(src._mortar_aim[0])
    for a_, b_ in ((0, 1), (1, 0)):
        for r in range(W.MORTAR_ROUNDS):
            st, px, py = src._mortar_q[b_, r]
            dst._mortar_q[a_, r] = (st, px, W.ARENA - py) if st >= 0 else (-1.0, 0.0, 0.0)
    return dst


def check_obs_invariant(n_zones=0, fog=False, mode="velocity", trials=6):
    tag = f"зоны={n_zones} туман={fog} режим={mode}"
    worst = 0.0
    for t in range(trials):
        src = W.WarGameEnv(n_zones=n_zones, fog=fog, action_mode=mode)
        src.reset(seed=100 + t)
        # ВАЖНО: reset(seed=) сеет env.np_random, но НЕ action_space — без явного seed траектория
        # каждый раз своя и падение теста невоспроизводимо (наступали на это).
        src.action_space.seed(100 + t)
        for _ in range(6):                     # разболтать состояние, чтобы оно было несимметричным
            src.step(src.action_space.sample())
        dst = build_mirrored_env(src)
        a = src._obs(side=1)
        b = dst._obs(side=0)
        worst = max(worst, float(np.abs(a - b).max()))
    ok = worst < 1e-5
    print(f"[{'OK ' if ok else 'FAIL'}] инвариант obs ({tag}): макс. расхождение {worst:.2e}")
    return ok


def check_selfplay_fairness(episodes=40, mode="velocity", n_zones=0):
    """Сквозная проверка: если ОБЕИМИ сторонами правит одна и та же (случайная) политика,
    доля побед обязана быть около 50%. Систематический перекос = ошибка в зеркалировании.

    Режим "command" здесь ВАЖЕН отдельно: только он проверяет переворот лево/право. Инвариант
    obs его не ловит, если среда и тест ошибаются согласованно — а они писались одной рукой."""
    space = W.WarGameEnv(action_mode=mode, n_zones=n_zones).action_space
    space.seed(0)
    env = W.WarGameEnv(action_mode=mode, n_zones=n_zones, opponent=lambda obs: space.sample())
    wins = losses = draws = 0
    for s in range(episodes):
        env.reset(seed=s)
        done = False
        while not done:
            _, _, te, tr, info = env.step(space.sample())
            done = te or tr
        if info["friendly_alive"] > info["enemy_alive"]:
            wins += 1
        elif info["friendly_alive"] < info["enemy_alive"]:
            losses += 1
        else:
            draws += 1
    # биномиальный ДИ по решающим партиям
    dec = wins + losses
    frac = wins / dec if dec else 0.5
    ci = 1.96 * np.sqrt(max(frac * (1 - frac), 1e-9) / max(dec, 1))
    # При малом числе РЕШАЮЩИХ партий доля бессмысленна: со взводом из четырёх подразделений
    # случайная политика почти всегда доводит до ничьей, и 0/2 выглядит как жуткий перекос,
    # хотя выборки просто нет. Требуем минимум решающих партий, иначе честно говорим «мало данных».
    if dec < 8:
        print(f"[--- ] честность self-play ({mode}): решающих партий всего {dec} из {episodes} — "
              f"выборки недостаточно, проверка пропущена")
        return True
    ok = abs(frac - 0.5) < max(ci, 0.02) + 0.15
    print(f"[{'OK ' if ok else 'FAIL'}] честность self-play ({mode}): перевес у синих в {wins}, у красных в {losses}, "
          f"поровну {draws} -> доля {frac:.2f} ±{ci:.2f} (ждём ~0.50)")
    return ok


def check_player_only_masked(episodes=6):
    """Задачи из PLAYER_ONLY_TASKS не должны быть доступны модели НИ В ОДНОМ состоянии.

    Проверяем прогоном, а не чтением кода: маска собирается из нескольких веток (обычная,
    прижатый элемент, запасная «хоть что-то доступно»), и достаточно одной, где фильтр забыли.
    Молча это не всплывёт — агент просто получит мёртвое действие обратно."""
    idx = [W.TASKS.index(t) for t in W.PLAYER_ONLY_TASKS]
    if not idx:
        print("[--- ] PLAYER_ONLY_TASKS пуст — проверять нечего")
        return True
    bad = 0
    for s in range(episodes):
        env = W.WarGameEnv(action_mode="command", n_zones=3, opponent="scripted")
        env.reset(seed=s)
        env.action_space.seed(s)
        done = False
        while not done:
            m = env.action_masks()
            for e in range(env.n_elements):
                off = int(env.action_space.nvec[:e * 3].sum())
                bad += sum(int(m[off + t]) for t in idx)
            _, _, te, tr, _ = env.step(env.action_space.sample())
            done = te or tr
        env.close()
    ok = bad == 0
    print(f"[{'OK ' if ok else 'FAIL'}] приказы игрока скрыты от модели "
          f"({', '.join(W.PLAYER_ONLY_TASKS)}): доступны в {bad} случаях, ждём 0")
    return ok


if __name__ == "__main__":
    results = [
        check_player_only_masked(),
        check_sense_permutation(),
        check_obs_invariant(),
        check_obs_invariant(n_zones=3),
        check_obs_invariant(fog=True),
        check_obs_invariant(n_zones=3, fog=True),
        check_obs_invariant(n_zones=3, mode="command"),
        check_obs_invariant(n_zones=3, fog=True, mode="command"),
        check_selfplay_fairness(),
        check_selfplay_fairness(mode="command", n_zones=3),
    ]
    print("\nИТОГ:", "всё сошлось" if all(results) else "ЕСТЬ ПАДЕНИЯ — зеркалирование чинить до обучения")
    raise SystemExit(0 if all(results) else 1)
