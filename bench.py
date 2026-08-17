"""Сравнение политики с ТОЧКАМИ ОТСЧЁТА на одних и тех же сценариях.

Само по себе «win_rate 0.23» не значит ничего. Нужны две границы снизу и сверху:

  СЛУЧАЙНАЯ   — что даёт бессмысленное тыканье. Ниже неё модель не научилась ничему.
  СКРИПТОВАЯ  — тот же командир, против которого модель обучалась. Это и есть планка «сравнялась
                с правилами». Она обязательна: замер показал, что скриптовый командир слабый
                (объектов держит 0.19 из 3, половину приказов тратит на ЗАКРЕПИТЬСЯ), и случайная
                политика обыгрывает его по остаткам сил. Без обеих границ рост награды невозможно
                истолковать: непонятно, модель научилась командовать или просто перестала тупить.

Все политики гоняются по ОДИНАКОВЫМ seed'ам, поэтому разница считается парным тестом — так
дисперсия карты и стартовых позиций сокращается и нужно втрое меньше эпизодов.

    py -3.12 bench.py models/platoon_z3/best_model.zip --command --zones 3 --fixed-maps
    py -3.12 bench.py --command --zones 3 --episodes 40      # только точки отсчёта, без модели
"""
import argparse
import glob
import os

import numpy as np

import wargame_env


class _ScriptedPolicy:
    """Скриптовый командир как политика. Он живёт ВНУТРИ среды (_scripted_commander), поэтому
    здесь просто спрашиваем у неё приказ за синих — тот же интерфейс, что у модели и у человека."""
    def __init__(self, env_holder):
        self.env_holder = env_holder

    def __call__(self, obs):
        return self.env_holder[0]._scripted_commander(0)


class _ModelPolicy:
    """Обёртка над обученной моделью. Отдельный класс нужен из-за RecurrentPPO: у него между
    шагами надо протаскивать состояние LSTM и сбрасывать его на границе эпизода — иначе память
    молча не работает и модель ведёт себя как обычная MLP (замер получился бы заниженным)."""
    def __init__(self, path, env_holder=None):
        from play import load_any
        self.model = load_any(path)
        self.recurrent = type(self.model).__name__ == "RecurrentPPO"
        # МАСКИ ОБЯЗАТЕЛЬНЫ для MaskablePPO. Без них predict() разрешает недопустимые приказы,
        # и замер занижает модель: она обучалась в мире, где часть действий закрыта, а меряется
        # в мире, где открыто всё. Ровно эта ошибка сидела в EvalCallback во время обучения.
        self.maskable = type(self.model).__name__ == "MaskablePPO"
        self.env_holder = env_holder
        self.reset()

    def reset(self):
        self.state = None
        self.first = True

    def __call__(self, obs):
        if self.maskable:
            m = self.env_holder[0].action_masks() if self.env_holder else None
            return self.model.predict(obs, action_masks=m, deterministic=True)[0]
        if not self.recurrent:
            return self.model.predict(obs, deterministic=True)[0]
        a, self.state = self.model.predict(obs, state=self.state,
                                           episode_start=np.array([self.first]), deterministic=True)
        self.first = False
        return a


def run(policy, episodes, seeds, fixed, env_kw=None, holder=None):
    env = wargame_env.WarGameEnv(fixed_map_files=fixed, fixed_map_prob=1.0 if fixed else 0.0,
                                 **(env_kw or {}))
    if holder is not None:
        holder[0] = env                     # скриптовой политике нужна сама среда
    wins, rews, lens, surv, zones = [], [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        total, done = 0.0, False
        if hasattr(policy, "reset"):
            policy.reset()                 # сбросить состояние LSTM между эпизодами
        while not done:
            a = policy(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        wins.append(1.0 if info.get("win") else 0.0)
        rews.append(total)
        lens.append(env.steps)
        surv.append(info["friendly_alive"] - info["enemy_alive"])  # перевес сил на конце
        zones.append(float(info.get("zones_held", 0)))
    env.close()
    return (np.array(wins), np.array(rews), np.array(lens), np.array(surv), np.array(zones))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None, help="путь к .zip; без него — только случайная")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--fixed-maps", action="store_true", help="гонять на отобранных настоящих картах")
    ap.add_argument("--orders", action="store_true", help="режим приказов (как при обучении модели)")
    ap.add_argument("--command", action="store_true", help="режим командира (как обучается взводный)")
    ap.add_argument("--zones", type=int, default=0, help="число точек-объектов (как при обучении)")
    ap.add_argument("--fog", action="store_true", help="туман войны (как при обучении)")
    args = ap.parse_args()
    mode = "command" if args.command else ("orders" if args.orders else "velocity")
    env_kw = dict(action_mode=mode, n_zones=args.zones, fog=args.fog)
    if mode == "command":
        env_kw["opponent"] = "scripted"      # красными правит скрипт — как при обучении

    fixed = []
    if args.fixed_maps:
        fixed = wargame_env.map_pool()       # ОТОБРАННЫЕ вырезки, без вырожденных
        print(f"вырезок настоящих карт: {len(fixed)}")

    seeds = np.arange(args.episodes)
    probe = wargame_env.WarGameEnv(**env_kw)
    space = probe.action_space
    space.seed(0)
    probe.close()

    rows = [("случайная", run(lambda o: space.sample(), args.episodes, seeds, fixed, env_kw))]

    if mode == "command":
        holder = [None]
        rows.append(("скриптовая", run(_ScriptedPolicy(holder), args.episodes, seeds, fixed,
                                       env_kw, holder=holder)))

    if args.model:
        mh = [None]
        rows.append(("модель", run(_ModelPolicy(args.model, env_holder=mh), args.episodes, seeds,
                                   fixed, env_kw, holder=mh)))

    print(f"\n{args.episodes} эпизодов, одинаковые seed'ы, карты: {'реальные вырезки' if fixed else 'процедурные'}\n")
    print(f"{'политика':<12} {'win_rate':>18} {'ср.награда':>16} {'ср.длина':>10} "
          f"{'перевес сил':>12} {'объектов':>10}")
    for name, (w, r, l, sv, z) in rows:
        # 95% доверительный интервал доли побед (нормальное приближение)
        ci = 1.96 * np.sqrt(max(w.mean() * (1 - w.mean()), 1e-9) / len(w))
        print(f"{name:<12} {w.mean():>10.3f} ±{ci:<6.3f} {r.mean():>9.2f} ±{1.96*r.std()/np.sqrt(len(r)):<5.2f} "
              f"{l.mean():>10.0f} {sv.mean():>12.2f} {z.mean():>10.2f}")

    # ПАРНЫЕ сравнения с каждой точкой отсчёта: одни и те же seed'ы, поэтому вычитаем поэпизодно
    base = {n: v for n, v in rows}
    if "модель" in base:
        print()
        for ref in ("случайная", "скриптовая"):
            if ref not in base:
                continue
            d = base["модель"][1] - base[ref][1]
            se = d.std(ddof=1) / np.sqrt(len(d))
            verdict = ("ЛУЧШЕ" if d.mean() - 1.96 * se > 0
                       else ("ХУЖЕ" if d.mean() + 1.96 * se < 0 else "НЕ ОТЛИЧИМА"))
            print(f"модель против «{ref}»: разница награды {d.mean():+.2f} ±{1.96*se:.2f} -> {verdict}")
        print("\nЧитать так: не лучше СЛУЧАЙНОЙ — не научилась ничему; лучше случайной, но не"
              " лучше СКРИПТОВОЙ —\nучится, но не доросла до правил; лучше обеих — есть чему учить"
              " дальше, скриптовый противник исчерпан.")


if __name__ == "__main__":
    main()
