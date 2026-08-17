"""Обучение ВЗВОДНОГО КОМАНДИРА против скриптового взводного того же уровня.

Чем отличается от train_command.py (тот писался под ротный масштаб и устарел):
  * противник — не реактивный побочный бот, а СКРИПТОВЫЙ КОМАНДИР с тем же словарём приказов
    и тем же темпом. Реактивный бот не знал про объекты и решал 720 раз за бой против 60 у
    агента — сравнение было бесчестным, а обучение шло против форы в скорости реакции;
  * MaskablePPO: маскирование недопустимых приказов. В Gym-uRTS это назвали одной из ДВУХ вещей,
    без которых не заработало (вторая — композиция действий, у нас это приказы элементам);
  * периодически печатается ПОТОК ПРИКАЗОВ и сводка потерь — по win_rate не понять, осмысленный
    ли замысел, а именно осмысленность и есть цель.

ЧТО СМОТРЕТЬ. Условие победы у нас — полное уничтожение противника, событие редкое, поэтому
win_rate плохой сигнал (на прошлых прогонах он равнялся случайному даже когда модель училась).
Главное — eval/mean_reward и перевес сил из bench.py.

    py -3.12 train_platoon.py --timesteps 300000            # разведочный, ~3 ч на 16 ядрах
    py -3.12 train_platoon.py --timesteps 2000000 --fixed-map-prob 0.5

СКОЛЬКО ЭТО ИДЁТ. Шаг среды ~150 мс — почти весь он уходит на лучевую трассировку обнаружения,
поэтому упор в ядра, а не в питон и не в сеть. Замер: 8 сред 27 шаг/с, 12 — 37, 16 — 42; с учётом
обновлений политики и оценок выходит ~28 шаг/с, то есть 300k ≈ 3 часа. Ещё ~17% времени съедают
периодические оценки (12 штук по 15 боёв на ОДНОЙ среде) — если тесно, режь n_eval_episodes.
"""

import argparse
import os
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import wargame_env

RUN_DIR = wargame_env.project_dir()
GAMMA = 0.995        # выше обычного: последствия приказа видны нескоро (решение раз в 2 мин боя)
ROLLOUT = 4096       # сколько шагов собирается между обновлениями политики (n_steps * n_envs)


class WinRateCallback(BaseCallback):
    """Доля побед и ПЕРЕВЕС СИЛ. Перевес важнее: он показывает, что агент учится, ещё до того,
    как научится добивать — ровно так и было на прошлых прогонах."""
    def __init__(self):
        super().__init__()
        self.wins, self.edge = [], []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "win" in info:
                self.wins.append(1.0 if info["win"] else 0.0)
                self.edge.append(float(info["friendly_alive"] - info["enemy_alive"]))
                if len(self.wins) > 200:
                    self.wins.pop(0)
                    self.edge.pop(0)
        if self.wins:
            self.logger.record("rollout/win_rate", float(np.mean(self.wins)))
            self.logger.record("rollout/force_edge", float(np.mean(self.edge)))
        return True


class ReportCallback(BaseCallback):
    """Раз в N шагов прогоняет один бой и печатает приказы и донесения — читать глазами."""
    def __init__(self, env_kw, every=50_000):
        super().__init__()
        self.env_kw, self.every, self._next = env_kw, every, every

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        env = wargame_env.WarGameEnv(**self.env_kw)
        obs, _ = env.reset(seed=0)
        done = False
        while not done:
            action, _ = self.model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, te, tr, info = env.step(action)
            done = te or tr
        print(f"\n=== {self.num_timesteps:,} шагов | исход {info['friendly_alive']}:{info['enemy_alive']} "
              f"| объектов {info.get('zones_held', 0)} ===")
        for ln in env.order_log_text().split("\n")[:4]:
            print("  " + ln)
        for ev in env.combat_log[-4:]:
            t = int(ev["step"] * wargame_env.SECONDS_PER_STEP)
            print(f"  {t//60:02d}:{t%60:02d} {ev['shooter']} ({ev['weapon']}) -> {ev['target']} -{ev['lost']}")
        print("===\n")
        env.close()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--timesteps", type=int, default=300_000)
    # 16, а не 8: шаг среды стоит ~150 мс (лучевая трассировка обнаружения), и она упирается
    # в число ядер, а не в питон. Замер на 16 ядрах: 8 сред — 27 шаг/с, 12 — 37, 16 — 42.
    # DummyVecEnv тут бесполезен (6 шаг/с при любом n_envs — среды идут последовательно).
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zones", type=int, default=3)
    parser.add_argument("--fog", action="store_true")
    # По умолчанию БОЛЬШИНСТВО боёв на настоящих картах: процедурная генерация не воспроизводит
    # форму реальной местности (12% леса против 21%), а от формы напрямую зависит видимость, то
    # есть вся дальняя стрельба и подавление. Немного процедурных оставляем для разнообразия —
    # годных вырезок всего 14, и на них одних легко переучиться.
    parser.add_argument("--fixed-map-prob", type=float, default=0.7,
                        help="доля боёв на настоящих вырезках (maps/platoon_pool.json)")
    args = parser.parse_args()

    fixed = []
    if args.fixed_map_prob > 0:
        fixed = wargame_env.map_pool()
        print(f"вырезок настоящих карт в пуле: {len(fixed)}")

    tag = f"_z{args.zones}" + ("_fog" if args.fog else "") + ("_maps" if fixed else "")
    log_dir = os.path.join(RUN_DIR, "logs")
    model_dir = os.path.join(RUN_DIR, "models", f"platoon{tag}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env_kw = dict(action_mode="command", n_zones=args.zones, fog=args.fog,
                  opponent="scripted",           # скриптовый ВЗВОДНЫЙ, не реактивный бот
                  fixed_map_files=fixed, fixed_map_prob=args.fixed_map_prob)

    def factory():
        return Monitor(wargame_env.WarGameEnv(**env_kw))

    train_env = make_vec_env(factory, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)
    # gamma ОБЯЗАН совпадать с gamma модели: VecNormalize нормирует награду на бегущую std
    # ДИСКОНТИРОВАННЫХ возвратов, и при разных gamma она считает масштаб не той величины,
    # которую оптимизирует агент. Было 0.99 против 0.995 у модели.
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=GAMMA, clip_reward=8.0)
    eval_env = make_vec_env(factory, n_envs=1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    print(f"CUDA: {torch.cuda.is_available()}   устройство: {args.device}")
    print(f"obs {train_env.observation_space.shape[0]}  action {train_env.action_space}")
    print(f"подразделения: {wargame_env.CALLSIGNS[0]}  против  {wargame_env.CALLSIGNS[1]}")
    print(f"бой до {wargame_env.MAX_STEPS * wargame_env.SECONDS_PER_STEP / 3600:.1f} ч игрового времени "
          f"= {wargame_env.MAX_STEPS // wargame_env.COMMAND_TEMPO} решений командира")

    model = MaskablePPO(
        policy="MlpPolicy", env=train_env, verbose=1, tensorboard_log=log_dir,
        # n_steps СЧИТАЕТСЯ ОТ n_envs, а не задаётся числом: важен размер роллаута (n_steps*n_envs),
        # он и определяет, сколько раз политика успеет обновиться. При n_steps=1024 и 16 средах
        # роллаут был бы 16384, то есть на 300k шагов всего 18 обновлений — для PPO мало.
        # ROLLOUT=4096 даёт ~73 обновления и ~68 боёв данных на каждое (эпизод = 60 решений).
        n_steps=max(64, ROLLOUT // args.n_envs),
        batch_size=256, n_epochs=5, gamma=GAMMA,
        gae_lambda=0.95, learning_rate=3e-4, clip_range=0.2,
        ent_coef=0.01,       # НЕ 0: словарь дискретный, без поощрения энтропии политика
                             # схлопывается на первую сработавшую задачу и перестаёт пробовать
        vf_coef=0.5, max_grad_norm=0.5, target_kl=0.08,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device=args.device, seed=args.seed,
    )

    # ИМЕННО Maskable-версия. Обычный EvalCallback зовёт predict() БЕЗ масок, то есть оценивает
    # политику, которой разрешено выбирать недопустимые приказы — а eval/mean_reward здесь
    # объявлен главной метрикой. Ошибка тихая: обучение идёт с масками, оценка без них, и
    # «лучшая модель» выбирается по чужому распределению действий.
    eval_cb = MaskableEvalCallback(eval_env, best_model_save_path=model_dir, log_path=log_dir,
                                   eval_freq=max(25_000 // args.n_envs, 1), n_eval_episodes=15,
                                   deterministic=True, render=False)

    print("\nСМОТРИ НА eval/mean_reward и rollout/force_edge, а НЕ на win_rate:\n"
          "победа засчитывается только за полное уничтожение противника — событие редкое,\n"
          "и на прошлых прогонах win_rate равнялся случайному даже когда модель училась.\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True,
                callback=[eval_cb, ckpt_cb, WinRateCallback(), ReportCallback(env_kw)])
    dt = time.time() - t0

    final = os.path.join(model_dir, "platoon_final")
    model.save(final)
    train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()
    print(f"\nЗаняло {dt/60:.1f} мин ({args.timesteps/max(dt,1):.0f} решений/с)")
    print(f"Модель: {final}.zip   лучшая: {os.path.join(model_dir, 'best_model.zip')}")
    print(f"\nСравнить с эталоном:  py -3.12 bench.py {os.path.join(model_dir,'best_model.zip')} --command")


if __name__ == "__main__":
    main()
