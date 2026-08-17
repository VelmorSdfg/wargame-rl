"""Self-play: агент играет против СВОИХ ЖЕ прошлых версий, а не против скриптового бота.

Зачем: фиксированный скриптовый бот — это шаблон с фиксированной дыркой. Агент выучивает не
тактику, а «как обыграть вот этого бота», и упирается в потолок. Self-play работает как подвижная
программа обучения: соперник растёт вместе с агентом.

ВАЖНО про измерение прогресса: против самого себя доля побед всегда ~50%, по ней роста НЕ видно.
Поэтому eval здесь идёт против СКРИПТОВОГО бота — это неподвижный эталон. Смотреть надо на
eval/mean_reward, а не на rollout/win_rate.

    py -3.12 train_selfplay.py --timesteps 1000000 --zones 3
    py -3.12 train_selfplay.py --timesteps 1000000 --orders --zones 3
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import selfplay
import wargame_env

RUN_DIR = wargame_env.project_dir()


class SnapshotCallback(BaseCallback):
    """Периодически кладёт текущую политику в пул соперников. Воркеры подхватывают файлы сами
    (см. selfplay.PoolOpponent) — межпроцессного обмена не требуется."""
    def __init__(self, snap_dir, every_steps, keep=10):
        super().__init__()
        self.snap_dir = snap_dir
        self.every = every_steps
        self.keep = keep
        self._next = every_steps

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.every
            path = os.path.join(self.snap_dir, f"snap_{self.num_timesteps:09d}")
            self.model.save(path)
            snaps = sorted(f for f in os.listdir(self.snap_dir) if f.startswith("snap_"))
            for old in snaps[:-self.keep]:      # держим только последние keep штук
                try:
                    os.remove(os.path.join(self.snap_dir, old))
                except OSError:
                    pass
            if self.verbose:
                print(f"  [self-play] снапшот в пул: {os.path.basename(path)}.zip (всего {min(len(snaps), self.keep)})")
        return True


class WinRateCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.wins = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "win" in info:
                self.wins.append(1.0 if info["win"] else 0.0)
                if len(self.wins) > 200:
                    self.wins.pop(0)
        if self.wins:
            self.logger.record("rollout/win_rate", float(np.mean(self.wins)))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-map-prob", type=float, default=0.0)
    parser.add_argument("--orders", action="store_true")
    parser.add_argument("--zones", type=int, default=0)
    parser.add_argument("--fog", action="store_true")
    parser.add_argument("--snapshot-every", type=int, default=50_000,
                        help="как часто класть версию себя в пул соперников")
    args = parser.parse_args()

    fixed_map_files = []
    if args.fixed_map_prob > 0:
        import glob
        fixed_map_files = sorted(p[:-4] for p in glob.glob(os.path.join(RUN_DIR, "maps", "tactical_crop_*.npy")))
        print(f"найдено вырезок карт: {len(fixed_map_files)}")

    tag = "".join(["_orders" if args.orders else "", f"_z{args.zones}" if args.zones else "",
                   "_fog" if args.fog else ""])
    log_dir = os.path.join(RUN_DIR, "logs")
    model_dir = os.path.join(RUN_DIR, "models", f"selfplay{tag}")
    snap_dir = os.path.join(model_dir, "pool")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)
    for stale in os.listdir(snap_dir):          # пул от прошлого прогона только мешает
        if stale.startswith("snap_"):
            os.remove(os.path.join(snap_dir, stale))

    print(f"CUDA доступна: {torch.cuda.is_available()}   устройство: {args.device}")

    env_kw = dict(fixed_map_files=fixed_map_files, fixed_map_prob=args.fixed_map_prob,
                  action_mode="orders" if args.orders else "velocity",
                  n_zones=args.zones, fog=args.fog)

    def factory_selfplay():
        # PoolOpponent создаётся ВНУТРИ воркера — модели соперников не пересылаются между процессами
        env = wargame_env.WarGameEnv(**env_kw)
        env.opponent = selfplay.PoolOpponent(snap_dir)
        return Monitor(env)

    def factory_scripted():
        return Monitor(wargame_env.WarGameEnv(**env_kw))    # эталон: скриптовый бот

    train_env = make_vec_env(factory_selfplay, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=8.0)
    eval_env = make_vec_env(factory_scripted, n_envs=1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    print(f"observation_space: {train_env.observation_space}, action_space: {train_env.action_space}")

    model = PPO(
        policy="MlpPolicy", env=train_env, verbose=1, tensorboard_log=log_dir,
        n_steps=2048, batch_size=256, n_epochs=5, gamma=0.99, gae_lambda=0.95,
        learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        target_kl=0.08,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
        device=args.device, seed=args.seed,
    )
    print(f"Фактически используется устройство: {model.device}")

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=model_dir, log_path=log_dir,
        eval_freq=max(50_000 // args.n_envs, 1), n_eval_episodes=20,
        deterministic=True, render=False,
    )

    print(f"\nСтарт (self-play): {args.timesteps:,} шагов агента. Пока пул пуст, красными правит "
          f"скриптовый бот — это нормальный прогрев.\n"
          f"СМОТРИ НА eval/mean_reward (против скриптового эталона), а НЕ на rollout/win_rate:\n"
          f"против самого себя доля побед всегда около 50% и роста по ней не видно.\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True,
                callback=[eval_callback, WinRateCallback(),
                          SnapshotCallback(snap_dir, args.snapshot_every)])
    dt = time.time() - t0

    final_path = os.path.join(model_dir, "selfplay_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()

    print(f"\nОбучение заняло {dt / 60:.1f} мин ({args.timesteps / max(dt, 1):.0f} шагов/с)")
    print(f"Финальная модель: {final_path}.zip")
    print(f"Лучшая (против скриптового эталона): {os.path.join(model_dir, 'best_model.zip')}")


if __name__ == "__main__":
    main()
