"""Обучение с ПАМЯТЬЮ: RecurrentPPO (LSTM) вместо обычного PPO.

Зачем: MlpPolicy не имеет состояния — она заново выводит всё из текущего кадра и физически
не может держать план («сковать ПТУРами, пока БМП обходят»). Плюс туман войны без памяти почти
бессмыслен: невидимый враг замирает в obs на последней известной позиции, и помнить, что он там
был, нечем. LSTM закрывает и то, и другое.

    py -3.12 train_recurrent.py --timesteps 1000000
    py -3.12 train_recurrent.py --timesteps 1000000 --zones 3 --fog --orders
"""

import argparse
import os
import time

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import wargame_env

RUN_DIR = wargame_env.project_dir()


class WinRateCallback(BaseCallback):
    """Та же метрика, что в train.py — чтобы прогоны были сравнимы напрямую."""
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
    parser.add_argument("--fixed-maps", type=str, default="auto")
    parser.add_argument("--orders", action="store_true")
    parser.add_argument("--zones", type=int, default=0)
    parser.add_argument("--fog", action="store_true")
    args = parser.parse_args()

    if args.fixed_map_prob <= 0:
        fixed_map_files = []
    elif args.fixed_maps == "auto":
        import glob
        fixed_map_files = sorted(p[:-4] for p in glob.glob(os.path.join(RUN_DIR, "maps", "tactical_crop_*.npy")))
        print(f"найдено вырезок карт: {len(fixed_map_files)}")
    else:
        fixed_map_files = [p.strip() for p in args.fixed_maps.split(",") if p.strip()]

    tag = "".join(["_orders" if args.orders else "", f"_z{args.zones}" if args.zones else "",
                   "_fog" if args.fog else ""])
    log_dir = os.path.join(RUN_DIR, "logs")
    model_dir = os.path.join(RUN_DIR, "models", f"lstm{tag}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"CUDA доступна: {torch.cuda.is_available()}   устройство: {args.device}")

    def factory():
        return Monitor(wargame_env.WarGameEnv(
            fixed_map_files=fixed_map_files, fixed_map_prob=args.fixed_map_prob,
            action_mode="orders" if args.orders else "velocity",
            n_zones=args.zones, fog=args.fog))

    train_env = make_vec_env(factory, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=8.0)
    eval_env = make_vec_env(factory, n_envs=1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    print(f"observation_space: {train_env.observation_space}, action_space: {train_env.action_space}")

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        # n_steps меньше, чем у PPO (2048): BPTT по всей длине роллаута дорог по памяти и времени,
        # а 256*8 = 2048 сэмплов на обновление — всё ещё достаточный батч.
        n_steps=256,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.08,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            lstm_hidden_size=256,
            n_lstm_layers=1,
            log_std_init=-0.5,   # для MultiDiscrete игнорируется, для Box — стартовый шум ~0.61
        ),
        device=args.device,
        seed=args.seed,
    )
    print(f"Фактически используется устройство: {model.device}")

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=model_dir, log_path=log_dir,
        eval_freq=max(50_000 // args.n_envs, 1), n_eval_episodes=20,
        deterministic=True, render=False,
    )

    print(f"\nСтарт (RecurrentPPO): {args.timesteps:,} шагов агента. "
          f"LSTM заметно медленнее MLP — это ожидаемо.\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps,
                callback=[eval_callback, WinRateCallback()], progress_bar=True)
    dt = time.time() - t0

    final_path = os.path.join(model_dir, "lstm_wargame_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()

    print(f"\nОбучение заняло {dt / 60:.1f} мин ({args.timesteps / max(dt, 1):.0f} шагов/с)")
    print(f"Финальная модель: {final_path}.zip")
    print(f"Лучшая:           {os.path.join(model_dir, 'best_model.zip')}")


if __name__ == "__main__":
    main()
