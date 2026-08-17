"""Обучение РОТНОГО КОМАНДИРА: приказы элементам (взводам), а не отдельным юнитам.

Отличия от train.py:
  * action_mode="command" — задача x объект x направление на каждый элемент;
  * MaskablePPO — маскирование бессмысленных приказов. В Gym-uRTS маскирование назвали одной
    из ДВУХ вещей, без которых не заработало (вторая — композиция действий, это и есть наши
    приказы элементам вместо юнитов);
  * приказы липкие, темп COMMAND_TEMPO физических шагов вместо ACTION_REPEAT.

Пространство действий: 11^3 ~ 1300 против 5^10 ~ 9.8 млн при командовании юнитами.

    py -3.12 train_command.py --timesteps 1000000 --zones 3
    py -3.12 train_command.py --timesteps 1000000 --zones 3 --fixed-map-prob 0.5
"""

import argparse
import os
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import wargame_env

RUN_DIR = wargame_env.project_dir()


class WinRateCallback(BaseCallback):
    """Та же метрика, что в train.py — для прямого сравнения с прошлыми прогонами."""
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


class OrderLogCallback(BaseCallback):
    """Раз в N шагов прогоняет один эпизод и печатает ПОТОК ПРИКАЗОВ.

    Без этого про главную цель («выглядит как замысел») судить нечем: win_rate и награда
    показывают исход, но не показывают, осмысленные ли приказы отдаются."""
    def __init__(self, env_kw, every=100_000):
        super().__init__()
        self.env_kw = env_kw
        self.every = every
        self._next = every

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        env = wargame_env.WarGameEnv(**self.env_kw)
        obs, _ = env.reset(seed=0)
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, te, tr, _ = env.step(action)
            done = te or tr
        lines = env.order_log_text().split("\n")
        print(f"\n--- приказы на {self.num_timesteps:,} шагов (первые 6 решений) ---")
        print("\n".join(lines[:6]))
        print("---\n")
        env.close()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-map-prob", type=float, default=0.0)
    parser.add_argument("--zones", type=int, default=3)
    parser.add_argument("--fog", action="store_true")
    args = parser.parse_args()

    fixed_map_files = []
    if args.fixed_map_prob > 0:
        import glob
        fixed_map_files = sorted(p[:-4] for p in glob.glob(os.path.join(RUN_DIR, "maps", "tactical_crop_*.npy")))
        print(f"найдено вырезок карт: {len(fixed_map_files)}")

    tag = f"_z{args.zones}" + ("_fog" if args.fog else "")
    log_dir = os.path.join(RUN_DIR, "logs")
    model_dir = os.path.join(RUN_DIR, "models", f"command{tag}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"CUDA доступна: {torch.cuda.is_available()}   устройство: {args.device}")

    env_kw = dict(fixed_map_files=fixed_map_files, fixed_map_prob=args.fixed_map_prob,
                  action_mode="command", n_zones=args.zones, fog=args.fog)

    def factory():
        return Monitor(wargame_env.WarGameEnv(**env_kw))

    train_env = make_vec_env(factory, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=8.0)
    eval_env = make_vec_env(factory, n_envs=1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    print(f"observation_space: {train_env.observation_space}")
    print(f"action_space:      {train_env.action_space}  (элементы: {wargame_env.ELEMENT_NAMES})")
    print(f"темп командования: приказ держится {wargame_env.COMMAND_TEMPO} физ.шагов "
          f"=> ~{wargame_env.MAX_STEPS // wargame_env.COMMAND_TEMPO} решений за эпизод")

    model = MaskablePPO(
        policy="MlpPolicy", env=train_env, verbose=1, tensorboard_log=log_dir,
        n_steps=2048, batch_size=256, n_epochs=5, gamma=0.99, gae_lambda=0.95,
        learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        target_kl=0.08,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device=args.device, seed=args.seed,
    )
    print(f"Фактически используется устройство: {model.device}")

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=model_dir, log_path=log_dir,
        eval_freq=max(50_000 // args.n_envs, 1), n_eval_episodes=20,
        deterministic=True, render=False,
    )

    print(f"\nСтарт (ротный командир): {args.timesteps:,} решений.\n"
          f"Периодически будет печататься ПОТОК ПРИКАЗОВ — читай его глазами, "
          f"win_rate не покажет, осмысленный ли замысел.\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=True,
                callback=[eval_callback, WinRateCallback(), OrderLogCallback(env_kw)])
    dt = time.time() - t0

    final_path = os.path.join(model_dir, "command_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()

    print(f"\nОбучение заняло {dt / 60:.1f} мин ({args.timesteps / max(dt, 1):.0f} решений/с)")
    print(f"Финальная модель: {final_path}.zip")
    print(f"Лучшая:           {os.path.join(model_dir, 'best_model.zip')}")


if __name__ == "__main__":
    main()
