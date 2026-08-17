"""
Обучение агента-командира мини-варгейма (SAC, MlpPolicy) — альтернатива PPO (train.py).

Зачем пробовать: диагностика прошлого PPO-прогона показала, что std/entropy политики
практически не двигались 2.55M шагов (approx_kl весь прогон упирался в target_kl) — trust-region
PPO душил обновления. SAC регулирует exploration автоматически через auto-tuned entropy
(ent_coef="auto") — нет клиппинга/kl-стоп-крана, которые могли забить прогресс.

Запуск (аналогично train.py):
    python train_sac.py --timesteps 3000000
    python train_sac.py --fixed-map-prob 0.3
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import wargame_env

RUN_DIR = wargame_env.project_dir()


class WinRateCallback(BaseCallback):
    """Пишет в лог долю побед — та же метрика, что и в train.py, для честного сравнения."""
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
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=500_000,
                        help="реплей-буфер (транзакций); 500к ~ разумный компромисс памяти/истории")
    parser.add_argument("--fixed-map-prob", type=float, default=0.0,
                        help="вероятность взять реальную вырезанную карту (maps/tactical_crop_*) вместо процедурной")
    parser.add_argument("--fixed-maps", type=str, default="auto",
                        help="через запятую пути без расширения, либо 'auto' — все maps/tactical_crop_*.npy")
    args = parser.parse_args()
    if args.fixed_map_prob <= 0:
        fixed_map_files = []
    elif args.fixed_maps == "auto":
        import glob
        fixed_map_files = sorted(p[:-4] for p in glob.glob(os.path.join(RUN_DIR, "maps", "tactical_crop_*.npy")))
        print(f"найдено вырезок карт: {len(fixed_map_files)}")
    else:
        fixed_map_files = [p.strip() for p in args.fixed_maps.split(",") if p.strip()]

    log_dir = os.path.join(RUN_DIR, "logs")
    model_dir = os.path.join(RUN_DIR, "models", "sac")  # своя подпапка — не затирать best_model.zip у PPO
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"CUDA доступна: {torch.cuda.is_available()}   устройство: {args.device}")

    def factory():
        return Monitor(wargame_env.WarGameEnv(fixed_map_files=fixed_map_files, fixed_map_prob=args.fixed_map_prob))

    train_env = make_vec_env(factory, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)
    # та же нормализация награды, что у PPO-версии — для сравнимости; obs уже нормализованы средой
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=8.0)
    eval_env = make_vec_env(factory, n_envs=1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    print(f"observation_space: {train_env.observation_space}, action_space: {train_env.action_space}")

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        buffer_size=args.buffer_size,
        learning_starts=10_000,     # прогреть буфер случайными действиями перед первым обновлением
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        # ВАЖНО: у off-policy алгоритмов в SB3 train_freq/gradient_steps считаются в ВЫЗОВАХ
        # env.step(), а не в транзакциях. При n_envs=8 один вызов даёт 8 транзакций, а
        # gradient_steps=1 — всего один шаг оптимизатора: replay ratio 0.125 вместо 1.0.
        # Прошлый прогон на 1.77M шагов среды сделал лишь ~220k обновлений (недоучен в 8 раз).
        # -1 = столько шагов, сколько собрано транзакций. См. доки SB3 (guide/examples, «multiple envs»).
        gradient_steps=-1,
        learning_rate=3e-4,
        ent_coef="auto",             # ключевое отличие от PPO: энтропия НЕ фиксирована, подстраивается сама
        target_entropy="auto",       # эвристика SB3: -dim(action) = -20 для нашего Box(20)
        policy_kwargs=dict(net_arch=[256, 256]),
        device=args.device,
        seed=args.seed,
    )
    print(f"Фактически используется устройство: {model.device}")

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=model_dir, log_path=log_dir,
        eval_freq=max(50_000 // args.n_envs, 1), n_eval_episodes=20,
        deterministic=True, render=False,
    )

    print(f"\nСтарт (SAC): {args.timesteps:,} шагов. Следи за rollout/win_rate, ep_rew_mean, "
          f"train/ent_coef (должен сам подстраиваться, не стоять колом).\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps,
                callback=[eval_callback, WinRateCallback()], progress_bar=True)
    dt = time.time() - t0

    final_path = os.path.join(model_dir, "sac_wargame_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vecnormalize_sac.pkl"))
    train_env.close()
    eval_env.close()

    print(f"\nОбучение заняло {dt / 60:.1f} мин ({args.timesteps / max(dt, 1):.0f} шагов/с)")
    print(f"Финальная модель: {final_path}.zip")
    print(f"Лучшая:           {os.path.join(model_dir, 'best_model.zip')}")
    print(f"\nПосмотреть бой:  python play.py models/sac/best_model.zip")


if __name__ == "__main__":
    main()
