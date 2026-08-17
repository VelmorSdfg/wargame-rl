"""
Обучение агента-командира мини-варгейма (PPO, MlpPolicy).

Наблюдение — вектор состояния всех юнитов, поэтому MlpPolicy (не картинка).
Враг — скриптовый бот (Фаза 1). Позже можно заменить на self-play.

Запуск:
    python train.py --timesteps 3000000
    python train.py --device cpu     # для маленькой сети CPU обычно быстрее
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

import wargame_env

RUN_DIR = wargame_env.project_dir()


class WinRateCallback(BaseCallback):
    """Пишет в лог долю побед — самая понятная метрика для боя."""
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
    parser.add_argument("--fixed-map-prob", type=float, default=0.0,
                        help="вероятность взять реальную вырезанную карту (maps/tactical_crop_*) вместо процедурной")
    parser.add_argument("--fixed-maps", type=str, default="auto",
                        help="через запятую пути без расширения, либо 'auto' — все maps/tactical_crop_*.npy")
    # --- опциональные механики среды; все ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНЫ, чтобы каждую можно было
    # --- включать по одной и понимать, какая именно дала эффект (см. wargame_env.WarGameEnv)
    parser.add_argument("--orders", action="store_true",
                        help="действия уровня приказов (MultiDiscrete) вместо сырых (vx,vy)")
    parser.add_argument("--zones", type=int, default=0,
                        help="сколько точек-объектов ставить (0 = выключено). Даёт игре цель, кроме размена")
    parser.add_argument("--fog", action="store_true",
                        help="туман войны (осмысленно только с train_recurrent.py — нужна память)")
    args = parser.parse_args()
    if args.fixed_map_prob <= 0:
        fixed_map_files = []
    elif args.fixed_maps == "auto":
        import glob
        fixed_map_files = sorted(p[:-4] for p in glob.glob(os.path.join(RUN_DIR, "maps", "tactical_crop_*.npy")))
        print(f"найдено вырезок карт: {len(fixed_map_files)}")
    else:
        fixed_map_files = [p.strip() for p in args.fixed_maps.split(",") if p.strip()]

    # Конфиги пишутся в РАЗНЫЕ папки: иначе прогон с зонами затрёт best_model.zip прогона без них
    # и сравнивать будет нечего (ровно так SAC чуть не затёр модель PPO).
    tag = "".join(["_orders" if args.orders else "", f"_z{args.zones}" if args.zones else "",
                   "_fog" if args.fog else ""])
    log_dir = os.path.join(RUN_DIR, "logs")
    model_dir = os.path.join(RUN_DIR, "models", f"ppo{tag}" if tag else "")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"CUDA доступна: {torch.cuda.is_available()}   устройство: {args.device}")

    def factory():
        return Monitor(wargame_env.WarGameEnv(
            fixed_map_files=fixed_map_files, fixed_map_prob=args.fixed_map_prob,
            action_mode="orders" if args.orders else "velocity",
            n_zones=args.zones, fog=args.fog))

    train_env = make_vec_env(factory, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)
    # obs уже нормализованы в среде -> норм только НАГРАДУ. Раньше терминал ±300 гонял скользящую
    # std нормализатора (после каждой крупной победы/поражения она резко пере-калибровалась и
    # гасила плотный dmg/kill-shaping на соседних шагах). Теперь терминал сопоставим по масштабу
    # с kill-бонусом (см. TERMINAL_REWARD в wargame_env.py) + clip_reward пониже default — меньше
    # риска, что редкий всплеск раскачает статистику.
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=8.0)
    eval_env = make_vec_env(factory, n_envs=1, seed=args.seed + 1000)
    # eval — та же обёртка, но БЕЗ нормализации награды (реальные награды для метрик) и без обновления статистики
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    print(f"observation_space: {train_env.observation_space}, action_space: {train_env.action_space}")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,       # было 10: 16384 сэмплов / batch 256 * 10 эпох = 640 шагов по одним данным.
                          # К 5-й эпохе политика уезжала далеко от распределения, которым сэмплили —
                          # clip_fraction сидел на 0.53 (здоровые 0.05-0.2), approx_kl 0.09-0.13 при
                          # потолке 0.08, early stopping почти каждую итерацию (доходило ~6.3 эпохи из 10).
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.08,   # было 0.03 — душило политику (approx_kl весь прогон упирался в потолок,
                          # std/entropy не двигались 2.55M шагов); ослаблено, чтобы обновления доходили
        # log_std_init=-0.5 => стартовый шум исследования ~0.61 вместо ~1.01 при дефолтном 0.0.
        # Дефолт давал разброс размером СО ВЕСЬ диапазон действий Box(-1,1): за 3M шагов std сдвинулась
        # только 1.0 -> 0.954, т.е. агент всё обучение исполнял почти чистый шум, и среднее не получало
        # внятного сигнала. gSDE тут НЕ используем: временную связность шума уже даёт ACTION_REPEAT,
        # а у gSDE своя семантика log_std (замер: при -0.5 шум 2.86, эквивалент нашему -0.5 это ~-2.0).
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
        device=args.device,
        seed=args.seed,
    )
    print(f"Фактически используется устройство: {model.device}")

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=model_dir, log_path=log_dir,
        eval_freq=max(50_000 // args.n_envs, 1), n_eval_episodes=20,
        deterministic=True, render=False,
    )

    print(f"\nСтарт: {args.timesteps:,} шагов. Следи за rollout/win_rate (доля побед) "
          f"и ep_rew_mean.\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps,
                callback=[eval_callback, WinRateCallback()], progress_bar=True)
    dt = time.time() - t0

    final_path = os.path.join(model_dir, "ppo_wargame_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))  # статистика награды (для дообучения)
    train_env.close()
    eval_env.close()

    print(f"\nОбучение заняло {dt / 60:.1f} мин ({args.timesteps / max(dt, 1):.0f} шагов/с)")
    print(f"Финальная модель: {final_path}.zip")
    print(f"Лучшая:           {os.path.join(model_dir, 'best_model.zip')}")
    print(f"\nПосмотреть бой:  python play.py models/best_model.zip")


if __name__ == "__main__":
    main()
