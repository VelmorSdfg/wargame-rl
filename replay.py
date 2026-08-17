"""Запись и просмотр боёв.

Зачем: по числам не понять, ЧТО именно пошло не так. Прогон обучения выдаёт win_rate и награду,
а вопрос «почему пулемётное всю дорогу простояло в тылу» решается только просмотром. Плюс запись
переживает изменения кода: в файле лежат кадры и метаданные, а не объект среды.

    py -3.12 replay.py record --out replays/scripted.npz --seed 3 --fixed-maps
    py -3.12 replay.py record --model models/platoon_z3/best_model.zip --out replays/agent.npz
    py -3.12 replay.py play   replays/agent.npz

При просмотре доступно то же, что в игре: перемотка, осмотр подразделения, линия видимости
по L, кольца дальностей, приказы и донесения.
"""
import argparse
import glob
import os

import numpy as np

import wargame_env


def record(out, seed=0, zones=3, fog=False, fixed=False, model_path=None):
    """Прогнать бой и сохранить покадрово. Приказы синих — от модели, если она задана,
    иначе от скриптового взводного (тогда это эталонная запись для сравнения)."""
    files = []
    if fixed:
        files = wargame_env.map_pool()
    env = wargame_env.WarGameEnv(action_mode="command", n_zones=zones, fog=fog,
                                 opponent="scripted",
                                 fixed_map_files=files, fixed_map_prob=1.0 if files else 0.0)
    model = None
    if model_path:
        from play import load_any
        model = load_any(model_path)

    obs, _ = env.reset(seed=seed)
    frames = [env.snapshot()]
    done = False
    info = {"friendly_alive": env.n_side, "enemy_alive": env.n_side}
    # Кадр на КАЖДЫЙ ФИЗИЧЕСКИЙ ШАГ, а не на решение командира: решение принимается раз в
    # COMMAND_TEMPO (2 минуты боя), и запись по решениям давала бы рваную анимацию.
    env._opp_action = "scripted"
    while not done:
        if model is not None:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        else:
            action = env._scripted_commander(0)
        env._set_orders(0, np.asarray(action, dtype=np.int64).reshape(env.n_elements, 3))
        env._set_orders(1, env._scripted_commander(1))
        for _ in range(wargame_env.COMMAND_TEMPO):
            obs, _, te, tr, info = env._physics_step(None)
            frames.append(env.snapshot())
            if te or tr:
                done = True
                break

    save_frames(out, env, frames, model_path or "скриптовый взводный", seed=seed)
    mins = frames[-1]["step"] * wargame_env.SECONDS_PER_STEP / 60.0
    print(f"записано кадров: {len(frames)}  ({mins:.0f} мин боя)")
    print(f"исход: наши {info['friendly_alive']} / противник {info['enemy_alive']}, "
          f"объектов {info.get('zones_held', 0)}")
    print(f"файл: {out}")
    env.close()


def auto_name(seed, tag="auto"):
    """Имя для автоматической записи из игры: по времени, чтобы записи не затирали друг друга."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(wargame_env.project_dir(), "replays", f"{tag}_{ts}_seed{seed}.npz")


def save_frames(out, env, frames, source, seed=0):
    """Записать кадры в .npz. Вынесено сюда, потому что пишет и replay.py record, и сама игра:
    две копии этого кода разошлись бы при первом же добавлении поля в snapshot().
    Параметры боя берутся ИЗ СРЕДЫ, а не передаются отдельно — иначе рано или поздно разъедутся."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        # СЕТКА МЕСТНОСТИ И ЗОНЫ — чтобы реплей рисовался без пересоздания той же карты:
        # процедурная генерация зависит от кода, а он меняется, и запись бы «поехала».
        grid=env.map.grid, cell=np.float32(env.map.cell), zones=env.zones,
        seed=np.int32(seed), n_zones=np.int32(env.n_zones), fog=np.bool_(env.fog),
        pos=np.stack([f["pos"] for f in frames]),
        hp=np.stack([f["hp"] for f in frames]),
        alive=np.stack([f["alive"] for f in frames]),
        suppr=np.stack([f["suppr"] for f in frames]),
        settle=np.stack([f["settle"] for f in frames]),
        moved=np.stack([f["moved"] for f in frames]),
        orders=np.stack([f["orders"] for f in frames]),
        free=np.stack([f["free"] for f in frames]),
        fire=np.stack([f["fire"] for f in frames]),
        acq=np.stack([f["acq"] for f in frames]),
        mortar=np.stack([f["mortar"] for f in frames]),
        ammo=np.stack([np.concatenate(f["ammo"]) for f in frames]),
        combat_step=np.array([e["step"] for e in env.combat_log], dtype=np.int32),
        # ВАЖНО: строки сохраняем как обычный юникодный массив, а НЕ dtype=object. Объектные
        # массивы внутри .npz — это pickle, и такой файл читается только с allow_pickle=True,
        # то есть загрузка чужого реплея означала бы исполнение кода из него. Здесь это ни к чему.
        combat_txt=np.array([f"{e['shooter']}|{e['weapon']}|{e['target']}|{e['lost']}|"
                             f"{int(e['destroyed'])}|{e['friendly_fire_side']}"
                             for e in env.combat_log] or [""], dtype="U64"),
        source=np.array(source, dtype="U128"),
    )


def load(path):
    # allow_pickle НЕ нужен: в файле только массивы чисел и строк фиксированной ширины.
    # Так чужую запись можно открыть, не рискуя исполнить из неё код.
    return np.load(path, allow_pickle=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="записать бой")
    r.add_argument("--out", default="replays/battle.npz")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--zones", type=int, default=3)
    r.add_argument("--fog", action="store_true")
    r.add_argument("--fixed-maps", action="store_true")
    r.add_argument("--model", default=None, help="если задана — приказы отдаёт она, иначе скрипт")

    p = sub.add_parser("play", help="смотреть запись")
    p.add_argument("path")
    p.add_argument("--fps", type=int, default=30)

    args = ap.parse_args()
    if args.cmd == "record":
        record(args.out, seed=args.seed, zones=args.zones, fog=args.fog,
               fixed=args.fixed_maps, model_path=args.model)
    else:
        import game
        game.run_replay(args.path, fps=args.fps)


if __name__ == "__main__":
    main()
