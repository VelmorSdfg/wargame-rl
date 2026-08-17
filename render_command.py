"""Гифка боя скриптовых взводных.

Своего отрисовщика тут БОЛЬШЕ НЕТ. Раньше был, и он отстал: рисовал без нижней панели, без
приказов на карте, с легендой, где значились ПТУР, БМП и ОБТ — техника ротного масштаба,
которой у взвода нет. Две копии одного отрисовщика расходятся молча; сегодня из-за этого
вызов миномётов пришлось добавлять дважды. Теперь путь один: записать бой в .npz и прогнать
его тем же кодом, что рисует живую игру и просмотр записей.

    py -3.12 render_command.py --fixed-maps --seed 3
    py -3.12 render_command.py --model models/platoon_z3/best_model.zip --out agent.gif
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import replay
import wargame_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--zones", type=int, default=3)
    ap.add_argument("--fixed-maps", action="store_true")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--every", type=int, default=4, help="брать каждый N-й кадр боя")
    ap.add_argument("--model", default=None, help="если задана — синими правит она, иначе скрипт")
    ap.add_argument("--out", default="battle_command.gif")
    ap.add_argument("--keep-npz", action="store_true", help="не удалять промежуточную запись")
    args = ap.parse_args()

    npz = os.path.join(wargame_env.project_dir(), "replays",
                       f"_render_seed{args.seed}.npz")
    replay.record(npz, seed=args.seed, zones=args.zones, fixed=args.fixed_maps,
                  model_path=args.model)

    import game
    game.render_replay_gif(npz, os.path.join(wargame_env.project_dir(), args.out),
                           fps=args.fps, every=args.every)
    if not args.keep_npz:
        os.remove(npz)


if __name__ == "__main__":
    main()
