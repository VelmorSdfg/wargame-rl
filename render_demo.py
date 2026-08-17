"""
Рендерит демонстрационный бой в GIF (без обученной модели) — чтобы увидеть,
как выглядит поле и работают механики. Обеими сторонами управляет простой скрипт
«идти к ближайшему врагу»; огонь/броня/подавление считает сама среда.

Запуск:
    python render_demo.py            # сохранит battle.gif
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import imageio

import wargame_env
from wargame_env import ARENA
from play import draw_unit, draw_fire_lines, draw_legend, draw_terrain, SIZE, SCALE


def friendly_script(env):
    """Приказ своим для демо без модели: то же ролевое движение, что и у бота."""
    act = np.zeros((env.n_side, 2), dtype=np.float32)
    for i in range(env.n_side):
        if env.alive[i]:
            act[i] = env._role_move(i)
    return act.reshape(-1)


def main():
    env = wargame_env.WarGameEnv()
    pygame.init()
    screen = pygame.display.set_mode((SIZE, SIZE))
    font = pygame.font.SysFont("consolas", 18)

    frames = []
    obs, _ = env.reset(seed=0)
    done = False
    while not done and env.steps < 220:
        action = friendly_script(env)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

        screen.fill((18, 20, 24))
        draw_terrain(screen, env)
        draw_fire_lines(screen, env)
        for i in range(env.n):
            if env.alive[i]:
                draw_unit(screen, env, i, font)
        hud = f"свои: {info['friendly_alive']}  враги: {info['enemy_alive']}  шаг {env.steps}"
        screen.blit(font.render(hud, True, (230, 230, 230)), (10, 8))
        draw_legend(screen, font)
        pygame.display.flip()

        frame = pygame.surfarray.array3d(screen)     # (W,H,3)
        frames.append(np.transpose(frame, (1, 0, 2)))  # -> (H,W,3)

    out = os.path.join(wargame_env.project_dir(), "battle.gif")
    try:
        imageio.mimsave(out, frames, duration=1000.0 / 10, loop=0)
    except TypeError:
        imageio.mimsave(out, frames, fps=10)
    pygame.quit()
    print(f"кадров: {len(frames)}  ->  {out}")


if __name__ == "__main__":
    main()
