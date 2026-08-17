"""Демо-бой на РЕАЛЬНОЙ вырезке 7x7км (maps/tactical_crop) — показать дороги/капасити вживую."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import imageio

import wargame_env
from play import draw_unit, draw_fire_lines, draw_legend, draw_terrain, SIZE
from render_demo import friendly_script


def main():
    env = wargame_env.WarGameEnv(fixed_map_files=["maps/tactical_crop"], fixed_map_prob=1.0)
    pygame.init()
    screen = pygame.display.set_mode((SIZE, SIZE))
    font = pygame.font.SysFont("consolas", 18)

    frames = []
    obs, _ = env.reset(seed=5)
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
        hud = f"свои: {info['friendly_alive']}  враги: {info['enemy_alive']}  шаг {env.steps}  [реальная вырезка 7x7км]"
        screen.blit(font.render(hud, True, (230, 230, 230)), (10, 8))
        draw_legend(screen, font)
        pygame.display.flip()

        frame = pygame.surfarray.array3d(screen)
        frames.append(np.transpose(frame, (1, 0, 2)))

    out = os.path.join(wargame_env.project_dir(), "battle_tactical.gif")
    try:
        imageio.mimsave(out, frames, duration=1000.0 / 10, loop=0)
    except TypeError:
        imageio.mimsave(out, frames, fps=10)
    pygame.quit()
    print(f"кадров: {len(frames)}  ->  {out}")


if __name__ == "__main__":
    main()
