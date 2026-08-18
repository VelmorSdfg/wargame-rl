"""Сборка редактора карт в один .exe (PyInstaller).

    py -3.12 build_exe.py

Что получится: dist/WarGameMapEditor.exe. Рядом с ним должны лежать папки maps/ и scenarios/ —
это рабочие файлы, и внутрь бинарника они не прячутся: карты нужно и редактировать, и отдавать
в игру. Данные, которые редактор только читает (units.json, terrain.json и исходник
wargame_env.py, откуда берутся ARENA и ZONE_RADIUS), упакованы внутрь.

Лишние зависимости выброшены явно: обучение тянет torch и sb3 на сотни мегабайт, а редактору
нужны только numpy, Pillow и tkinter.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NAME = "WarGameMapEditor"
SEP = ";" if os.name == "nt" else ":"

EXCLUDE = ("torch", "stable_baselines3", "sb3_contrib", "gymnasium", "gym", "scipy", "pygame",
           "matplotlib", "pandas", "IPython", "pytest", "cv2")


def main():
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
            "--windowed", "--name", NAME,
            "--paths", os.path.join(ROOT, "editor"), "--paths", ROOT]
    for data in ("terrain.json", "units.json", "wargame_env.py"):
        args += ["--add-data", f"{os.path.join(ROOT, data)}{SEP}."]
    for mod in ("project", "measure", "vectormap", "mapgen", "terrain"):
        args += ["--hidden-import", mod]
    for mod in EXCLUDE:
        args += ["--exclude-module", mod]
    args += [os.path.join(ROOT, "editor", "map_editor.py")]

    print(" ".join(args[:6]), "…")
    r = subprocess.run(args, cwd=ROOT)
    if r.returncode:
        raise SystemExit(r.returncode)

    exe = os.path.join(ROOT, "dist", NAME + (".exe" if os.name == "nt" else ""))
    size = os.path.getsize(exe) / 1024 / 1024
    # рядом с exe кладём рабочие папки, иначе первый же запуск не найдёт куда сохранять
    for d in ("maps", "scenarios"):
        os.makedirs(os.path.join(ROOT, "dist", d), exist_ok=True)
    src_maps = os.path.join(ROOT, "maps")
    dst_maps = os.path.join(ROOT, "dist", "maps")
    for f in os.listdir(src_maps):
        if f.endswith(".vector.json"):
            shutil.copy2(os.path.join(src_maps, f), os.path.join(dst_maps, f))
    print(f"\nготово: {exe}  ({size:.0f} МБ)")
    print("рядом созданы dist/maps и dist/scenarios; векторные карты скопированы")


if __name__ == "__main__":
    main()
