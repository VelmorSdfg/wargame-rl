"""Всё, что редактор знает о проекте, — в одном месте и ЧИТАЕТСЯ ИЗ ПРОЕКТА, а не зашито.

Масштаб уже менялся (поле 100 -> 170 единиц, ротный конфиг -> взводный), и карта, нарисованная
под старый размер, покрывает лишь угол арены. Поэтому размер сетки, состав сторон и радиус
рубежа берутся из units.json / terrain.json / wargame_env.py, а не из констант редактора.

Импорт wargame_env сюда намеренно НЕ делается: он тянет gymnasium и sb3, а редактору нужны
три числа. Константы вычитываются регуляркой, и если их не нашли — печатается предупреждение,
а не молча берётся значение по умолчанию.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Собранный .exe разводит две вещи, которые в исходниках лежат вместе:
#   BUNDLE — данные внутри самого exe (units.json, terrain.json, исходник wargame_env.py,
#            из которого читаются ARENA и ZONE_RADIUS). Только чтение, правке не подлежит.
#   ROOT   — папка РЯДОМ с exe, где живут карты и сценарии. Это рабочие файлы пользователя,
#            и прятать их внутрь бинарника нельзя — их же надо редактировать и класть в игру.
FROZEN = getattr(sys, "frozen", False)
BUNDLE = getattr(sys, "_MEIPASS", os.path.dirname(HERE))
ROOT = os.path.dirname(sys.executable) if FROZEN else os.path.dirname(HERE)
MAPS = os.path.join(ROOT, "maps")
SCENARIOS = os.path.join(ROOT, "scenarios")

if not FROZEN:
    sys.path.insert(0, ROOT)
import terrain  # noqa: E402

# Палитра — та же, что у play.py (TERRAIN_COLORS): редактор показывает карту такой, какой её
# увидят в бою, иначе «на превью читалось, в игре не читается».
TILE_COLORS = {0: (92, 96, 62), 1: (26, 58, 30), 2: (78, 76, 84), 3: (28, 52, 88), 4: (140, 124, 74)}
TILE_NAMES = {0: "открытое", 1: "лес", 2: "здания", 3: "вода", 4: "дороги"}

MARKER_KINDS = ("zones", "friendly", "enemy")
MARKER_RU = {"zones": "объект захвата", "friendly": "позиция своих", "enemy": "позиция врага"}
MARKER_COLORS = {"zones": (235, 205, 90), "friendly": (110, 160, 255), "enemy": (240, 100, 95)}


def _env_const(name, default):
    """Константа верхнего уровня из wargame_env.py. Возвращает (значение, найдено ли)."""
    try:
        with open(os.path.join(BUNDLE, "wargame_env.py"), "r", encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return default, False
    m = re.search(rf"^{name}\s*=\s*([\d.]+)", src, re.M)
    return (float(m.group(1)), True) if m else (default, False)


with open(os.path.join(BUNDLE, "units.json"), "r", encoding="utf-8") as _f:
    _UNITS = json.load(_f)

M_PER_UNIT = float(_UNITS["m_per_unit"])
CELL_UNITS = float(terrain._CFG["cell_size"])
CELL_M = CELL_UNITS * M_PER_UNIT

ARENA, _ok_arena = _env_const("ARENA", 170.0)
ZONE_RADIUS, _ok_zone = _env_const("ZONE_RADIUS", 17.0)
GRID_N = int(round(ARENA / CELL_UNITS))
ARENA_M = ARENA * M_PER_UNIT

# Состав стороны: сценарий задаёт ПОЗИЦИИ, а не наряд сил, и число позиций должно совпадать
# со слотами obs (scenario.validate это проверяет). Порядок элементов = порядок в units.json.
_OOB = _UNITS.get("order_of_battle") or [{"name": "все", "units": _UNITS.get("force", [])}]
ELEMENT_NAMES = [el["name"] for el in _OOB]
SLOT_NAMES = [el["name"] for el in _OOB for _ in el["units"]]
N_SIDE = len(SLOT_NAMES)

if not (_ok_arena and _ok_zone):
    print("ВНИМАНИЕ: не удалось вычитать ARENA/ZONE_RADIUS из wargame_env.py — "
          f"взяты значения по умолчанию (ARENA={ARENA:.0f}, ZONE_RADIUS={ZONE_RADIUS:.0f}). "
          "Проверьте, что редактор лежит рядом с проектом.")


def describe():
    return (f"масштаб {M_PER_UNIT:.0f} м/ед · клетка {CELL_M:.0f} м · поле {ARENA:.0f} ед = "
            f"{ARENA_M:.0f} м · сетка {GRID_N}x{GRID_N} · состав {N_SIDE} на сторону")


def units_of_cell(fx, fy):
    """Мировая точка в клетках -> игровые единицы 0..ARENA (в них пишется сценарий)."""
    return fx * CELL_UNITS, fy * CELL_UNITS


def cell_of_units(ux, uy):
    return ux / CELL_UNITS, uy / CELL_UNITS
