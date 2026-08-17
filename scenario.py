"""Эталонные сценарии: вырезка карты + расставленные вручную объекты и позиции сторон.

Зачем отдельно от обучения: обучать на 15 фиксированных сценариях вредно (агент их заучит),
а вот МЕРИТЬ на них — то, что нужно. Всегда одна и та же задача, значит прогресс между
прогонами виден без шума случайной генерации.

Формат (JSON), координаты в ИГРОВЫХ единицах 0..ARENA:
    {
      "name":     "деревня на перекрёстке",
      "map":      "maps/tactical_crop_3",     # без расширения; null = процедурная карта
      "map_seed": 0,                          # seed для процедурной карты, если map = null
      "zones":    [[45.2, 50.1], ...],
      "friendly": [[x, y], ...],              # ровно n_side штук, порядок = порядок FORCE
      "enemy":    [[x, y], ...]
    }

Состав сил менять нельзя: n_side зашит в размер obs/action, и сценарий с другим составом
не загрузится в ту же модель. Сценарий задаёт ПОЗИЦИИ, а не наряд сил.
"""
import glob
import json
import os


def path_of(name_or_path):
    """Принимает и 'scenarios/foo', и 'foo', и 'scenarios/foo.json'."""
    p = name_or_path
    if not p.endswith(".json"):
        p += ".json"
    if not os.path.exists(p):
        alt = os.path.join("scenarios", os.path.basename(p))
        if os.path.exists(alt):
            return alt
    return p


def load(name_or_path):
    with open(path_of(name_or_path), "r", encoding="utf-8") as f:
        return json.load(f)


def save(sc, name_or_path):
    p = path_of(name_or_path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False, indent=2)
    return p


def find_all(pattern="scenarios/*.json"):
    return sorted(glob.glob(pattern))


def validate(sc, n_side, n_zones=None):
    """Возвращает список проблем (пустой = всё в порядке). Лучше поймать здесь, чем получить
    молчаливо кривой замер."""
    problems = []
    for key in ("friendly", "enemy"):
        got = len(sc.get(key, []))
        if got != n_side:
            problems.append(f"{key}: {got} позиций, а состав стороны — {n_side}")
    if n_zones is not None and len(sc.get("zones", [])) != n_zones:
        problems.append(f"zones: {len(sc.get('zones', []))} шт, а среда создана с n_zones={n_zones}")
    m = sc.get("map")
    if m and not os.path.exists(m + ".npy"):
        problems.append(f"карта не найдена: {m}.npy")
    return problems
