"""Дымовой прогон редактора без mainloop: стартовый экран, слои, рисование, подложка, точки,
сохранение карты и сценария, чтение всего обратно.

Смысл теста в двух местах. Первое: карта для игры — это СВЕДЁННАЯ стопка, и надо убедиться, что
сводится она правильно (верхний слой перекрывает нижний, «пусто» пропускает, скрытый слой не
идёт). Второе: результат должен грузиться игровым terrain.from_grid, а сценарий — проходить
scenario.validate(). Заодно проверяются оси (низ экрана = малый Y) и обратимость экран↔мир при
ПОВЁРНУТОМ виде: на осях в проекте горели дважды, с поворотом ошибиться ещё легче.

    py -3.12 editor/check_gui.py
"""
import json
import os
import sys
import tkinter as tk

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from PIL import Image  # noqa: E402

import map_editor as ed  # noqa: E402
import project as P  # noqa: E402
import scenario  # noqa: E402
import terrain  # noqa: E402

NAME = "_smoke_test"
FAILED = []


def ok(cond, what):
    print(("  ok  " if cond else "ПРОВАЛ ") + what)
    if not cond:
        FAILED.append(what)


class Ev:
    """Подделка события мыши: tkinter-обработчикам нужны только координаты и delta."""

    def __init__(self, x, y, delta=0):
        self.x, self.y, self.delta = x, y, delta


def main():
    root = tk.Tk()
    root.geometry("1400x880")
    app = ed.App(root)
    root.update()
    ok(isinstance(app.frame, ed.StartScreen), "при запуске показан стартовый экран, а не холст")
    ok(app.frame.lst.size() > 0, f"перечислены карты из maps/ ({app.frame.lst.size()})")

    app.frame._create(P.GRID_N, P.GRID_N, P.CELL_M)
    root.update()
    e = app.frame
    ok(isinstance(e, ed.EditorFrame), "после создания карты открылся редактор")
    e.W, e.H = 1040, 840
    e.fit_view()
    e._refresh_layers(select_model=0)
    Gx, Gy = e.doc.shape
    print(f"      сетка {e.doc.shape}, клетка {e.doc.cell_m:.0f} м, зум {e.view.zoom:.1f} px/клетка")

    # --- вид
    cx, cy = e.view.to_world(e.W, e.H, e.W / 2, e.H / 2)
    ok(abs(cx - Gx / 2) < 0.5 and abs(cy - Gy / 2) < 0.5, "центр экрана = центр карты")
    ok(e.view.to_world(e.W, e.H, e.W / 2, e.H - 5)[1] < e.view.to_world(e.W, e.H, e.W / 2, 5)[1],
       "ось Y: низ экрана — малый Y (свои внизу поля)")
    p0 = e.view.to_world(e.W, e.H, 300, 250)
    e.on_wheel(Ev(300, 250, delta=120))
    p1 = e.view.to_world(e.W, e.H, 300, 250)
    ok(abs(p0[0] - p1[0]) < 0.05 and abs(p0[1] - p1[1]) < 0.05,
       "зум колесом не уводит точку из-под курсора")
    e.fit_view()
    e.rotate_view(37)
    w = e.view.to_world(e.W, e.H, 420, 380)
    sx, sy = e.view.to_screen(e.W, e.H, *w)
    ok(abs(sx - 420) < 0.01 and abs(sy - 380) < 0.01, "при повёрнутом виде экран<->мир обратимы")
    e.rotate_view(-37)

    # --- рисование в базовом слое
    ok(len(e.doc.layers) == 1 and e.doc.layers[0].kind == "tiles",
       "новая карта — один слой местности «основа»")
    e.tile.set(1); e.tool.set("brush"); e.brush.set(3)
    e.on_press(Ev(300, 300)); e.on_motion(Ev(520, 430)); e.on_release(Ev(520, 430))
    base_forest = int((e.doc.layers[0].grid == 1).sum())
    ok(base_forest > 0, f"кисть положила лес в основу ({base_forest} клеток)")

    # --- второй слой: дороги отдельно
    e.add_tile_layer()
    roads = e.doc.active_layer()
    ok(len(e.doc.layers) == 2 and (roads.grid == ed.EMPTY).all(),
       "новый слой местности создан пустым")
    e.tile.set(4); e.tool.set("line")
    e.on_press(Ev(200, 500)); e.on_motion(Ev(900, 505)); e.on_release(Ev(900, 505))
    ok(int((roads.grid == 4).sum()) > 0 and int((e.doc.layers[0].grid == 4).sum()) == 0,
       "линия легла в активный слой, основа не тронута")
    comp = e.doc.composite()
    ok((comp == 4).sum() == (roads.grid == 4).sum(), "сведение: верхний слой перекрывает нижний")
    ok((comp == 1).sum() > 0, "сведение: сквозь «пусто» виден нижний слой")

    # --- видимость влияет на сведение, замок и скрытость запрещают правку
    roads.visible = False
    e.doc.bump()
    ok((e.doc.composite() == 4).sum() == 0, "скрытый слой в сведённую карту не идёт")
    ok(e._paint_target() is None, "в скрытый слой рисовать нельзя")
    roads.visible = True
    e.doc.bump()
    roads.locked = True
    ok(e._paint_target() is None, "слой под замком не правится")
    roads.locked = False
    ok(e._paint_target() is roads, "разблокированный видимый слой снова пишется")

    # --- ластик открывает нижний слой
    e.tool.set("brush"); e.brush.set(2)
    before = int((e.doc.composite() == 4).sum())
    e.on_right(Ev(400, 503)); e.on_release(Ev(400, 503))
    ok(int((e.doc.composite() == 4).sum()) < before, "ПКМ стирает в «пусто», открывая слой ниже")

    # --- заливка берёт область по видимой карте, пишет в активный слой
    e.add_tile_layer()
    water = e.doc.active_layer()
    e.tile.set(3); e.tool.set("fill")
    e.on_press(Ev(150, 150)); e.on_release(Ev(150, 150))
    ok(int((water.grid == 3).sum()) > 0 and int((e.doc.layers[0].grid == 3).sum()) == 0,
       "заливка пишет в активный слой, а область берёт по видимой карте")

    # --- дублирование, порядок, объединение, сведение
    n_before = len(e.doc.layers)
    e.duplicate_layer()
    ok(len(e.doc.layers) == n_before + 1, "слой дублируется")
    e.delete_layer()
    ok(len(e.doc.layers) == n_before, "слой удаляется")
    e.doc.active = 1
    e._refresh_layers(select_model=1)
    top_before = e.doc.layers[1].name
    e.move_layer(+1)
    ok(e.doc.layers[2].name == top_before, "слой поднимается по стопке")
    e.move_layer(-1)
    comp_before = e.doc.composite().copy()
    e.doc.active = 1
    e.merge_down()
    ok(len(e.doc.layers) == n_before - 1 and np.array_equal(e.doc.composite(), comp_before),
       "объединение с нижним не меняет вид карты")
    e.flatten()
    ok(len(e.doc.tile_layers()) == 1 and np.array_equal(e.doc.composite(), comp_before),
       "сведение всего не меняет вид карты")

    # --- отмена возвращает всю стопку
    stack_before = len(e.doc.layers)
    e.add_tile_layer()
    e.do_undo()
    ok(len(e.doc.layers) == stack_before, "отмена возвращает стопку слоёв")
    e.do_redo()
    ok(len(e.doc.layers) == stack_before + 1, "возврат снова добавляет слой")
    e.do_undo()

    # --- подложка
    src = Image.open(os.path.join(P.MAPS, "preview", "platoon_crop_0.png")).convert("RGBA")
    L = ed.ImageLayer(src, "подложка", (Gx / 2, Gy / 2), max(Gx / src.width, Gy / src.height),
                      path=os.path.join(P.MAPS, "preview", "platoon_crop_0.png"))
    e.doc.layers.append(L)
    e.doc.bump()
    e._refresh_layers(select_model=len(e.doc.layers) - 1)
    L.angle, L.opacity = 30.0, 0.5
    e.draw()
    ok(e._paint_target() is None, "с активной подложкой рисование отклоняется с подсказкой")
    e.layer_move.set(True)
    e.on_press(Ev(400, 400)); e.on_motion(Ev(470, 440)); e.on_release(Ev(470, 440))
    ok(abs(L.cx - Gx / 2) > 0.5 or abs(L.cy - Gy / 2) > 0.5, "подложка тащится мышью")
    e.layer_move.set(False)
    ok(np.array_equal(e.doc.composite(), comp_before), "подложка не попадает в сведённую карту")

    # --- точки
    e.doc.active = 0
    e._refresh_layers(select_model=0)
    e.tool.set("marker")
    e.marker_kind.set("zones")
    for x in (400, 560):
        e.on_press(Ev(x, 420)); e.on_release(Ev(x, 420))
    for kind, y in (("friendly", 740), ("enemy", 130)):
        e.marker_kind.set(kind)
        for i in range(P.N_SIDE + 1):                # на одну больше, чем слотов
            e.on_press(Ev(260 + i * 95, y)); e.on_release(Ev(260 + i * 95, y))
    ok(len(e.doc.markers["zones"]) == 2, "объекты захвата ставятся")
    ok(len(e.doc.markers["friendly"]) == P.N_SIDE and len(e.doc.markers["enemy"]) == P.N_SIDE,
       f"позиций на сторону ровно по составу ({P.N_SIDE}), лишняя отклонена")
    e.on_right(Ev(260, 740))
    ok(len(e.doc.markers["friendly"]) == P.N_SIDE - 1, "ПКМ убирает точку")
    e.marker_kind.set("friendly")
    e.on_press(Ev(260, 740)); e.on_release(Ev(260, 740))
    e._refresh_points()
    ok(not e._scenario_problems(), "сценарий считается полным")

    e.remeasure()
    print("\nпанель замера:\n" + e.info.get("1.0", "end").strip() + "\n")

    # --- сохранение и чтение обратно
    e.name_var.set(NAME)
    e.save_map(); e.save_preview()
    grid = np.load(os.path.join(P.MAPS, NAME + ".npy"))
    with open(os.path.join(P.MAPS, NAME + ".json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    tm = terrain.from_grid(grid, meta["cell_m"] / P.M_PER_UNIT)
    ok(np.array_equal(grid, e.doc.composite()), "в maps/*.npy записана именно сведённая карта")
    ok(grid.dtype == np.int8 and grid.min() >= 0, "в карте нет «пусто» — все клетки определены")
    ok(abs(tm.width_m - P.ARENA) < 0.01, f"поле {tm.width_m:.0f} ед совпадает с ARENA {P.ARENA:.0f}")
    ok(os.path.exists(os.path.join(P.MAPS, NAME + ".editor.npz")), "стопка слоёв записана в .editor.npz")

    doc2 = ed.load_doc(os.path.join(P.MAPS, NAME + ".npy"))
    ok(len(doc2.layers) == len(e.doc.layers), f"стопка читается обратно целиком ({len(doc2.layers)} слоёв)")
    ok(np.array_equal(doc2.composite(), e.doc.composite()), "перечитанная стопка сводится в ту же карту")
    ok(any(L.kind == "image" for L in doc2.layers), "подложка восстановлена по пути к файлу")
    ok(len(doc2.markers["zones"]) == 2 and len(doc2.markers["enemy"]) == P.N_SIDE,
       "точки читаются обратно")

    sc_path = os.path.join(P.SCENARIOS, NAME + ".json")
    os.makedirs(P.SCENARIOS, exist_ok=True)
    with open(sc_path, "w", encoding="utf-8") as f:
        json.dump(ed.scenario_dict(NAME, e.doc.markers), f, ensure_ascii=False, indent=2)
    os.chdir(ROOT)                       # scenario.validate ищет карту относительно корня
    sc = scenario.load(sc_path)
    problems = scenario.validate(sc, P.N_SIDE, n_zones=len(sc["zones"]))
    ok(not problems, "сценарий проходит scenario.validate(): " + (", ".join(problems) or "без замечаний"))
    ok(all(0 <= v <= P.ARENA for p in sc["friendly"] + sc["enemy"] + sc["zones"] for v in p),
       "все координаты сценария лежат в 0..ARENA")

    app.show_start()
    root.update()
    ok(isinstance(app.frame, ed.StartScreen), "«закрыть карту» возвращает на стартовый экран")

    for p in (NAME + ".npy", NAME + ".json", NAME + ".editor.npz",
              os.path.join("preview", NAME + ".png")):
        f = os.path.join(P.MAPS, p)
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(sc_path):
        os.remove(sc_path)
    root.destroy()
    print("\nПРОВАЛОВ: " + (str(len(FAILED)) + " — " + "; ".join(FAILED) if FAILED else "нет"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
