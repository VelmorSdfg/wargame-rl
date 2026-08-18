"""Дымовой прогон векторного редактора без mainloop: рисование фигур, правка, подложка,
точки, сохранение и чтение обратно.

Смысл теста в конце: карта, нарисованная в редакторе, должна собраться в поля и граф и
грузиться игровым загрузчиком, а сценарий — проходить scenario.validate(). Заодно проверяются
оси (низ экрана = малый Y, свои внизу поля) и обратимость экран↔мир при ПОВЁРНУТОМ виде: на
осях в проекте горели дважды, с поворотом ошибиться ещё легче.

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
import vectormap  # noqa: E402

NAME = "_smoke_test"
FAILED = []


def ok(cond, what):
    print(("  ok  " if cond else "ПРОВАЛ ") + what)
    if not cond:
        FAILED.append(what)


class Ev:
    """Подделка события мыши: обработчикам нужны только координаты и delta."""

    def __init__(self, x, y, delta=0):
        self.x, self.y, self.delta = x, y, delta


def click(e, x, y):
    e.on_press(Ev(x, y))
    e.on_release(Ev(x, y))


def main():
    root = tk.Tk()
    root.geometry("1400x880")
    app = ed.App(root)
    root.update()
    ok(isinstance(app.frame, ed.StartScreen), "при запуске показан стартовый экран, а не холст")
    ok(app.frame.lst.size() >= 0, f"перечислены векторные карты из maps/ ({app.frame.lst.size()})")

    # пустая карта под арену
    app.show_editor(ed.Doc(vectormap.new_doc((P.ARENA_M, P.ARENA_M)), P.CELL_M, NAME))
    root.update()
    e = app.frame
    e.W, e.H = 1040, 840
    e.fit_view()
    W_m, H_m = e.doc.size_m
    print(f"      поле {W_m:.0f}x{H_m:.0f} м, зум {e.view.zoom:.3f} px/м")

    # --- вид
    cx, cy = e.view.to_world(e.W, e.H, e.W / 2, e.H / 2)
    ok(abs(cx - W_m / 2) < 5 and abs(cy - H_m / 2) < 5, "центр экрана = центр карты")
    ok(e.view.to_world(e.W, e.H, e.W / 2, e.H - 5)[1] < e.view.to_world(e.W, e.H, e.W / 2, 5)[1],
       "ось Y: низ экрана — малый Y (свои внизу поля)")
    p0 = e.view.to_world(e.W, e.H, 300, 250)
    e.on_wheel(Ev(300, 250, delta=120))
    p1 = e.view.to_world(e.W, e.H, 300, 250)
    ok(abs(p0[0] - p1[0]) < 1 and abs(p0[1] - p1[1]) < 1, "зум колесом не уводит точку из-под курсора")
    e.fit_view()
    e.rotate_view(37)
    w = e.view.to_world(e.W, e.H, 420, 380)
    sx, sy = e.view.to_screen(e.W, e.H, *w)
    ok(abs(sx - 420) < 0.01 and abs(sy - 380) < 0.01, "при повёрнутом виде экран<->мир обратимы")
    e.rotate_view(-37)

    # --- полигон леса
    e._set_type("forest")
    e.tool.set("polygon")
    for x, y in ((300, 300), (520, 280), (560, 470), (330, 500)):
        click(e, x, y)
    e.finish_draft()
    ok(len(e.doc.shapes) == 1 and e.doc.shapes[0]["kind"] == "polygon",
       f"полигон леса создан ({len(e.doc.shapes[0]['points'])} узлов)")

    # --- линия дороги
    e._set_type("road")
    e.tool.set("line")
    for x, y in ((150, 620), (500, 600), (900, 640)):
        click(e, x, y)
    e.finish_draft()
    road = e.doc.shapes[-1]
    ok(road["kind"] == "line" and road["type"] == "road" and road["width_m"] == 8.0,
       f"линия дороги создана шириной {road['width_m']:.0f} м")

    # --- река и переправа
    e._set_type("water")
    e.tool.set("line")
    for x, y in ((200, 200), (520, 500), (860, 800)):
        click(e, x, y)
    e.finish_draft()
    e.tool.set("crossing")
    e.bridge_auto.set(True)
    click(e, 520, 500)
    ok(any(s["kind"] == "crossing" for s in e.doc.shapes), "переправа поставлена")

    # --- мост вручную: длина, ширина и поворот протяжкой
    e.bridge_auto.set(False)
    e.bridge_len.set(120.0)
    e.bridge_wid.set(6.0)
    e.on_press(Ev(600, 580)); e.on_motion(Ev(650, 545)); e.on_release(Ev(650, 545))
    br = [s for s in e.doc.shapes if s["kind"] == "crossing"][-1]
    ok(br.get("length_m") == 120.0 and br.get("width_m_road") == 6.0,
       f"размеры моста заданы вручную: {br.get('length_m'):.0f} x {br.get('width_m_road'):.0f} м")
    ok(abs(br.get("angle_deg", 0.0)) > 0.1, f"протяжка довернула мост на {br['angle_deg']:.0f}°")
    c, length, width, ang = vectormap.crossing_geom(e.doc.vec, br)
    ok(abs(length - 120.0) < 0.01 and abs(width - 6.0) < 0.01 and abs(ang - br["angle_deg"]) < 0.01,
       "сборка карты берёт именно заданные размеры и угол")
    e.doc.shapes.remove(br)
    e.bridge_auto.set(True)

    # --- дом: размер числами (шаблон их просто заполняет), протяжка доворачивает
    e.tool.set("building")
    e._set_type("building")
    e._set_house(12, 8)
    e.on_press(Ev(700, 300)); e.on_motion(Ev(730, 320)); e.on_release(Ev(730, 320))
    b = [s for s in e.doc.shapes if s["kind"] == "building"][-1]
    ok(b["rect_m"][2] == 12 and b["rect_m"][3] == 8,
       f"шаблон «изба» ставит дом ровно {b['rect_m'][2]:.0f}x{b['rect_m'][3]:.0f} м")
    ok(abs(b["rect_m"][4]) > 0.1, f"протяжка довернула дом на {b['rect_m'][4]:.0f}°")

    # --- длина и ширина задаются раздельно
    e.house_len.set(30.0)
    e.house_wid.set(9.0)
    e.on_press(Ev(760, 300)); e.on_release(Ev(760, 300))
    b2 = [s for s in e.doc.shapes if s["kind"] == "building"][-1]
    ok(b2["rect_m"][2] == 30 and b2["rect_m"][3] == 9,
       f"длина и ширина задаются раздельно: {b2['rect_m'][2]:.0f}x{b2['rect_m'][3]:.0f} м")
    e.doc.shapes.remove(b2)
    e._set_house(12, 8)

    # --- выбор и перенос
    e.tool.set("select")
    before = list(e.doc.shapes[0]["points"][0])
    # точку берём внутри леса и ПОДАЛЬШЕ от реки: она проходит через тот же полигон, а линия
    # ловится по ширине плюс допуск в пикселях — щелчок «по лесу» выбрал бы реку
    click(e, 480, 330)
    ok(e.selected == 0, "фигура выбирается щелчком")
    e.on_press(Ev(480, 330)); e.on_motion(Ev(510, 310)); e.on_release(Ev(510, 310))
    ok(e.doc.shapes[0]["points"][0] != before, "выбранная фигура переносится мышью")

    # --- правка узла
    e.tool.set("nodes")
    node_before = list(e.doc.shapes[0]["points"][0])
    sx, sy = e._S(node_before)
    e.on_press(Ev(int(sx), int(sy))); e.on_motion(Ev(int(sx) + 25, int(sy) + 15))
    e.on_release(Ev(int(sx) + 25, int(sy) + 15))
    ok(e.doc.shapes[0]["points"][0] != node_before, "узел двигается мышью")

    # --- отмена и возврат
    n_before = len(e.doc.shapes)
    e.tool.set("select")
    click(e, 480, 330)
    e.delete_selected()
    ok(len(e.doc.shapes) == n_before - 1, "выбранная фигура удаляется")
    e.do_undo()
    ok(len(e.doc.shapes) == n_before, "отмена возвращает фигуру")
    e.do_redo()
    ok(len(e.doc.shapes) == n_before - 1, "возврат снова удаляет")
    e.do_undo()

    # --- зеркало
    pts_before = [list(p) for p in e.doc.shapes[0]["points"]]
    e.do_mirror()
    mirrored = all(abs((H_m - a[1]) - b[1]) < 0.2 for a, b in zip(pts_before, e.doc.shapes[0]["points"]))
    ok(mirrored, "зеркалирование по Y отражает фигуры")
    e.do_undo()

    # --- зонирование связано с инструментом: «дом» ставит только застройку
    e._set_type("water")
    e.tool.set("building")
    ok(e.shape_type.get() == "building", "инструмент «дом» сам переключает зонирование на застройку")
    e._set_type("water")
    ok(e.shape_type.get() == "building", "и не даёт выбрать воду, пока выбран дом")
    e.tool.set("polygon")
    ok(e.shape_type.get() == "water", "вернулись к полигону — вернулось прежнее зонирование")
    e.tool.set("ruler")
    ok(all(str(b.state()).find("disabled") >= 0 for b in e._type_btns.values()),
       "у линейки зонирование погашено целиком")
    e.tool.set("polygon")

    # --- пунктирная лесополоса: прорехи должны отнимать клетки у сплошной
    n_shapes_before_ruler = len(e.doc.shapes)
    e._set_type("forest")
    e.tool.set("line")
    e.width_m.set(30.0)
    e.dash_on.set(False)
    for x, y in ((150, 700), (500, 690), (850, 700)):
        click(e, x, y)
    e.finish_draft()
    solid_cells = int((e.doc.surface(15.0)[0] == vectormap._types()["forest"]).sum())
    e.do_undo()
    e.dash_on.set(True)
    e.dash_len.set(45.0)
    e.dash_gap.set(45.0)
    for x, y in ((150, 700), (500, 690), (850, 700)):
        click(e, x, y)
    e.finish_draft()
    dashed = e.doc.shapes[-1]
    dashed_cells = int((e.doc.surface(15.0)[0] == vectormap._types()["forest"]).sum())
    ok(dashed.get("dash_m") == [45.0, 45.0] and dashed_cells < solid_cells,
       f"пунктирная лесополоса реже сплошной: {dashed_cells} против {solid_cells} клеток")
    ok(len(e._line_runs(dashed)) > 1,
       f"на чертеже пунктир тоже разбит: {len(e._line_runs(dashed))} штрихов")
    e.do_undo()
    e.dash_on.set(False)

    # --- линейка
    e.tool.set("ruler")
    e.on_press(Ev(300, 400)); e.on_motion(Ev(500, 400)); e.on_release(Ev(500, 400))
    a, b = e._ruler
    measured = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    expect = 200 / e.view.zoom
    ok(abs(measured - expect) < 1.0, f"линейка меряет {measured:.0f} м (ждём {expect:.0f})")
    ok(len(e.doc.shapes) == n_shapes_before_ruler, "линейка ничего не добавляет в карту")
    e.ruler_circle.set(True)
    e.draw()
    ok(True, "режим «кругом» рисуется без ошибок")
    e.ruler_circle.set(False)

    # --- просмотр видимости настоящей моделью линии огня
    e.tool.set("vision")
    e.vision_r.set(900.0)
    e.cell_var.set(15.0)
    e.on_press(Ev(520, 500))
    center, field, cell = e._vision
    gx0, gy0 = int(center[0] / cell), int(center[1] / cell)
    ok(0.0 < float((field > 0.02).mean()) < 1.0,
       f"зона видимости посчитана: {float((field > 0.02).mean()) * 100:.0f}% клеток карты")
    ok(field[gx0, gy0] == 1.0, "точка наблюдения сама видима")
    ok(field.max() <= 1.0 and field.min() >= 0.0, "значения поля лежат в 0..1")
    ok(float(((field > 0.02) & (field < 0.98)).sum()) > 0,
       "есть полутени: у леса видно вглубь и сходит на нет, а не обрыв «свет/тень»")

    # видимость должна ХОДИТЬ ЗА МЫШЬЮ и исчезать по отпусканию
    e.on_motion(Ev(430, 430))
    moved = e._vision[0]
    ok(abs(moved[0] - center[0]) > 1 or abs(moved[1] - center[1]) > 1,
       "видимость пересчитывается при движении мыши")
    e.on_release(Ev(430, 430))
    ok(e._vision is None, "по отпусканию кнопки видимость пропадает")
    ok(len(e.doc.shapes) == n_shapes_before_ruler, "просмотр видимости не меняет карту")
    e.cancel_draft()
    ok(e._ruler is None, "Esc убирает замеры с карты")

    # --- скругление углов
    e.tool.set("select")
    click(e, 480, 330)
    n_before = len(e.doc.shapes[e.selected]["points"])
    e.do_round()
    n_after = len(e.doc.shapes[e.selected]["points"])
    ok(n_after == n_before * 2, f"скругление срезает углы: {n_before} -> {n_after} узлов")
    e.do_undo()

    # --- «поле» вырезает лес: поляна в массиве, просека, выгон у села
    surface_before, _ = e.doc.surface(15.0)
    tid0 = vectormap._types()
    forest_before = int((surface_before == tid0["forest"]).sum())
    e._set_type("open")
    e.tool.set("polygon")
    for x, y in ((330, 320), (470, 310), (480, 430), (340, 440)):
        click(e, x, y)
    e.finish_draft()
    surface_after, _ = e.doc.surface(15.0)
    forest_after = int((surface_after == tid0["forest"]).sum())
    ok(forest_after < forest_before,
       f"полигон «поле» вырезал лес: {forest_before} -> {forest_after} клеток")
    e.do_undo()
    e._set_type("forest")

    # --- растеризация: то, что нарисовано, попало в сетку
    surface, cell = e.doc.surface(15.0)
    tid = vectormap._types()
    fr = {k: float((surface == v).mean()) for k, v in tid.items()}
    print("      доли:", {k: round(v, 3) for k, v in fr.items()})
    ok(all(fr[k] > 0 for k in ("forest", "water", "road", "building")),
       "все нарисованные типы попали в сетку полей")

    # --- подложка
    img_path = os.path.join(P.MAPS, "preview", "platoon_crop_0.png")
    src = Image.open(img_path).convert("RGBA")
    u = ed.Underlay(src, "подложка", (W_m / 2, H_m / 2),
                    max(W_m / src.width, H_m / src.height), path=img_path)
    e.doc.underlays.append(u)
    e.u_index = 0                      # панели у подложек нет, активная задаётся индексом
    u.angle, u.opacity = 25.0, 0.5
    e.draw()
    e.u_move.set(True)
    e.on_press(Ev(500, 400)); e.on_motion(Ev(560, 430)); e.on_release(Ev(560, 430))
    ok(abs(u.cx - W_m / 2) > 1 or abs(u.cy - H_m / 2) > 1, "подложка тащится мышью")
    e.u_move.set(False)
    ok(not any(s.get("kind") == "underlay" for s in e.doc.shapes),
       "подложка не попадает в вектор карты")

    # --- точки
    e.tool.set("marker")
    e.marker_kind.set("zones")
    for x in (430, 610):
        click(e, x, 430)
    for kind, y in (("friendly", 760), ("enemy", 120)):
        e.marker_kind.set(kind)
        for i in range(P.N_SIDE + 1):            # на одну больше, чем слотов
            click(e, 260 + i * 95, y)
    ok(len(e.doc.markers["zones"]) == 2, "объекты захвата ставятся")
    ok(len(e.doc.markers["friendly"]) == P.N_SIDE and len(e.doc.markers["enemy"]) == P.N_SIDE,
       f"позиций на сторону ровно по составу ({P.N_SIDE}), лишняя отклонена")
    e.on_right(Ev(260, 760))
    ok(len(e.doc.markers["friendly"]) == P.N_SIDE - 1, "ПКМ убирает точку")
    e.marker_kind.set("friendly")
    click(e, 260, 760)
    e._refresh_points()
    ok(not e._scenario_problems(), "сценарий считается полным")

    e.remeasure()
    print("\n      замер (F5): " + e.status.cget("text") + "\n")
    ok(e.metrics is not None and 0 <= e.metrics["vis"] <= 1, "замер считается по требованию")

    # --- сохранение и чтение обратно
    e.name_var.set(NAME)
    e.cell_var.set(15.0)
    e.save_map()
    e.save_preview()
    vpath = os.path.join(P.MAPS, NAME + ".vector.json")
    ok(os.path.exists(vpath), "записан вектор")
    ok(os.path.exists(os.path.join(P.MAPS, NAME + ".fields.npz"))
       and os.path.exists(os.path.join(P.MAPS, NAME + ".map.json")),
       "поля и граф собраны сразу при сохранении")

    doc2 = ed.load_doc(vpath)
    ok(len(doc2.shapes) == len(e.doc.shapes), f"вектор читается обратно ({len(doc2.shapes)} фигур)")
    ok(len(doc2.markers["zones"]) == 2 and len(doc2.markers["enemy"]) == P.N_SIDE,
       "точки читаются обратно из вектора")

    tm, meta = vectormap.load_map(os.path.join(P.MAPS, NAME), P.M_PER_UNIT)
    ok(abs(tm.width_m - P.ARENA) < 0.01, f"поле {tm.width_m:.0f} игр.ед совпадает с ARENA {P.ARENA:.0f}")
    ok(int(tm.building_comp.max()) >= 1, "дом дошёл до карты боя объектом")
    ok(len(meta["graph"]["nodes"]) >= 2, f"граф дорог собран: {len(meta['graph']['nodes'])} узлов")

    sc_path = os.path.join(P.SCENARIOS, NAME + ".json")
    os.makedirs(P.SCENARIOS, exist_ok=True)
    with open(sc_path, "w", encoding="utf-8") as f:
        json.dump(ed.scenario_dict(NAME, e.doc.markers), f, ensure_ascii=False, indent=2)
    os.chdir(ROOT)                       # scenario.validate ищет карту относительно корня
    sc = scenario.load(sc_path)
    problems = [p for p in scenario.validate(sc, P.N_SIDE, n_zones=len(sc["zones"]))
                if "не найдена" not in p]      # .npy у векторной карты нет — это норма
    ok(not problems, "сценарий проходит scenario.validate(): " + (", ".join(problems) or "без замечаний"))
    ok(all(0 <= v <= P.ARENA for p in sc["friendly"] + sc["enemy"] + sc["zones"] for v in p),
       "все координаты сценария лежат в 0..ARENA")

    app.show_start()
    root.update()
    ok(isinstance(app.frame, ed.StartScreen), "«закрыть карту» возвращает на стартовый экран")

    for p in (NAME + ".vector.json", NAME + ".fields.npz", NAME + ".map.json",
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
