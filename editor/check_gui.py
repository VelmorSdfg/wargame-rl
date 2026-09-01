"""Дымовой прогон векторного редактора без mainloop: рисование фигур, правка, подложка,
точки, сохранение и чтение обратно.

Смысл теста в конце: карта, нарисованная в редакторе, должна собраться в поля и граф и
грузиться игровым загрузчиком, а сценарий — проходить scenario.validate(). Заодно проверяются
оси (низ экрана = малый Y, свои внизу поля) и обратимость экран↔мир при ПОВЁРНУТОМ виде: на
осях в проекте горели дважды, с поворотом ошибиться ещё легче.

    py -3.12 editor/check_gui.py
"""
import io
import json
import math
import re
import time
import os
import sys
import tkinter as tk
from tkinter import ttk

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
    """Подделка события мыши: обработчикам нужны координаты, delta и state.

    state — биты клавиш-модификаторов, как их кладёт tkinter; 0x0001 это Shift. Он нужен
    рамке выделения и добавлению к выделенному."""

    def __init__(self, x, y, delta=0, state=0):
        self.x, self.y, self.delta, self.state = x, y, delta, state


def click(e, x, y):
    e.on_press(Ev(x, y))
    e.on_release(Ev(x, y))


def main():
    # Весь набор проверяет ЗАКАДРОВЫЙ путь: он снимает кадры подменой _blit, а в виджете с
    # GL-контекстом _blit не зовётся вовсе — кадр уходит прямо в окно. Путь этот никуда не делся,
    # он же запасной для машин без видеокарты.
    #
    # Гасим виджет НА УРОВНЕ МОДУЛЯ, а не у отдельных редакторов. Точечное «у этого и вон у того»
    # уже стоило двадцати ТИХО ПРОПУЩЕННЫХ проверок: блок огорожен «e._gl is not None», а _gl у
    # виджета заводится лишь при первом показе окна — в прогоне без окна он оставался пустым, и
    # набор печатал «провалов нет», не выполнив пятую часть себя.
    real_gl_widget = ed._gl_widget_class
    ed._gl_widget_class = lambda: None
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

    # Узлы дорог — слой РАЗБОРА сети, а не карты: на плане он лежал поверх чертежа терпимо, а в
    # объёме ложится на местность и перекрывает её всюду, где есть дорога. Поэтому по умолчанию
    # выключен. Но предупреждение о разрыве сети от тумблера зависеть не должно — иначе
    # выключенный показ молча гасил бы и диагностику, ради которой граф вообще заведён.
    ok(e.show_graph.get() is False, "узлы дорог по умолчанию не показываются")

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

    # Линейка — ИЗМЕРЕНИЕ, а не объект карты: висящий поверх местности замер мешает смотреть
    # ровно на то, что померили. Поэтому гаснет на отпускании, а тумблер её закрепляет.
    e.tool.set("ruler")
    e.ruler_keep.set(False)
    e.on_press(Ev(300, 300))
    e.on_motion(Ev(430, 370))
    had_ruler = e._ruler is not None
    e.on_release(Ev(430, 370))
    ok(had_ruler and e._ruler is None, "линейка гаснет после отпускания")
    e.ruler_keep.set(True)
    e.on_press(Ev(300, 300))
    e.on_motion(Ev(430, 370))
    e.on_release(Ev(430, 370))
    ok(e._ruler is not None, "с тумблером «оставлять линейку» замер остаётся")
    e.ruler_keep.set(False)
    e.cancel_draft()
    e.tool.set("select")

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

    # --- ПЕРЕСЧЁТ ПЕРЕПРАВ ПОСЛЕ ПРАВКИ
    # Раньше проверка стояла ТОЛЬКО на завершении линии: мост появлялся, если пересечение
    # возникало прямо в момент рисования. Перенос, правка узлов, вставка копией и зеркало его
    # уже не создавали — карта выглядела целой, а была разрезана рекой пополам. Правка пачкой
    # сделала промах вероятнее: двигать дорогу стало дёшево.
    def bridges():
        return [s for s in e.doc.shapes if s["kind"] == "crossing"]

    keep_shapes = len(e.doc.shapes)
    n0 = len(bridges())
    rx, y_road = 0.86 * W_m, 0.86 * H_m
    e._set_type("water")
    e.tool.set("line")
    for wy in (0.70 * H_m, 0.99 * H_m):
        sx, sy = e._S((rx, wy))
        click(e, int(sx), int(sy))
    e.finish_draft()
    e._set_type("road")
    e.tool.set("line")
    for wx in (rx + 120.0, rx + 520.0):
        sx, sy = e._S((wx, y_road))
        click(e, int(sx), int(sy))
    e.finish_draft()
    ok(len(bridges()) == n0, "дорога рядом с рекой, но не через неё — переправы нет")

    # тащим дорогу НА реку: путь, который переправу не ставил вовсе
    e.tool.set("select")
    grab = (rx + 300.0, y_road)
    ok(e._hit_shape(grab) == len(e.doc.shapes) - 1, "под курсором именно та дорога, что тащим")
    sx, sy = e._S(grab)
    tx, ty = e._S((rx - 100.0, y_road))
    e.on_press(Ev(int(sx), int(sy)))
    e.on_motion(Ev(int(tx), int(ty)))
    e.on_release(Ev(int(tx), int(ty)))
    ok(len(bridges()) == n0 + 1, "ПЕРЕНОС дороги на реку ставит переправу")

    # Мост ЕДЕТ ВМЕСТЕ с дорогой. Раньше он оставался на старом месте — посреди поля, — а на
    # новом пересечении пересчёт ставил ещё один: два моста, ни один не там, где нужно. Двигать
    # мост на то же смещение нельзя: когда едет одна дорога, пересечение ползёт ВДОЛЬ реки.
    # Дорогу и реку берём С КОНЦА — те самые, что нарисованы этой проверкой. Первая попытка
    # искала воду с начала карты и находила чужую реку из прежних проверок: пересечений с ней
    # нет, и замер показывал бессмыслицу вместо ошибки.
    br_i = e.doc.shapes.index(bridges()[-1])
    road_i = next(i for i in range(len(e.doc.shapes) - 1, -1, -1)
                  if e.doc.shapes[i]["kind"] == "line"
                  and e.doc.shapes[i].get("type") == "road")
    water_i = next(i for i in range(len(e.doc.shapes) - 1, -1, -1)
                   if e.doc.shapes[i]["kind"] == "line"
                   and e.doc.shapes[i].get("type") == "water")
    ok(bool(vectormap.line_hits(e.doc.shapes[road_i]["points"],
                                e.doc.shapes[water_i]["points"])),
       "до переноса дорога и река той же проверки действительно пересекаются")
    n_before = len(bridges())
    e.tool.set("select")
    e._select([road_i])
    links = e._crossing_links([road_i])
    ok(any(l[0] == br_i for l in links),
       f"переправа привязана к паре дорога-река ({len(links)} связей)")
    grab2 = list(e.doc.shapes[road_i]["points"][0])
    sx2, sy2 = e._S(grab2)
    tx2, ty2 = e._S((grab2[0], grab2[1] + 260.0))
    e.on_press(Ev(int(sx2), int(sy2)))
    e.on_motion(Ev(int(tx2), int(ty2)))
    e.on_release(Ev(int(tx2), int(ty2)))
    hits = vectormap.line_hits(e.doc.shapes[road_i]["points"],
                               e.doc.shapes[water_i]["points"])
    q = e.doc.shapes[br_i]["point"]
    off = min((math.hypot(q[0] - h[0], q[1] - h[1]) for h in hits), default=1e9)
    ok(hits and off < 2.0,
       f"мост уехал вместе с дорогой и остался на пересечении ({off:.2f} м от него)")
    ok(len(bridges()) == n_before,
       f"второй мост при этом не появился: было {n_before}, стало {len(bridges())}")
    e.do_undo()

    # снесённый мост не воскресает: «мост взорван» — это решение, а не оплошность
    e._select([e.doc.shapes.index(bridges()[-1])])
    e.delete_selected()
    ok(len(bridges()) == n0, "переправа снесена — мост взорван")
    e.tool.set("select")
    sx, sy = e._S((rx - 100.0, y_road))
    e.on_press(Ev(int(sx), int(sy)))
    e.on_motion(Ev(int(sx) + 5, int(sy)))
    e.on_release(Ev(int(sx) + 5, int(sy)))
    ok(len(bridges()) == n0, "правка рядом не воскрешает взорванный мост")

    # а отмена сноса возвращает и мост, и его право быть пересчитанным
    e.do_undo()
    e.do_undo()
    ok(len(bridges()) == n0 + 1, "отмена возвращает снесённый мост")

    # брод — та же фигура, но со своей скоростью. Ставится ТОЛЬКО руками: автопостановка ищет
    # пересечения дороги с рекой, а туда нужен мост — дорога, упирающаяся в брод, это не
    # переправа, а место, где дорога тонет.
    e.tool.set("crossing")
    e.bridge_ford.set(True)
    sx, sy = e._S((rx, 0.78 * H_m))
    click(e, int(sx), int(sy))
    ford = e.doc.shapes[-1]
    ok(ford["kind"] == "crossing" and ford.get("ford") is True,
       "переключатель ставит брод, а не мост")
    ok(not any(s.get("ford") for s in bridges() if s is not ford),
       "автопостановка ставит мосты, бродов сама не делает")
    e.bridge_ford.set(False)

    for _ in range(4):
        e.do_undo()
    ok(len(e.doc.shapes) == keep_shapes, "проверка переправ убрала за собой")

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

    # --- рельеф: форма вдавливается в карту высот и фигурой НЕ остаётся
    n_shapes_before = len(e.doc.shapes)
    e.tool.set("relief_poly")
    e._tool_changed()
    ok(all(str(b.state()).find("disabled") >= 0 for b in e._type_btns.values()),
       "у печати рельефа зонирование погашено: высота не материал")
    e._set_relief(40)
    e.relief_slope.set(150.0)
    for x, y in ((600, 250), (780, 240), (800, 400), (620, 420)):
        click(e, x, y)
    e.finish_draft()
    ok(len(e.doc.shapes) == n_shapes_before and "height" in e.doc.vec,
       "холм +40 м ушёл в карту высот, лишней фигуры в векторе не осталось")
    e.tool.set("relief_line")
    e._tool_changed()
    e._set_relief(-20)
    e.relief_width.set(300.0)
    for x, y in ((200, 560), (520, 600), (880, 560)):
        click(e, x, y)
    e.finish_draft()
    h = e.doc.height_m(15.0)
    ok(h is not None and h.max() > 5 and h.min() < -2,
       f"поле высот собралось: от {h.min():.0f} до {h.max():.0f} м")

    # склон задаётся в МЕТРАХ и не зависит от размера карты — раньше он выводился из неё
    probe = ed.vectormap.new_doc((2550.0, 2550.0), [])
    poly = {"kind": "polygon", "type": "relief", "h_m": 50.0,
            "points": [[900, 900], [1600, 900], [1600, 1600], [900, 1600]]}
    steep = ed.vectormap.stamp(dict(probe), [poly], cell_m=5.0, slope_m=20.0)["height"]["h"]
    soft = ed.vectormap.stamp(dict(probe), [poly], cell_m=5.0, slope_m=300.0)["height"]["h"]

    def rise_m(field):
        row = field[:, field.shape[1] // 2]
        return (int(np.argmax(row > 0.95 * row.max()))
                - int(np.argmax(row > 0.05 * row.max()))) * 5.0

    ok(rise_m(steep) < 40 < 150 < rise_m(soft),
       f"склон слушается: обрыв поднимается за {rise_m(steep):.0f} м, пологий за {rise_m(soft):.0f}")
    ok(abs(float(steep.max()) - 50.0) < 0.5 and abs(float(soft.max()) - 50.0) < 0.5,
       f"высота штампа — та, что задана ({float(steep.max()):.1f} и {float(soft.max()):.1f} м)")
    flat = ed.vectormap.stamp(ed.vectormap.stamp(dict(probe), [poly], cell_m=5.0, slope_m=150.0),
                              [{"kind": "polygon", "type": "relief", "h_m": 12.0,
                                "points": [[1100, 1100], [1400, 1100], [1400, 1400],
                                           [1100, 1400]]}],
                              slope_m=40.0, absolute=True)["height"]["h"]
    ok(abs(float(flat[250, 250]) - 12.0) < 0.5,
       f"выравнивание до отметки даёт плато: в середине {float(flat[250, 250]):.1f} м из 12")
    ok(e._shade_overlay() is not None, "отмывка рельефа рисуется на плане")

    # --- топографический стиль: высоты горизонталями ЛИНИЯМИ
    import paint as painter  # noqa: E402
    hh = e.doc.height_m(15.0)
    step = painter.contour_step(float(hh.max() - hh.min()))
    cs = painter.contour_set(hh, 15.0, step)
    ok(step in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0) and len(cs.pts) > 5,
       f"горизонтали считаются линиями: сечение {step:.0f} м, {len(cs.pts)} линий, "
       f"{sum(len(p) for p in cs.pts)} точек")
    # Линия должна ЛЕЖАТЬ на своём уровне — это и есть проверка самой геометрии, а не
    # похожести картинки: берём точки линии и спрашиваем у карты высоту в них.
    hm = {"cell_m": 15.0, "h": hh}
    err = np.concatenate([np.abs(ed.vectormap.sample_height(hm, p[:, 0], p[:, 1]) - k * step)
                          for k, p in list(zip(cs.k, cs.pts))[:300]])
    ok(float(err.mean()) < 1.0,
       f"точка горизонтали лежит на своём уровне: расхождение {err.mean():.2f} м, "
       f"наибольшее {err.max():.2f} м")
    flat = painter.contour_set(np.zeros_like(hh), 15.0, step)
    ok(not flat.pts, "на ровном месте горизонталей нет — иначе поле покрывается рябью")

    # Главное здесь. Набор линий ОДИН на все увеличения, а уровень подробности решает лишь,
    # каждую ли показывать. Раньше каждый кусок считал горизонтали сам по своей выборке
    # высот, и при смене уровня рисовались другие линии — это и было видно как «ломаются».
    mults = [cs.mult_for(mpp) for mpp in (1.25, 5.0, 20.0, 40.0)]
    ok(mults == sorted(mults) and all(m & (m - 1) == 0 for m in mults),
       f"прореживание растёт с масштабом и кратно двум: {mults}")
    near = cs.select(0, 0, 2550, 2550, mults[0])
    far = cs.select(0, 0, 2550, 2550, mults[-1])
    same = all(any(q is p for _, q in near) for _, p in far)
    ok(same and len(far) <= len(near),
       f"дальний вид показывает ПОДМНОЖЕСТВО ближних линий ({len(far)} из {len(near)}), "
       f"а не другие линии")

    # И то же на картинке: один и тот же квадрат мира, нарисованный на двух уровнях
    # подробности, должен дать горизонтали в одних и тех же местах.
    def contour_mask(px, mult):
        img = painter.paint(e.doc.shapes, 300.0, 300.0, 600.0, 600.0, px, px,
                            style="topo", lines=cs, mult=mult)
        d = np.abs(img.astype(np.int16) - np.array(painter.TOPO["open"], dtype=np.int16))
        brown = (img[:, :, 0] > img[:, :, 2] + 20) & (d.sum(axis=2) > 40)
        return brown

    fine = contour_mask(256, 1)
    coarse = contour_mask(64, 1)
    small = fine.reshape(64, 4, 64, 4).any(axis=(1, 3))      # тонкая маска в клетку грубой
    grown = small.copy()
    for sh in (1, -1):
        for ax in (0, 1):
            grown |= np.roll(small, sh, axis=ax)
    hit = float((coarse & grown).sum()) / max(1, int(coarse.sum()))
    ok(hit > 0.9,
       f"линии на разных уровнях подробности совпадают: {hit*100:.0f}% точек грубой "
       f"картинки лежат на линиях подробной")

    e.map_style.set("topo")
    e.draw()
    ok(e._contour_overlay() is not None, "горизонтали ложатся на план вместо отмывки")
    topo = painter.paint(e.doc.shapes, 0.0, 0.0, 600.0, 600.0, 128, 128,
                         style="topo", lines=cs)
    live = painter.paint(e.doc.shapes, 0.0, 0.0, 600.0, 600.0, 128, 128)
    ok(topo.shape == live.shape and int(np.abs(topo.astype(int) - live.astype(int)).sum()) > 0,
       "топостиль даёт другую картинку куска, чем живой")
    e.map_style.set("vector")
    e.draw()
    ok("relief" not in ed.SHAPE_TYPES and len(ed.SHAPE_TYPES) == 5,
       f"зонирование снова только материалы: {', '.join(ed.SHAPE_TYPES)}")
    e.tool.set("polygon")
    e._tool_changed()

    # --- знак высоты в объёме. Проверка не «красиво ли», а «холм — это вверх»: в проекции
    # вертикаль и глубина стояли местами, и рельеф читался наизнанку — овраг выглядел бугром.
    import view3d  # noqa: E402
    cam = ed.view3d.Camera(target=(1275.0, 1275.0), dist=4000.0, yaw=0.0, pitch=30.0)
    _, sy0, sz0 = view3d.project(np.array([1275.0]), np.array([1275.0]),
                                 np.array([0.0]), cam, (800, 600))
    _, sy1, sz1 = view3d.project(np.array([1275.0]), np.array([1275.0]),
                                 np.array([100.0]), cam, (800, 600))
    ok(sy1[0] < sy0[0] and sz1[0] < sz0[0],
       f"высота идёт ВВЕРХ и ближе к камере (экран {sy0[0]:.0f}->{sy1[0]:.0f})")

    # отмывка светит с северо-запада: иначе глаз выворачивает рельеф наизнанку
    G = 48
    gxi, gyi = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    cone = np.clip(1 - np.hypot(gxi - G / 2, gyi - G / 2) / 14, 0, 1).astype(np.float32) * 80
    flat = np.zeros((G, G), dtype=np.int32)
    top = ed.view3d.Camera(target=(1275.0, 1275.0), dist=5200.0, yaw=0.0, pitch=88.0)
    im = np.asarray(view3d.render(flat, cone, 2550.0 / G, top, (400, 300), ground=False)).astype(float)
    nw = im[110:140, 160:190].mean()
    se = im[160:190, 210:240].mean()
    ok(nw > se, f"склон на северо-запад светлее теневого ({nw:.0f} против {se:.0f})")
    im_pit = np.asarray(view3d.render(flat, -cone, 2550.0 / G, top, (400, 300),
                                      ground=False)).astype(float)
    ok(im_pit[110:140, 160:190].mean() < im_pit[160:190, 210:240].mean(),
       "у лощины отмывка ровно обратная — яма не путается с холмом")

    # --- срез грунта: слои под картой
    side = ed.view3d.Camera(target=(1275.0, 1275.0), dist=5200.0, yaw=0.0, pitch=30.0)
    # --- две отрисовки объёма должны давать ОДНУ картинку. Расходятся они молча: на глаз
    # зеркальный поворот заметен только если знать, где что стоит на карте.
    import view3d_gpu  # noqa: E402
    pts = np.array([[0., 0., 0.], [2550., 0., 0.], [0., 2550., 0.], [1275., 1275., 100.]],
                   dtype=np.float32)
    sx, sy, _ = view3d.project(pts[:, 0], pts[:, 1], pts[:, 2] * view3d.VSCALE, side, (400, 300))
    m = view3d_gpu._mvp(side, (400, 300)).T
    hom = np.concatenate([pts[:, :2], pts[:, 2:3] * view3d.VSCALE,
                          np.ones((len(pts), 1), np.float32)], axis=1)
    clip = hom @ m.T
    ndc = clip[:, :3] / clip[:, 3:4]
    ok(np.allclose(sx, (ndc[:, 0] * 0.5 + 0.5) * 400, atol=0.5)
       and np.allclose(sy, (ndc[:, 1] * 0.5 + 0.5) * 300, atol=0.5),
       "камера видеокарты совпадает с программной до полпикселя")

    bare = np.asarray(view3d.render(flat, cone, 2550.0 / G, side, (400, 300), ground=False))
    with_g = np.asarray(view3d.render(flat, cone, 2550.0 / G, side, (400, 300), ground=True))
    diff = (bare != with_g).any(axis=2)
    add = with_g.astype(int)[diff]
    # «теплее неба», а не просто тёплая: на дальних планах дымка уводит все цвета к горизонту,
    # и земля там уже не тёплая сама по себе — а теплее окружающего остаётся всегда
    sky_rb = float(ed.view3d.SKY_HORIZON[0]) - float(ed.view3d.SKY_HORIZON[2])
    warm = int(((add[:, 0] - add[:, 2]) > sky_rb + 4).sum())
    ok(int(diff.sum()) > 500 and warm > 300,
       f"под картой нарисован срез грунта ({int(diff.sum())} точек, из них {warm} земляных)")
    e.do_undo(); e.do_undo()
    e._set_type("forest")

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

    # Мелкое строение до боя не доживает и раньше пропадало МОЛЧА: на чертеже дом есть, в
    # объёме стоит коробкой, а бою не достаётся ничего. Тот же тихий отказ, что был у мостов.
    # Подгонять размеры нельзя — заставь дом занимать клетку, и сарай в 48 м2 станет блоком
    # 30x30, а это ложь в другую сторону. Поэтому замер про такие дома ГОВОРИТ.
    n_keep = len(e.doc.shapes)
    e.push_undo()
    lb_cx, lb_cy = W_m * 0.35, H_m * 0.35
    e.doc.shapes.append({"kind": "building", "rect_m": [lb_cx, lb_cy, 8.0, 6.0, 0.0],
                         "capacity": 1})                      # сарай — не доживёт
    e.doc.shapes.append({"kind": "building", "rect_m": [lb_cx + 200.0, lb_cy, 18.0, 10.0, 0.0],
                         "capacity": 1})                      # дом — доживёт
    e.doc.bump()
    lb_surf, _ = e.doc.surface(e.doc.cell_m)
    lb_lost = e.lost_buildings(lb_surf, e.doc.cell_m)
    ok((n_keep in lb_lost) and (n_keep + 1 not in lb_lost),
       f"замер видит строения без своей клетки: сарай 8x6 помечен, дом 18x10 нет "
       f"(помечено {len(lb_lost)})")
    e.do_undo()

    # ЛИНИЯ ОГНЯ ПО ВЕКТОРУ. У сетки есть порог существования: дом мельче примерно 7x7 м при
    # клетке 30 м не даёт ни одной клетки, и луч проходит сквозь нарисованный дом. Вектор меряет
    # помеху ДЛИНОЙ ЛУЧА ВНУТРИ ФИГУРЫ, порога у него нет, и цена та же.
    sq_ = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    ok(abs(vectormap._poly_inside_len((-5.0, 5.0), (15.0, 5.0), sq_) - 10.0) < 1e-6
       and abs(vectormap._poly_inside_len((5.0, 5.0), (15.0, 5.0), sq_) - 5.0) < 1e-6
       and vectormap._poly_inside_len((-5.0, 50.0), (15.0, 50.0), sq_) == 0.0,
       "длина луча внутри фигуры: насквозь, изнутри наружу и мимо")
    # невыпуклая фигура: буква П, луч режет две ножки по 3
    u_ = [(0, 0), (10, 0), (10, 30), (7, 30), (7, 3), (3, 3), (3, 30), (0, 30)]
    ok(abs(vectormap._poly_inside_len((-5.0, 20.0), (15.0, 20.0), u_) - 6.0) < 1e-6,
       "и на невыпуклой фигуре тоже верно (две ножки по 3)")

    SZ = 2550.0
    los_doc = vectormap.new_doc((SZ, SZ))
    los_doc["shapes"].append({"kind": "building", "rect_m": [SZ / 2, SZ / 2, 8.0, 6.0, 0.0],
                              "capacity": 1})
    lo = vectormap.rasterize(los_doc, cell_m=30.0)
    los_tm = terrain.from_fields(lo[0], dict(lo[1]), 30.0 / P.M_PER_UNIT)
    lp0 = ((SZ / 2 - 300.0) / P.M_PER_UNIT, (SZ / 2) / P.M_PER_UNIT)
    lp1 = ((SZ / 2 + 300.0) / P.M_PER_UNIT, (SZ / 2) / P.M_PER_UNIT)
    was_cells = los_tm.blocked(lp0, lp1)
    los_tm.attach_vector(los_doc, P.M_PER_UNIT)
    ok((not was_cells) and los_tm.blocked(lp0, lp1),
       "сарай 8x6: сеткой луч проходит насквозь, вектором перекрыт")

    # лес копит толщину, а не гасит сразу — порог 90 м, как в бою
    for w_m, want in ((40.0, False), (200.0, True)):
        fd = vectormap.new_doc((SZ, SZ))
        fd["shapes"].append({"kind": "polygon", "type": "forest",
                             "points": [[SZ / 2 - w_m / 2, 0], [SZ / 2 + w_m / 2, 0],
                                        [SZ / 2 + w_m / 2, SZ], [SZ / 2 - w_m / 2, SZ]]})
        vl_ = vectormap.VectorTerrain(fd, P.M_PER_UNIT, cell_u=30.0 / P.M_PER_UNIT)
        ok(vl_.blocked(lp0, lp1) is want,
           f"лес {w_m:.0f} м {'гасит' if want else 'пропускает'} луч (порог 90 м)")
    ok(not vl_.blocked(lp0, lp1, demolish=True),
       "фугас видит сквозь тот же лес: у него свой порог")

    bd = vectormap.new_doc((SZ, SZ))
    bd["shapes"].append({"kind": "building", "rect_m": [SZ / 2, SZ / 2, 40.0, 30.0, 0.0],
                         "capacity": 1})
    vb_ = vectormap.VectorTerrain(bd, P.M_PER_UNIT, cell_u=30.0 / P.M_PER_UNIT)
    ok(vb_.blocked(lp0, lp1) and not vb_.blocked(lp0, lp1, transparent={1}),
       "своё здание прозрачно для своего огня — номера домов те же, что у building_comp")
    ok(not vb_.blocked(((SZ / 2) / P.M_PER_UNIT, (SZ / 2) / P.M_PER_UNIT), lp1),
       "изнутри дома наружу видно: своё укрытие не мешает")

    # УКРЫТИЕ И СКОРОСТЬ тоже по вектору. У клетки материал решался ДОЛЕЙ покрытия, и мелкая
    # фигура спор проигрывала: сарай 8x6 при клетке 30 м не давал ни одной клетки, то есть и
    # укрытия тому, кто в нём сидит. После перевода линии огня на вектор это разошлось совсем —
    # дом пули задерживал, а своего же обитателя не прикрывал.
    cv_doc = vectormap.new_doc((SZ, SZ))
    cv_doc["shapes"].append({"kind": "building", "rect_m": [SZ / 2, SZ / 2, 8.0, 6.0, 0.0],
                             "capacity": 1})
    cv = vectormap.rasterize(cv_doc, cell_m=30.0)
    cv_tm = terrain.from_fields(cv[0], dict(cv[1]), 30.0 / P.M_PER_UNIT)
    cv_mid = (SZ / 2 / P.M_PER_UNIT, SZ / 2 / P.M_PER_UNIT)
    cov_cells = cv_tm.cover_at(cv_mid)
    cv_tm.attach_vector(cv_doc, P.M_PER_UNIT)
    ok(cov_cells == 0.0 and cv_tm.cover_at(cv_mid) > 0.6,
       f"сарай 8x6 укрывает того, кто в нём сидит: сеткой {cov_cells:.2f}, "
       f"вектором {cv_tm.cover_at(cv_mid):.2f}")
    ok(cv_tm.speed_at(cv_mid) < 0.9,
       f"и замедляет: скорость в строении {cv_tm.speed_at(cv_mid):.2f}")

    # переправа главнее спора и здесь: мост быстрее поля, брод медленнее всего
    for is_ford, nm, fast in ((False, "мост", True), (True, "брод", False)):
        cr = {"kind": "crossing", "point": [SZ / 2, SZ / 2]}
        if is_ford:
            cr["ford"] = True
        cd = vectormap.new_doc((SZ, SZ))
        cd["shapes"].extend([{"kind": "line", "type": "water", "width_m": 40.0,
                              "points": [[SZ / 2, 0], [SZ / 2, SZ]]},
                             {"kind": "line", "type": "road", "width_m": 8.0,
                              "points": [[0, SZ / 2], [SZ, SZ / 2]]}, cr])
        co = vectormap.rasterize(cd, cell_m=30.0)
        ct = terrain.from_fields(co[0], dict(co[1]), 30.0 / P.M_PER_UNIT)
        ct.attach_vector(cd, P.M_PER_UNIT)
        ok((ct.speed_at(cv_mid) > 1.0) is fast,
           f"{nm}: скорость {ct.speed_at(cv_mid):.2f} "
           f"({'быстрее' if fast else 'медленнее'} поля)")

    # ВОДА В ЖИВОМ СТИЛЕ ПРОЗРАЧНАЯ. Сплошная заливка читалась как синяя лента поверх карты, а
    # не как река. Прозрачность даёт и полезное: дорога рисуется раньше воды, поэтому под водой
    # остаётся видна — там, где она уходит в реку и не выходит мостом, это сразу заметно.
    # В топостиле вода СПЛОШНАЯ: голубая заливка там условный знак, размывать его нельзя.
    import paint as wpaint  # noqa: E402
    W_SZ = 1000.0
    wet = [{"kind": "line", "type": "road", "width_m": 10.0,
            "points": [[0, 500], [W_SZ, 500]]},
           {"kind": "polygon", "type": "forest",
            "points": [[0, 560], [W_SZ, 560], [W_SZ, 700], [0, 700]]},
           {"kind": "line", "type": "water", "width_m": 120.0,
            "points": [[500, 0], [500, W_SZ]]}]

    def wet_at(img, wx, wy):
        px = np.asarray(img)[int(wy / W_SZ * 255), int(wx / W_SZ * 255)]
        return tuple(int(v) for v in px)          # без int вывод пестрит np.int64(...)

    live_t = wpaint.paint(wet, 0.0, 0.0, W_SZ, W_SZ, 256, 256, style="live", ss=2)
    topo_t = wpaint.paint(wet, 0.0, 0.0, W_SZ, W_SZ, 256, 256, style="topo", ss=2)
    w_road, w_open = wet_at(live_t, 500, 500), wet_at(live_t, 500, 200)
    w_col = tuple(int(v) for v in wpaint._color("water", "live"))
    ok(w_road != w_open and all(abs(a - b) < 70 for a, b in zip(w_open, w_col)),
       f"сквозь воду видно дно и это всё ещё вода: над дорогой {w_road}, "
       f"над полем {w_open}, цвет воды {w_col}")
    ok(wet_at(topo_t, 500, 500) == wet_at(topo_t, 500, 200),
       "в топостиле вода сплошная — условный знак не размывается")

    # --- линейка
    # Значение снимаем ПОКА ДЕРЖИМ кнопку: линейка гаснет на отпускании (она измерение, а не
    # объект карты), и читать её после — значит читать пустоту.
    e.tool.set("ruler")
    e.on_press(Ev(300, 400)); e.on_motion(Ev(500, 400))
    a, b = e._ruler
    e.on_release(Ev(500, 400))
    measured = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    expect = 200 / e.view.zoom
    ok(abs(measured - expect) < 1.0, f"линейка меряет {measured:.0f} м (ждём {expect:.0f})")
    ok(len(e.doc.shapes) == n_shapes_before_ruler, "линейка ничего не добавляет в карту")
    e.ruler_circle.set(True)
    e.draw()
    ok(True, "режим «кругом» рисуется без ошибок")
    e.ruler_circle.set(False)

    # Просмотр прострелов обязан считать РЕЛЬЕФ, а не только материал. Пока он смотрел лишь на
    # лес и строения, холм между наблюдателем и целью для него не существовал: обратный скат
    # подсвечивался как видимый. Обманывало вдвойне — маска ложится на объёмную местность,
    # обтекает холм, и глаз заключает, что холм учтён. Бой при этом рельеф считал, и замер
    # годности карты тоже: расходился ровно тот показ, по которому выбирают позиции.
    vs_cx, vs_cy = W_m / 2, H_m / 2
    e.push_undo()
    h_keep, slope_keep = e.relief_h.get(), e.relief_slope.get()
    e.relief_h.set(60.0)
    e.relief_slope.set(60.0)
    e._stamp_relief({"kind": "polygon", "type": "relief",
                     "points": [[vs_cx - 900, vs_cy + 40], [vs_cx + 900, vs_cy + 40],
                                [vs_cx + 900, vs_cy + 160], [vs_cx - 900, vs_cy + 160]]})
    eye_m = np.array([vs_cx, vs_cy - 400.0], dtype=np.float32)
    vtm, vorg = e._vision_terrain(eye_m, 900.0, e.doc.cell_m)
    vloc = np.array([eye_m[0] - vorg[0], eye_m[1] - vorg[1]], dtype=np.float32)
    vfield = ed.viewshed(vtm, vloc, 900.0, P.M_PER_UNIT)

    def vs_at(wx, wy):
        lx = (wx - vorg[0]) / P.M_PER_UNIT
        ly = (wy - vorg[1]) / P.M_PER_UNIT
        return float(vfield[int(np.clip(lx / vtm.cell, 0, vtm.Gx - 1)),
                            int(np.clip(ly / vtm.cell, 0, vtm.Gy - 1))])

    ok(vs_at(vs_cx, vs_cy - 120.0) > 0.5 and vs_at(vs_cx, vs_cy + 320.0) < 0.5,
       f"просмотр прячет обратный скат за грядой: перед ней "
       f"{vs_at(vs_cx, vs_cy - 120.0):.2f}, за ней {vs_at(vs_cx, vs_cy + 320.0):.2f}")

    # Контроль: сверяем с ТЕМ ЖЕ terrain.blocked, которым считает бой и мерка годности.
    # Убери проверку гребня из viewshed — эта строка покраснеет.
    agree = total = 0
    vp0 = (vloc[0] / P.M_PER_UNIT, vloc[1] / P.M_PER_UNIT)
    for ddy in range(-350, 700, 50):
        for ddx in (-200, 0, 200):
            wx_, wy_ = vs_cx + ddx, vs_cy + ddy
            vp1 = ((wx_ - vorg[0]) / P.M_PER_UNIT, (wy_ - vorg[1]) / P.M_PER_UNIT)
            total += 1
            if (not vtm.blocked(vp0, vp1)) == (vs_at(wx_, wy_) > 0.5):
                agree += 1
    ok(agree >= total * 0.9,
       f"просмотр сходится с боевым terrain.blocked на {agree} из {total} точек")
    e.relief_h.set(h_keep)
    e.relief_slope.set(slope_keep)
    e.do_undo()

    # --- просмотр видимости настоящей моделью линии огня
    e.tool.set("vision")
    e.vision_r.set(900.0)
    e.cell_var.set(15.0)
    e.on_press(Ev(520, 500))
    center, field, cell, origin = e._vision
    gx0, gy0 = int((center[0] - origin[0]) / cell), int((center[1] - origin[1]) / cell)
    e.vision_r.set(300.0)
    e._vis_tm = None                     # окно переиспользуется, пока точка внутри: для замера
    e.on_press(Ev(520, 500))             # берём его заново под меньший радиус
    small = e._vision[1].shape[0]
    e.vision_r.set(900.0)
    e.on_press(Ev(520, 500))
    ok(small * cell < e.doc.size_m[0],
       f"видимость считается по ОКНУ вокруг точки, а не по всей карте: на радиусе 300 м это "
       f"{small} клеток из {int(e.doc.size_m[0] / cell)}")
    center, field, cell, origin = e._vision
    gx0, gy0 = int((center[0] - origin[0]) / cell), int((center[1] - origin[1]) / cell)
    ok(0.0 < float((field > 0.02).mean()) < 1.0,
       f"зона видимости посчитана: {float((field > 0.02).mean()) * 100:.0f}% клеток окна")
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

    # --- правка пачкой: рамка, Shift, перенос группой, копирование
    e.tool.set("select")

    class SEv(Ev):
        """Событие с зажатым Shift: бит 0x0001 в state, как его шлёт tk.

        Раньше бит стоял АТРИБУТОМ КЛАССА и работал лишь потому, что Ev не заводил своего
        поля state, — класс просвечивал насквозь. Как только у Ev появилось это поле (оно
        понадобилось рамке выделения), атрибут экземпляра заслонил классовый, и Shift молча
        перестал доходить: выделение пачкой «перестало работать» на верном коде."""

        def __init__(self, x, y, delta=0, state=0x0001):
            Ev.__init__(self, x, y, delta, state)

    base = len(e.doc.shapes)
    for k in range(4):                       # четыре домика рядком
        e._place_building(np.array([600.0 + k * 60, 1800.0], dtype=np.float32))
        e.on_release(Ev(0, 0))
    ok(len(e.doc.shapes) == base + 4, f"поставлены четыре дома для правки пачкой")
    e.tool.set("select")
    e._select([])
    # рамка по пустому месту: тянем через все четыре. Начало ищем там, где под курсором
    # ничего нет: рамка стартует только с пустого места, иначе это перенос фигуры
    corner = None
    for cx, cy in ((520.0, 1740.0), (520.0, 1700.0), (500.0, 1690.0), (540.0, 1710.0),
                   (520.0, 1900.0), (900.0, 1700.0)):
        if e._hit_shape((cx, cy)) is None:
            corner = (cx, cy)
            break
    ok(corner is not None, "нашлось пустое место для начала рамки")
    a = e._S(corner)
    b = e._S((max(880.0, corner[0] + 380), 1860.0 if corner[1] < 1860 else corner[1] - 200))
    e.on_press(Ev(int(a[0]), int(a[1])))
    e.on_motion(Ev(int(b[0]), int(b[1])))
    e.on_release(Ev(int(b[0]), int(b[1])))
    ok(len(e.sel) >= 4, f"рамка выделила несколько фигур разом: {len(e.sel)}")
    picked = list(e.sel)
    before = [list(e.doc.shapes[i]["rect_m"][:2]) for i in picked
              if e.doc.shapes[i]["kind"] == "building"]
    # тащим за одну из них — ехать должны все
    g0 = e._S(tuple(before[0]))
    e.on_press(Ev(int(g0[0]), int(g0[1])))
    e.on_motion(Ev(int(g0[0]) + 40, int(g0[1])))
    e.on_release(Ev(int(g0[0]) + 40, int(g0[1])))
    after = [list(e.doc.shapes[i]["rect_m"][:2]) for i in picked
             if e.doc.shapes[i]["kind"] == "building"]
    moved = [i for i, (u, v) in enumerate(zip(before, after)) if abs(u[0] - v[0]) > 1]
    ok(len(moved) == len(before) and len(before) >= 4,
       f"перетаскивание двигает ВСЮ группу: сдвинулось {len(moved)} из {len(before)}")
    e.do_undo()

    # Shift добавляет и снимает по одной
    e._select([])
    h = e._S((600.0, 1800.0))
    e.on_press(Ev(int(h[0]), int(h[1])))
    e.on_release(Ev(int(h[0]), int(h[1])))
    one = len(e.sel)
    h2 = e._S((660.0, 1800.0))
    e.on_press(SEv(int(h2[0]), int(h2[1])))
    e.on_release(SEv(int(h2[0]), int(h2[1])))
    two = len(e.sel)
    e.on_press(SEv(int(h2[0]), int(h2[1])))
    e.on_release(SEv(int(h2[0]), int(h2[1])))
    ok(one == 1 and two == 2 and len(e.sel) == 1,
       f"Shift добавляет и снимает по одной: {one} -> {two} -> {len(e.sel)}")

    # копирование и вставка — со сдвигом, иначе копия ложится точно поверх
    e._select(picked)
    n0 = len(e.doc.shapes)
    e.copy_selected()
    e.paste_clipboard()
    ok(len(e.doc.shapes) == n0 + len(picked),
       f"вставка добавила {len(e.doc.shapes) - n0} фигур из {len(picked)} скопированных")
    src = e.doc.shapes[picked[0]]
    dst = e.doc.shapes[n0]
    if src["kind"] == "building" and dst["kind"] == "building":
        d = abs(src["rect_m"][0] - dst["rect_m"][0]) + abs(src["rect_m"][1] - dst["rect_m"][1])
        ok(d > 1.0, f"вставка сдвинута от исходной на {d:.0f} м, а не легла точно поверх")
    ok(e.sel == list(range(n0, len(e.doc.shapes))), "после вставки выделены именно копии")
    n1 = len(e.doc.shapes)
    e.delete_selected()
    ok(len(e.doc.shapes) == n1 - len(picked) and not e.sel,
       f"Del удаляет всё выделенное: {n1} -> {len(e.doc.shapes)}")
    e.do_undo()
    e.do_undo()
    e._select([])

    # прилипание: точка садится ТОЧНО на узел соседней фигуры, а не рядом
    e.snap_on.set(True)
    node = e.doc.shapes[0]["points"][0] if e.doc.shapes[0].get("points") else None
    if node:
        near = (node[0] + e._tol_m() * 0.3, node[1] + e._tol_m() * 0.3)
        got, snapped = e._snap(near)
        ok(snapped and abs(got[0] - node[0]) < 1e-6 and abs(got[1] - node[1]) < 1e-6,
           "точка притягивается ровно к узлу соседней фигуры")
        e.snap_on.set(False)
        got2, snapped2 = e._snap(near)
        ok(not snapped2 and abs(got2[0] - near[0]) < 1e-6,
           "с выключенным прилипанием точка остаётся где поставили")
        e.snap_on.set(True)

        # Переправа НЕ цель прилипания. Она ставится САМА, на пересечении дороги с рекой, то
        # есть сидит там, где чаще всего и рисуют, — и пока она была целью, точка липла к мостам
        # вместо того места, куда её ставят. Контроль: верни crossing в _snap — покраснеет.
        far = (node[0] + 400.0, node[1] + 400.0)
        e.doc.shapes.append({"kind": "crossing", "point": [far[0], far[1]]})
        probe = (far[0] + max(e._tol_m(), 4.0) * 0.3, far[1] + max(e._tol_m(), 4.0) * 0.3)
        got3, snapped3 = e._snap(probe)
        ok(not snapped3 and abs(got3[0] - probe[0]) < 1e-6,
           "точка НЕ липнет к переправе — мосты ставятся сами и концом ничему не приходятся")
        e.doc.shapes.pop()

        # В объёме допуск меряется КАМЕРОЙ, а не зумом плана: 8 пикселей у земли это единицы
        # метров, а зум плана давал бы под два десятка при любом удалении камеры
        was3d = e.mode3d.get()
        e.mode3d.set(True)
        e.cam.dist, e.cam.tx, e.cam.ty = 300.0, node[0], node[1]
        e.cam.bounds = tuple(e.doc.size_m)
        e.cam.clamp()
        t3 = e._tol_m_3d((node[0], node[1]))
        ok(t3 < e._tol_m(),
           f"в объёме допуск прилипания от камеры, а не от плана: {t3:.1f} м против "
           f"{e._tol_m():.1f} м у плана")
        e.mode3d.set(was3d)

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
    # Остатки прошлого прогона убираем САМИ: иначе редактор спросит «заменить?» модальным
    # окном, и проверка без человека рядом висит до утра, а не падает.
    for ext in (".vector.json", ".fields.npz", ".map.json", ".height.npz"):
        stale = os.path.join(P.MAPS, NAME + ext)
        if os.path.exists(stale):
            os.remove(stale)
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

    # --- мост ставится сам, когда дорога рисуется через реку
    e._set_type("water")
    e.tool.set("line")
    e.width_m.set(40.0)
    e.dash_on.set(False)
    for x, y in ((520, 120), (540, 760)):
        click(e, x, y)
    e.finish_draft()
    n_cross0 = sum(1 for sh in e.doc.shapes if sh["kind"] == "crossing")
    e._set_type("road")
    e.width_m.set(10.0)
    for x, y in ((150, 430), (900, 450)):
        click(e, x, y)
    e.finish_draft()
    n_cross1 = sum(1 for sh in e.doc.shapes if sh["kind"] == "crossing")
    ok(n_cross1 > n_cross0,
       f"дорога через реку сама получила мост, и по мосту на каждое пересечение "
       f"({n_cross0} -> {n_cross1} переправ)")
    ok(not vectormap.crossing_gaps(e.doc.vec),
       "непокрытых пересечений дороги с рекой не осталось")
    e.do_undo(); e.do_undo()
    e._set_type("forest")

    # --- счётчик кадров
    for _ in range(4):
        e.draw()
    fps_txt = e.canvas.itemcget(e._fps_item, "text") if e._fps_item else ""
    ok("к/с" in fps_txt and "мс" in fps_txt, f"счётчик кадров показывает цену отрисовки: {fps_txt}")
    e.show_fps.set(False)
    e.draw()
    ok(e._fps_item is None, "галка снимает счётчик с холста")
    e.show_fps.set(True)
    e.draw()

    # --- вкладки: вид отдельно от рисования, и срез грунта не выключается
    tabs = [e.nb.tab(t, "text") for t in e.nb.tabs()]
    ok(tabs == ["рисование", "вид", "файл"],
       "вид вынесен во вкладку между рисованием и файлом: " + " | ".join(tabs))
    ok(not hasattr(e, "show_ground"),
       "срез грунта не выключается — переменной под галку больше нет")
    src = io.open(ed.__file__, encoding="utf-8").read()
    ground_off = re.findall(r"(?<![a-z])ground=(?!True)\w+", src)
    ok(not ground_off,
       "срез грунта уходит в отрисовку жёстко включённым: " + str(ground_off))

    # --- тумблеры вместо квадратов с крестиками
    switches = []

    def walk(w):
        for c in w.winfo_children():
            if isinstance(c, ed.Switch):
                switches.append(c)
            walk(c)

    walk(e)
    ok(len(switches) >= 8, f"переключатели нарисованы тумблерами ({len(switches)} шт.)")
    boxes = []

    def walk_box(w):
        for c in w.winfo_children():
            if isinstance(c, ttk.Checkbutton):
                boxes.append(c)
            walk_box(c)

    walk_box(e)
    ok(not boxes, f"квадратов с крестиком в панели не осталось ({len(boxes)})")
    sw = [c for c in switches if c.var is e.show_fps][0]

    def knob(w):
        """Бегунок рисуется последним — по его левому краю и видно состояние."""
        for _ in range(40):
            w._step()
        return w.cv.coords(w.cv.find_all()[-1])[0]

    e.show_fps.set(True)
    x_on = knob(sw)
    e.show_fps.set(False)
    x_off = knob(sw)
    ok(x_on - x_off > 8,
       f"бегунок ездит между краями: выкл {x_off:.0f} px, вкл {x_on:.0f} px")
    was = e.show_fps.get()
    sw.toggle()
    ok(e.show_fps.get() != was, "щелчок по тумблеру переключает саму переменную")
    ok(abs(knob(sw) - x_on) < 1.0, "и бегунок уезжает следом за щелчком")
    e.show_fps.set(False)
    ok(abs(knob(sw) - x_off) < 1.0,
       "тумблер идёт за переменной, даже если её поставили мимо мыши")
    e.show_fps.set(True)
    e.draw()

    # --- камера как в варгейме: точка под курсором не должна уезжать
    import tiles as tilemod  # noqa: E402
    war = ed.view3d.Camera(target=(1275.0, 1275.0), dist=3000.0, yaw=25.0,
                           bounds=(2550.0, 2550.0), auto=True)
    war.tz = 20.0
    war.clamp()
    p0 = war.ground_at(700, 240, (1000, 700), war.tz)
    war.zoom_at(0.5, 700, 240, (1000, 700), war.tz)
    p1 = war.ground_at(700, 240, (1000, 700), war.tz)
    ok(p0 and p1 and math.hypot(p0[0] - p1[0], p0[1] - p1[1]) < 0.5,
       f"приближение к курсору не уводит точку (ушла на {math.hypot(p0[0]-p1[0], p0[1]-p1[1]):.2f} м)")
    grab = war.ground_at(400, 500, (1000, 700), war.tz)
    war.drag_to(grab, 620, 430, (1000, 700), war.tz)
    now = war.ground_at(620, 430, (1000, 700), war.tz)
    ok(math.hypot(grab[0] - now[0], grab[1] - now[1]) < 0.5,
       "перетаскивание держит схваченную землю под курсором")
    ok(war.auto_pitch() < ed.view3d.Camera(dist=9000.0).auto_pitch(),
       f"наклон идёт за удалением: вблизи {war.auto_pitch():.0f}°, издали "
       f"{ed.view3d.Camera(dist=9000.0).auto_pitch():.0f}°")

    # --- уровни подробности: вдали крупная клетка, вблизи мелкая, и число тайлов ограничено
    grid = tilemod.TileGrid(e.doc.vec, e.doc.size_m)
    far = ed.view3d.Camera(target=(1275.0, 1275.0), dist=9000.0, yaw=20.0, auto=True)
    near = ed.view3d.Camera(target=(1275.0, 1275.0), dist=500.0, yaw=20.0, auto=True)
    far.clamp()
    near.clamp()
    lv_far = {k[0] for k in grid.select(far, (1000, 700))}
    sel_near = grid.select(near, (1000, 700))
    lv_near = {k[0] for k in sel_near}
    # Раньше здесь стояло min(lv_near) == 0 — проверка держала не свойство, а КОНСТАНТУ
    # тогдашнего дна. Когда дробление разрешили ниже нуля, она покраснела на верном поведении.
    # Свойство же простое: вблизи мельче, чем издали, и спуск уходит ниже базового уровня.
    ok(min(lv_near) < min(lv_far) and min(lv_near) < 0,
       f"подробность растёт при подлёте: издали уровни {sorted(lv_far)}, вблизи {sorted(lv_near)}")
    ok(min(lv_near) >= tilemod.MIN_LEVEL,
       f"спуск не проваливается глубже предела {tilemod.MIN_LEVEL}: вблизи {min(lv_near)}")
    ok(len(sel_near) <= tilemod.MAX_TILES + 4,
       f"число кусков в кадре ограничено: {len(sel_near)}")
    # Контроль на снятое дно: у земли (сто метров до неё) дробление обязано дойти до предела и
    # дать десятые доли метра на точку. Верни в select() условие level > 0 — покраснеет.
    low = ed.view3d.Camera(target=(1275.0, 1275.0), dist=120.0, yaw=20.0, auto=True)
    low.clamp()
    sel_low = grid.select(low, (1000, 700))
    lv_low = min(k[0] for k in sel_low)
    mpp = grid.span(lv_low) / tilemod.PAINT_PX
    ok(lv_low == tilemod.MIN_LEVEL and mpp < 0.1,
       f"у земли дробление доходит до предела: уровень {lv_low}, {mpp:.3f} м на точку, "
       f"кусков {len(sel_low)}")
    key = sel_near[0]
    grid.set_mode("cells")
    cells_tile = grid.build_now(key)
    x0, y0, span = grid.rect(key)
    n_cells = grid.tile_cells + 2 * tilemod.MARGIN
    ok(cells_tile.shape[:2] == (n_cells, n_cells),
       f"кусок посчитан с запасом под мипмапы: {cells_tile.shape[0]} клеток на {grid.tile_cells}")
    c = grid.cell(key[0])
    direct = vectormap.surface_window(e.doc.vec, c, x0 + 5 * c, y0 + 5 * c, c, c)[0, 0]
    got = cells_tile[tilemod.MARGIN + 5, tilemod.MARGIN + 5].astype(float)
    near = min(ed.view3d.COLORS, key=lambda k: abs(np.asarray(ed.view3d.COLORS[k]) - got).sum())
    ok(near == int(direct), "клетка куска совпадает с посчитанной напрямую из вектора")

    # --- векторная картинка: та же местность, но краем фигуры, а не клеткой
    grid.set_mode("vector")
    vec_tile = grid.build_now(key)
    n_px = tilemod.PAINT_PX + 2 * tilemod.PAINT_MARGIN
    ok(vec_tile.shape[:2] == (n_px, n_px) and vec_tile.dtype == np.uint8,
       f"векторная картинка куска: {vec_tile.shape[0]} точек на {span:.0f} м "
       f"({span / tilemod.PAINT_PX:.2f} м на точку)")
    grid.stop()

    # Дорога в восемь метров должна быть шириной в восемь метров, а не в клетку. Это и есть вся
    # разница: клеточная картинка округляет её до сетки, векторная рисует как нарисовано.
    import paint as painter  # noqa: E402
    road = [{"kind": "line", "type": "road", "width_m": 8.0, "points": [[0, 100], [200, 100]]}]
    W = 200.0
    pic = painter.paint(road, 0.0, 0.0, W, W, 160, 160)          # 1.25 м на точку
    col = np.asarray(ed.view3d.COLORS[4], dtype=float)
    band_px = int((np.abs(pic[:, 80].astype(float) - col).sum(axis=1) < 60).sum())
    cellw = vectormap.surface_window({"shapes": road}, 5.0, 0.0, 0.0, W, W)
    band_cells = int((cellw[16, :] == 4).sum())
    ok(abs(band_px * W / 160 - 8.0) < 1.6,
       f"вектором дорога 8 м вышла шириной {band_px * W / 160:.1f} м; "
       f"клетками — {band_cells * 5.0:.0f} м")

    # --- рисование В ОБЪЁМЕ: луч должен попадать в рельеф, а не в горизонталь
    e.mode3d.set(True)
    e.cam.yaw = 30.0
    e._toggle_3d()
    root.update()
    e.cam.dist = 1800.0
    e.cam.clamp()
    e.draw_3d()
    hits, errs = 0, []
    for px, py in ((520, 300), (350, 480), (700, 380), (600, 250)):
        p = e.pick_ground(px, py)
        if p is None:
            continue
        hits += 1
        cellh = e._height_cell()
        hh = e.doc.height_m(cellh)
        u = min(max(p[0] / cellh - 0.5, 0.0), hh.shape[0] - 1.0)
        v = min(max(p[1] / cellh - 0.5, 0.0), hh.shape[1] - 1.0)
        i0, j0 = int(u), int(v)
        i1, j1 = min(i0 + 1, hh.shape[0] - 1), min(j0 + 1, hh.shape[1] - 1)
        fu, fv = u - i0, v - j0
        z = ((hh[i0, j0] * (1 - fu) + hh[i1, j0] * fu) * (1 - fv)
             + (hh[i0, j1] * (1 - fu) + hh[i1, j1] * fu) * fv)
        sx_, sy_, _ = view3d.project(np.array([p[0]]), np.array([p[1]]),
                                     np.array([z * view3d.VSCALE]), e.cam, (e.W, e.H))
        errs.append(math.hypot(float(sx_[0]) - px, float(sy_[0]) - py))
    ok(hits == 4 and max(errs) < 1.0,
       f"щелчок в объёме попадает в местность: {hits} из 4, промах до {max(errs):.2f} px")

    e.tool.set("relief_poly")
    e._tool_changed()
    e._set_relief(45)
    e.relief_slope.set(60.0)
    before = float(e.doc.height_m(15.0).max())
    hver, ver = e.doc.hversion, e.doc.version
    # Точки берём ПРОЕКЦИЕЙ известной земли, а не фиксированными пикселями. Раньше здесь стоял
    # список экранных координат, подобранный под тогдашнюю случайную карту: стоило генератору
    # начать ставить дома другого размера — поток случайных чисел сдвинулся, карта вышла иной,
    # и один из четырёх щелчков ушёл в небо. Проверка краснела на исправном коде.
    quad_m = [(e.cam.tx - 120.0, e.cam.ty - 120.0), (e.cam.tx + 120.0, e.cam.ty - 120.0),
              (e.cam.tx + 120.0, e.cam.ty + 120.0), (e.cam.tx - 120.0, e.cam.ty + 120.0)]
    quad_scr, quad_vis = e._project_ground(quad_m)
    for (qx, qy), seen_ in zip(quad_scr, quad_vis):
        if seen_:
            e.on_press(Ev(int(qx), int(qy)))
    ok(e._draft is not None and len(e._draft) == 4,
       f"точки фигуры ставятся прямо в объёме ({0 if e._draft is None else len(e._draft)} из 4)")

    # Чертёж должен быть ВИДЕН и ТЯНУТЬСЯ ЗА КУРСОРОМ: без нити рисование выглядит мёртвым —
    # поставил точку и до следующего щелчка на экране ничего не меняется.
    e._hover = None
    bare = np.asarray(e._draft_overlay_3d(Image.new("RGB", (e.W, e.H), (40, 44, 40))),
                      dtype=np.int16)
    e.on_hover(Ev(300, 620))
    pulled = np.asarray(e._draft_overlay_3d(Image.new("RGB", (e.W, e.H), (40, 44, 40))),
                        dtype=np.int16)
    seen = int((np.abs(bare - np.array([40, 44, 40])).max(axis=2) > 20).sum())
    ok(seen > 1500, f"незакрытая фигура видна на кадре: {seen} точек чертежа")
    ok(int((np.abs(pulled - bare).max(axis=2) > 20).sum()) > 200,
       "чертёж тянется за курсором: наведение меняет кадр, не дожидаясь щелчка")
    e._hover = None
    e.finish_draft()
    after = float(e.doc.height_m(15.0).max())
    ok(after > before + 5.0,
       f"печать в объёме подняла местность: было {before:.0f} м, стало {after:.0f} м")
    ok(e.doc.hversion > hver and e.doc.version == ver,
       "правка высоты не трогает версию вектора — картинки кусков не пересчитываются")
    e.do_undo()
    ok(abs(float(e.doc.height_m(15.0).max()) - before) < 0.01,
       "отмена возвращает карту высот на место")
    # закрытие фигуры правой кнопкой — как на плане, а не поворот камеры
    e.tool.set("relief_poly")
    e._tool_changed()
    for px, py in ((430, 390), (690, 360), (740, 500)):
        e.on_press(Ev(px, py))
    e.on_right(Ev(740, 500))
    ok(e._draft is None, "ПКМ в объёме замыкает фигуру, а не крутит камеру")

    # просмотр прострелов в объёме
    e.tool.set("vision")
    e._tool_changed()
    e.vision_r.set(700.0)
    e.on_press(Ev(560, 430))
    ok(e._vision is not None, "просмотр видимости работает в объёме")
    rgba, vcell, vorigin = e._vision_rgba()
    ok(rgba.ndim == 3 and rgba.shape[2] == 4 and vcell > 0 and len(vorigin) == 2,
       f"маска видимости готова к наложению на местность: {rgba.shape[1]}x{rgba.shape[0]} "
       f"по {vcell:.0f} м")
    e.draw_3d()
    e.on_release(Ev(560, 430))
    ok(e._vision is None, "отпустил кнопку — просмотр снялся")

    # дом и мост ставятся прямо в объёме
    n_before = len(e.doc.shapes)
    e.tool.set("building")
    e._tool_changed()
    e.on_press(Ev(520, 420))
    e.on_motion(Ev(600, 470))
    e.on_release(Ev(600, 470))
    house = e.doc.shapes[-1]
    ok(house["kind"] == "building" and abs(house["rect_m"][4]) > 1,
       f"дом ставится в объёме и доворачивается протяжкой ({house['rect_m'][4]:.0f}°)")
    e.tool.set("crossing")
    e._tool_changed()
    e.on_press(Ev(430, 470))
    e.on_motion(Ev(500, 500))
    e.on_release(Ev(500, 500))
    bridge = e.doc.shapes[-1]
    ok(bridge["kind"] == "crossing" and len(e.doc.shapes) == n_before + 2,
       "переправа ставится в объёме")
    e.do_undo()
    e.do_undo()

    # линейка в объёме
    e.tool.set("ruler")
    e._tool_changed()
    e.on_press(Ev(400, 400))
    e.on_motion(Ev(760, 520))
    measured3d = e._ruler is not None and len(e._ruler) == 2
    e.on_release(Ev(760, 520))
    ok(measured3d, "линейка меряет прямо в объёме")
    # и гаснет на отпускании — как на плане: замер, а не объект карты
    ok(e._ruler is None, "линейка в объёме гаснет после отпускания")
    e.cancel_draft()

    e.tool.set("select")
    e._tool_changed()
    e.mode3d.set(False)
    e._toggle_3d()
    root.update()

    # --- дымка в АБСОЛЮТНЫХ метрах: один и тот же кусок земли с удалением камеры бледнеет
    far_cam = ed.view3d.Camera(target=(1275.0, 1275.0), dist=6000.0, yaw=0.0, pitch=35.0)
    near_cam = ed.view3d.Camera(target=(1275.0, 1275.0), dist=1200.0, yaw=0.0, pitch=35.0)
    strip = np.zeros((G, G), dtype=np.int32)
    im_far = np.asarray(view3d.render(strip, cone, 2550.0 / G, far_cam, (300, 200),
                                      ground=False)).astype(float)
    im_near = np.asarray(view3d.render(strip, cone, 2550.0 / G, near_cam, (300, 200),
                                       ground=False)).astype(float)
    sky = np.asarray(ed.view3d.SKY_HORIZON, dtype=float)
    d_far = float(np.abs(im_far[120:160].mean(axis=(0, 1)) - sky).sum())
    d_near = float(np.abs(im_near[120:160].mean(axis=(0, 1)) - sky).sum())
    ok(d_far < d_near,
       f"дымка считается в метрах: издали земля ближе к цвету горизонта ({d_far:.0f} против "
       f"{d_near:.0f})")

    # --- непрерывный полёт: скорость растёт с высотой, после отпускания есть выбег
    e.mode3d.set(True)
    e._toggle_3d()
    root.update()
    e.cam.dist = 3000.0
    e.cam.yaw = 0.0
    e.cam.clamp()
    x0, y0 = e.cam.tx, e.cam.ty
    e._fly_key("f", True)
    t_fly = time.perf_counter()
    while time.perf_counter() - t_fly < 0.7:
        root.update()
        time.sleep(0.01)
    e._fly_key("f", False)
    flew = math.hypot(e.cam.tx - x0, e.cam.ty - y0)
    ok(flew > 300.0, f"камера летит непрерывно: {flew:.0f} м за 0.7 с на удалении 3000 м")
    coast_from = math.hypot(e.cam.tx - x0, e.cam.ty - y0)
    t_fly = time.perf_counter()
    while time.perf_counter() - t_fly < 0.5:
        root.update()
        time.sleep(0.01)
    coast = math.hypot(e.cam.tx - x0, e.cam.ty - y0) - coast_from
    ok(0.0 < coast < flew, f"после отпускания есть выбег и остановка ({coast:.0f} м)")

    # --- приближение доезжает плавно и не уводит точку из-под курсора
    # Замеряем на ТОЙ ЖЕ плоскости, на которой наезд и держит точку: он привязывается к высоте
    # земли в момент первого щелчка. Померить на другой высоте — увидеть сдвиг там, где его нет.
    h0_zoom = e._ground_h()
    p_zoom = e.cam.ground_at(600, 380, (e.W, e.H), h0_zoom)
    before = e.cam.dist
    for _ in range(3):
        e.on_wheel(Ev(600, 380, 120))
    t_zoom = time.perf_counter()
    while e._zoom_goal is not None and time.perf_counter() - t_zoom < 3.0:
        root.update()
        time.sleep(0.015)
    p_zoom2 = e.cam.ground_at(600, 380, (e.W, e.H), h0_zoom)
    # Допуск в долях удаления, а не в метрах: на трёх километрах полтора метра — это доля
    # пикселя, а на подлёте к земле тот же допуск в метрах был бы уже заметным сдвигом.
    ok(abs(e.cam.dist - before * 0.82 ** 3) < before * 0.05
       and math.hypot(p_zoom[0] - p_zoom2[0], p_zoom[1] - p_zoom2[1]) < 0.005 * before,
       f"колесо доезжает плавно до цели: {before:.0f} -> {e.cam.dist:.0f} м (ждём "
       f"{before * 0.82 ** 3:.0f}), точка под курсором ушла на "
       f"{math.hypot(p_zoom[0] - p_zoom2[0], p_zoom[1] - p_zoom2[1]):.2f} м")
    e.mode3d.set(False)
    e._toggle_3d()
    root.update()

    # --- иерархия генератора: у слоёв разные размеры, а размах высот растёт с картой
    import mapgen  # noqa: E402
    small, _ = mapgen.generate(2550.0, 5)
    big, info_big = mapgen.generate(10000.0, 5)

    def spread(doc, key):
        vals = [round(sh["width_m"]) for sh in doc["shapes"] if sh.get("type") == key
                and sh["kind"] == "line"]
        return len(set(vals))

    def relief_span(doc):
        h = doc["height"]["h"]
        return float(h.max() - h.min())

    ok(relief_span(big) > 2.0 * relief_span(small),
       f"размах высот растёт с размером карты: {relief_span(small):.0f} м на 2.5 км против "
       f"{relief_span(big):.0f} м на 10 км")
    ok(spread(big, "road") >= 3,
       f"дороги разных классов: {spread(big, 'road')} разных ширин на театре")
    houses = sum(1 for sh in big["shapes"] if sh["kind"] == "building")
    ok(info_big["towns"] >= 1 and houses / 100.0 > 1.5,
       f"на театре есть город и {houses / 100.0:.1f} здания на км² "
       f"(города {info_big['towns']}, сёла {info_big['villages']}, хутора {info_big['hamlets']})")
    radii = []
    for sh in big["shapes"]:
        if sh.get("type") == "forest" and sh["kind"] == "polygon":
            pts = np.asarray(sh["points"], dtype=float)
            radii.append(float(np.hypot(*(pts - pts.mean(axis=0)).T).mean()))
    ok(radii and max(radii) > 3.0 * min(radii),
       f"лес разных размеров: от {min(radii):.0f} до {max(radii):.0f} м")

    # --- срез грунта обязан совпасть с краем карты: местность рисуется до Gx*клетка, а срез
    # раньше строился по последнему УЗЛУ высоты и был короче на клетку — земля свисала над ним
    import view3d_gpu as _gpu  # noqa: E402
    gcell = 2550.0 / cone.shape[0]
    gv, _gi = _gpu._ground_mesh(cone, gcell)
    ok(abs(float(gv[:, 0].max()) - cone.shape[0] * gcell) < 1e-3
       and abs(float(gv[:, 1].max()) - cone.shape[1] * gcell) < 1e-3,
       f"срез грунта доходит ровно до края карты ({float(gv[:, 0].max()):.0f} м из "
       f"{cone.shape[0] * gcell:.0f})")

    # --- правка в объёме видна СРАЗУ, без переключения вида
    # Ошибка была тихая: тайл честно пересчитывался, но видеокарта держала картинки по
    # ключу «вид + номер квадрата», ключ после правки не менялся, и она оставляла прежнюю.
    # Нарисованное появлялось только после смены стиля, когда ключ менялся целиком.
    e = app.frame if not isinstance(app.frame, ed.StartScreen) else None
    # Пропуск блока делаем ГРОМКИМ. Молчаливый уже подводил: набор печатал «провалов нет», не
    # выполнив двадцати проверок, и заметить это можно было только сверив их ЧИСЛО.
    if e is None or e._gl is None:
        ok(False, "блок правки в объёме пропущен: редактор=%s, рисовальщик=%s"
                  % (e is not None, e._gl is not None if e is not None else None))
    if e is not None and e._gl is not None:
        frames = []
        blit = e._blit
        e._blit = lambda img: (frames.append(np.asarray(img.convert("RGB"), dtype=np.int16)),
                               blit(img))[1]
        e.mode3d.set(True)
        e._toggle_3d()
        e.cam.tx, e.cam.ty = P.ARENA_M * 0.5, P.ARENA_M * 0.5
        e.cam.dist, e.cam.yaw = 1200.0, 0.0
        e.cam.clamp()

        def settle(sec=6.0):
            """Дождаться, пока досчитаются все куски под камерой. ВОЗВРАЩАЕТ, дождались ли.

            Возврат важен: раньше она по истечении срока просто выходила молча, и снятый после
            неё кадр мог быть недосчитанным — с кусками-предками вместо своих. Сравнение таких
            кадров давало расхождение, неотличимое от настоящей ошибки в сбросе округи. Тест,
            который тихо сдаётся, ничего не проверяет, поэтому теперь о сдаче сообщают вслух."""
            t0 = time.time()
            done = False
            while time.time() - t0 < sec:
                root.update()
                e.draw()
                keys = e._tiles.select(e.cam, (e.W, e.H))
                if keys and all(e._tiles.get(k) is not None for k in keys):
                    done = True
                    break
                time.sleep(0.05)
            root.update()
            e.draw()
            root.update()
            return done

        settle()
        was = frames[-1]
        cx, cy, r = e.cam.tx, e.cam.ty - 180.0, 220.0
        e.push_undo()
        e.doc.shapes.append({"kind": "polygon", "type": "forest",
                             "points": [[cx - r, cy - r], [cx + r, cy - r],
                                        [cx + r, cy + r], [cx - r, cy + r]]})
        e._changed(dirty=(cx - r, cy - r, cx + r, cy + r))
        settle()
        now = frames[-1]
        hh, ww, _ = now.shape
        sl = (slice(int(hh * 0.45), int(hh * 0.78)), slice(int(ww * 0.35), int(ww * 0.65)))
        green_was = float(was[sl][:, :, 1].mean() - was[sl][:, :, 0].mean())
        green_now = float(now[sl][:, :, 1].mean() - now[sl][:, :, 0].mean())
        ok(green_now - green_was > 3.0,
           f"нарисованное в объёме появляется само, без переключения вида: зелень перед "
           f"камерой {green_was:.0f} -> {green_now:.0f}")

        # Наложения в объёме: обводка выделенного и ручки узлов. Пока их не было, в объёме
        # нельзя было ни увидеть, что выделено, ни найти узел, — и править там было нечего.
        idx = len(e.doc.shapes) - 1                  # только что добавленный лес перед камерой
        plain = frames[-1].astype(np.int16)
        e._select([idx])
        e.tool.set("select")
        e.draw()
        root.update()
        outlined = frames[-1].astype(np.int16)
        d_out = int((np.abs(outlined - plain).max(axis=2) > 24).sum())
        ok(d_out > 200,
           f"в объёме видно обводку выделенной фигуры: изменилось {d_out} точек кадра")

        e.tool.set("nodes")
        e.draw()
        root.update()
        handled = frames[-1].astype(np.int16)
        d_nodes = int((np.abs(handled - outlined).max(axis=2) > 24).sum())
        ok(d_nodes > 100,
           f"под инструментом «узлы» в объёме видны ручки узлов: ещё {d_nodes} точек")

        # Контроль: узлы кладутся НА РЕЛЬЕФ. Поднимем землю под фигурой — ручки обязаны уехать
        # по экрану вслед за ней, иначе они нарисованы по плоскости и на склоне соврут.
        #
        # Камеру задаём ЯВНО и сбоку. Первая попытка этого не делала, брала ту, что осталась от
        # предыдущей проверки, — а при взгляде почти сверху подъём земли двигает точку по экрану
        # почти никуда, и проверка падала на верном коде. Высоту тоже задаём через панель: её
        # значение _stamp_relief подставляет поверх того, что передали в фигуре.
        cam_was = (e.cam.tx, e.cam.ty, e.cam.dist, e.cam.auto)
        h_was = e.relief_h.get()
        e.relief_h.set(120.0)
        e.cam.tx, e.cam.ty, e.cam.dist, e.cam.auto = cx, cy - 600.0, 900.0, True
        e.cam.clamp()
        scr_before, _ = e._project_ground(e.doc.shapes[idx]["points"])
        e.push_undo()
        e._stamp_relief({"kind": "polygon", "type": "relief",
                         "points": [[cx - r * 2, cy - r * 2], [cx + r * 2, cy - r * 2],
                                    [cx + r * 2, cy + r * 2], [cx - r * 2, cy + r * 2]]})
        scr_after, _ = e._project_ground(e.doc.shapes[idx]["points"])
        moved = max(abs(a[1] - b[1]) for a, b in zip(scr_before, scr_after))
        ok(moved > 8.0,
           f"узлы в объёме лежат на рельефе: подъём земли на 120 м при взгляде сбоку сдвинул "
           f"их по экрану на {moved:.0f} px")
        e.do_undo()
        e.relief_h.set(h_was)

        # --- правка ПРЯМО В ОБЪЁМЕ: выбор, перенос пачки, узел
        # Камеру держим сбоку и близко, чтобы фигура была крупной: попадание считается по
        # настоящим экранным координатам, а не по условным.
        e.cam.tx, e.cam.ty, e.cam.dist, e.cam.auto = cx, cy - 700.0, 1100.0, True
        e.cam.clamp()
        e.draw_3d()
        root.update()

        def scr_of(world_pt):
            pts, vis = e._project_ground([world_pt])
            return int(pts[0][0]), int(pts[0][1]), bool(vis[0])

        e.tool.set("select")
        e._select([])
        sx, sy, seen = scr_of([cx, cy])
        e.on_press(Ev(sx, sy))
        ok(e.sel == [idx] and e._drag is not None and e._drag[0] == "move",
           f"щелчок в объёме выделяет фигуру и берёт её на перенос (sel={e.sel}, "
           f"drag={e._drag[0] if e._drag else None})")

        # Перенос обязан идти по СМЕЩЕНИЮ ЗЕМЛИ под курсором, а не по пикселям: при наклонённой
        # камере одинаковый сдвиг по экрану — это разный сдвиг по земле, и фигура уползала бы
        was_pts = [list(q) for q in e.doc.shapes[idx]["points"]]
        tx2, ty2, _ = scr_of([cx + 250.0, cy + 120.0])
        g_from = e.pick_ground(sx, sy)
        g_to = e.pick_ground(tx2, ty2)
        e.on_motion(Ev(tx2, ty2))
        now_pts = e.doc.shapes[idx]["points"]
        dx_ = now_pts[0][0] - was_pts[0][0]
        dy_ = now_pts[0][1] - was_pts[0][1]
        err = math.hypot(dx_ - (g_to[0] - g_from[0]), dy_ - (g_to[1] - g_from[1]))
        ok(math.hypot(dx_, dy_) > 50.0 and err < 1.0,
           f"перенос в объёме идёт по земле под курсором: уехала на {math.hypot(dx_, dy_):.0f} м, "
           f"расхождение с землёй {err:.2f} м")
        e.on_release(Ev(tx2, ty2))
        ok(e._drag is None, "после отпускания в объёме протяжка закрыта")

        # Узел ищется ПО ЭКРАНУ. Через землю нельзя: замер показал промах в сто метров, когда
        # узел закрыт бугром — луч честно упирается в бугор, а ручка нарисована поверх него
        e.tool.set("nodes")
        node0 = list(e.doc.shapes[idx]["points"][0])
        nx_, ny_, _ = scr_of(node0)
        e.on_press(Ev(nx_, ny_))
        ok(e._drag is not None and e._drag[0] == "node",
           f"под инструментом «узлы» щелчок по ручке берёт узел "
           f"(drag={e._drag[0] if e._drag else None})")
        if e._drag and e._drag[0] == "node":
            gx_, gy_, _ = scr_of([node0[0] + 150.0, node0[1] + 70.0])
            e.on_motion(Ev(gx_, gy_))
            got = e.doc.shapes[idx]["points"][0]
            ok(math.hypot(got[0] - node0[0], got[1] - node0[1]) > 50.0,
               f"узел уехал за курсором на "
               f"{math.hypot(got[0] - node0[0], got[1] - node0[1]):.0f} м")
            e.on_release(Ev(gx_, gy_))

        # Обводка обязана лежать на земле ВСЕЙ длиной, а не только вершинами. Пока сажались
        # одни вершины, экранный отрезок между ними шёл напрямик, и сторона через бугор
        # прочерчивалась сквозь него: замер на линии в километр через холм 120 м дал отход
        # до 116 пикселей. Это не косметика — ПКМ ставит узел, целясь в НАРИСОВАННУЮ линию.
        long_pts = [[cx - 500.0, cy], [cx + 500.0, cy]]

        def seg_dist(pt, s0, s1):
            vx, vy = s1[0] - s0[0], s1[1] - s0[1]
            L2 = vx * vx + vy * vy
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((pt[0] - s0[0]) * vx
                                                         + (pt[1] - s0[1]) * vy) / L2))
            return math.hypot(s0[0] + t * vx - pt[0], s0[1] + t * vy - pt[1])

        ends_scr, _ = e._project_ground(long_pts)
        dense_pts, _ = e._drape(long_pts, False)
        dense_scr, _ = e._project_ground(dense_pts)
        off_straight = max(seg_dist(q, ends_scr[0], ends_scr[1]) for q in dense_scr)
        off_draped = max(min(seg_dist(q, dense_scr[i], dense_scr[i + 1])
                             for i in range(len(dense_scr) - 1)) for q in dense_scr)
        ok(off_draped < 1.0,
           f"обводка в объёме лежит на земле всей длиной: дроблёная отходит на "
           f"{off_draped:.1f} px, прямая между вершинами — на {off_straight:.0f} px")

        # Рамка в объёме считается ПО ЭКРАНУ: прямоугольник на экране захватывает по земле
        # трапецию до горизонта, и что в неё попало — на глаз не предскажешь. По экрану же
        # обводят ровно то, что видят. Простая ЛКМ по пустому месту оставлена камере: тащить
        # землю — самое частое движение в объёме, и отбирать у него кнопку нельзя.
        e.tool.set("select")
        e._select([])
        bx0, by0 = int(e.W * 0.15), int(e.H * 0.15)
        bx1, by1 = int(e.W * 0.85), int(e.H * 0.85)
        e.on_press(Ev(bx0, by0, state=0x0001))
        started = e._drag is not None and e._drag[0] == "box3d"
        e.on_motion(Ev(bx1, by1, state=0x0001))
        e.on_release(Ev(bx1, by1, state=0x0001))
        missed = 0
        for si in e.sel:
            bb = e._shape_screen_bbox(e.doc.shapes[si])
            if bb is None or bb[2] < bx0 or bb[0] > bx1 or bb[3] < by0 or bb[1] > by1:
                missed += 1
        ok(started and len(e.sel) > 1 and missed == 0 and e._box3d is None,
           f"Shift+рамка в объёме обводит пачку по экрану: {len(e.sel)} фигур, мимо {missed}")
        e._select([])
        e._drag = None
        e._cam_drag = None
        e.on_press(Ev(bx0, by0))
        ok(e._drag is None and e._cam_drag is not None,
           "без Shift ЛКМ по пустому месту тащит землю, а не рамку")
        e._cam_drag = None
        e._select([idx])
        e.tool.set("nodes")          # вернуть инструмент: ниже проверяется ПКМ по узлам

        # ПКМ по ГРАНИЦЕ добавляет узел. Без этого изогнуть готовый контур было нечем: узлы
        # можно было только двигать и удалять, а добавить — никак, и фигуру перерисовывали
        # целиком. Новый узел садится РОВНО на ребро, поэтому контур от вставки не едет.
        pts_now = e.doc.shapes[idx]["points"]
        n_before = len(pts_now)

        def poly_area(ps):
            acc = 0.0
            for q in range(len(ps)):
                x1, y1 = ps[q]
                x2, y2 = ps[(q + 1) % len(ps)]
                acc += x1 * y2 - x2 * y1
            return abs(acc) / 2.0

        area_before = poly_area(pts_now)
        a_, b_ = pts_now[0], pts_now[1]
        mid_ = [(a_[0] + b_[0]) / 2.0, (a_[1] + b_[1]) / 2.0]
        mx_, my_, _ = scr_of(mid_)
        e.on_right(Ev(mx_, my_))
        pts_now = e.doc.shapes[idx]["points"]
        near = min(math.hypot(q[0] - mid_[0], q[1] - mid_[1]) for q in pts_now)
        ok(len(pts_now) == n_before + 1 and near < 20.0,
           f"ПКМ по границе в объёме добавляет узел ({n_before} -> {len(pts_now)}), "
           f"и он садится на неё ({near:.1f} м от места нажатия)")
        ok(abs(poly_area(pts_now) - area_before) < max(1.0, area_before * 1e-4),
           f"вставка узла не двигает контур: площадь {poly_area(pts_now):.0f} против "
           f"{area_before:.0f} м²")

        # ПКМ по ручке по-прежнему удаляет, а мимо фигуры — отдаёт кнопку камере
        n_before = len(pts_now)
        hx_, hy_, _ = scr_of(pts_now[0])
        e.on_right(Ev(hx_, hy_))
        ok(len(e.doc.shapes[idx]["points"]) == n_before - 1,
           f"ПКМ по ручке узел удаляет ({n_before} -> {len(e.doc.shapes[idx]['points'])})")
        n_before = len(e.doc.shapes[idx]["points"])
        e._cam_drag = None
        e.on_right(Ev(12, 8))
        ok(len(e.doc.shapes[idx]["points"]) == n_before and e._cam_drag is not None,
           "ПКМ мимо фигуры в объёме крутит камеру, а не правит узлы")
        e.do_undo()
        e.do_undo()
        e.do_undo()
        e.do_undo()
        e.tool.set("select")
        e._select([])

        e.cam.tx, e.cam.ty, e.cam.dist, e.cam.auto = cam_was
        e.cam.clamp()
        e.tool.set("select")
        e._select([])

        e.do_undo()
        # Срок больше прежних двух секунд: к этому месту набор успевает наделать правок, а
        # пересчёт кусков после протяжки приходит одним залпом на отпускании — во время самой
        # протяжки их не трогают. Проверка про переключение стиля, а не про то, за сколько
        # досчитаются куски; если не досчитались — скажем об этом прямо, а не покажем «не готовы».
        done_style = settle(20.0)

        # Переключение стиля НЕ должно грубить картинку. Куски лежат по стилям, поэтому прежний
        # стиль подменяет новый на время счёта — той же подробности, а не растянутым верхним
        # уровнем (вся карта в 512 точек). Возврат к посчитанному стилю — мгновенный.
        keys = e._tiles.select(e.cam, (e.W, e.H))
        ok(keys and all(e._tiles.get(k) is not None for k in keys),
           f"перед переключением все куски готовы ({len(keys)})"
           + ("" if done_style else "  [НЕ ДОСЧИТАЛОСЬ за 20 с — проверка ниже про стиль, "
                                    "а не про скорость счёта]"))
        was_style = e.map_style.get()
        e.map_style.set("cells")
        e.draw()
        keys = e._tiles.select(e.cam, (e.W, e.H))
        subs = [k for k in keys if e._tiles.get(k) is None and e._tiles.other_style(k)[0]]
        deep = [k for k in keys if e._tiles.get(k) is None
                and (e._tiles.ready_ancestor(k)[0] or (99, 0, 0))[0] - k[0] >= 2]
        ok(len(subs) == len(deep) and len(subs) > 0,
           f"на переключении стиля куски подменяются СВОИМ квадратом в прежнем стиле "
           f"({len(subs)} из {len(keys)}), а не растянутым верхним уровнем")
        done_back = settle(20.0)
        e.map_style.set(was_style)          # назад к тому, что уже посчитано
        e.draw()
        keys = e._tiles.select(e.cam, (e.W, e.H))
        ready = sum(1 for k in keys if e._tiles.get(k) is not None)
        ok(ready == len(keys),
           f"возврат к уже посчитанному стилю мгновенный: готовы все {ready} кусков"
           + ("" if done_back else "  [НЕ ДОСЧИТАЛОСЬ за 20 с]"))

        # Печать рельефа в топостиле трогает и горизонтали, поэтому куски сбрасываются по
        # ЗАДЕТОЙ ОКРУГЕ, а не все разом. Проверяем, что округа взята не на глазок: то же
        # место, пересчитанное частично и целиком, должно дать одну картинку.
        e.map_style.set("topo")
        settle()
        cx, cy, r = e.cam.tx, e.cam.ty - 200.0, 240.0
        n0 = e._tiles.built
        e.push_undo()
        e._stamp_relief({"kind": "polygon", "type": "relief", "h_m": 40.0,
                         "points": [[cx - r, cy - r], [cx + r, cy - r],
                                    [cx + r, cy + r], [cx - r, cy + r]]})
        done_part = settle(20.0)
        n_part = e._tiles.built - n0
        part = frames[-1]
        n0 = e._tiles.built
        e._tiles.set_source(e.doc.vec, -999, None)      # сброс всего до единого куска
        e._tiles.set_source(e.doc.vec, (e.doc.version, e.doc.hversion), None)
        # Срок больше прежних шести секунд: после углубления уровней полный сброс пересчитывает
        # не полсотни кусков, а под полторы сотни, и на неудачной машине шести секунд не хватало
        done_full = settle(20.0)
        whole = frames[-1]
        n_full = e._tiles.built - n0
        diff = np.abs(part - whole)
        bad = int((diff.max(axis=2) > 8).sum())
        ok(done_part and done_full and float(diff.mean()) < 0.5 and bad < diff.size / 3000,
           f"печать рельефа сбрасывает только задетую округу и картинка от этого не меняется: "
           f"{n_part} кусков против {n_full} при полном сбросе, расхождение "
           f"{float(diff.mean()):.3f} из 255, спорных точек {bad}"
           + ("" if done_part and done_full else
              "  [НЕ ДОСЧИТАЛОСЬ: частично %s, целиком %s — кадр снят недостроенным, "
              "расхождение об ошибке в сбросе НЕ говорит]"
              % ("да" if done_part else "НЕТ", "да" if done_full else "НЕТ")))
        e.do_undo()
        e.map_style.set("vector")
        settle(2.0)
        e._blit = blit
        e.mode3d.set(False)
        e._toggle_3d()

    # --- объём на видеокарте, если она есть. Нет — это не провал: программный вид остаётся
    try:
        gl = view3d_gpu.GLView((400, 300), samples=4)
        gl.set_height(cone, 2550.0 / G, key="check")
        gl.upload_tile("flat", view3d.surface_rgb(flat).transpose(1, 0, 2).astype(np.uint8))
        im_gl = np.asarray(gl.frame(side, [(0.0, 0.0, 2550.0, 2550.0, 0.0, 0.0, 1.0, 1.0,
                                            "flat", 48)])).astype(float)
        ok(im_gl.shape == (300, 400, 3) and im_gl.std() > 8,
           f"видеокарта рисует объём ({gl.renderer})")
        # Главная проверка на два рисовальщика: НЕСИММЕТРИЧНАЯ карта должна выйти одинаковой.
        # Зеркало по Y, поворот текстуры, обратный знак поворота камеры — всё это молчаливые
        # ошибки, каждая из которых уже случалась, и на глаз читается не как ошибка, а как
        # «почему-то другая карта».
        mark = np.zeros((G, G), dtype=np.int32)
        mark[:G // 3, :G // 2] = 1                   # лес в одном углу — чтобы ловить зеркало
        mark[-G // 4:, -G // 3:] = 3                 # и вода в противоположном
        gl.upload_tile("asym", view3d.surface_rgb(mark).transpose(1, 0, 2).astype(np.uint8))
        cell_a = 2550.0 / G
        whole = [(0.0, 0.0, G * cell_a, G * cell_a, 0.0, 0.0, 1.0, 1.0, "asym", 64)]
        a = np.asarray(gl.frame(side, whole)).astype(float)
        b = np.asarray(view3d.render(mark, cone, cell_a, side, (400, 300), ss=2)).astype(float)
        ok(float(np.abs(a - b).mean()) < 8.0,
           f"видеокарта и программный вид дают одну картинку "
           f"(расхождение {float(np.abs(a - b).mean()):.1f} из 255)")

        # ДОМА КОРОБКАМИ — та же проверка, но со строениями в кадре. Геометрия у обоих
        # рисовальщиков общая (view3d.building_faces) нарочно: разойдись она хоть освещением
        # стены, объём с видеокартой и без неё показывал бы разные сёла.
        # Камера СВОЯ и близкая, а место — ровное. Первая попытка ставила дома в центре карты,
        # где на этой сетке холм 80 м: коробки в 9 м оказались внутри него, да ещё и с 5200 м,
        # где дом меньше пикселя. Проверка показывала «домов не видно» на исправном коде.
        near = ed.view3d.Camera(target=(500.0, 500.0), dist=700.0, yaw=0.0, pitch=25.0)
        boxes = []
        for k in range(3):
            boxes.append((380.0 + k * 120.0, 500.0, 60.0, 36.0, 15.0 * k,
                          0.0, view3d.building_height_m(60.0, 36.0)))
        a1 = np.asarray(gl.frame(near, whole)).astype(float)
        gl.set_buildings(boxes, key=("проба", 1))
        a2 = np.asarray(gl.frame(near, whole)).astype(float)
        b2 = np.asarray(view3d.render(mark, cone, cell_a, near, (400, 300), ss=2,
                                      buildings=boxes)).astype(float)
        moved = int((np.abs(a2 - a1).max(axis=2) > 24).sum())
        ok(moved > 300, f"дома коробками видно в кадре: изменилось {moved} точек")
        ok(float(np.abs(a2 - b2).mean()) < 8.0,
           f"видеокарта и программный вид одинаково ставят дома "
           f"(расхождение {float(np.abs(a2 - b2).mean()):.1f} из 255)")
        # высота показная и считается от УЗКОЙ стороны: длинный сарай не становится башней
        ok(abs(view3d.building_height_m(60.0, 12.0)
               - view3d.building_height_m(12.0, 12.0)) < 1e-6,
           "высота дома берётся от узкой стороны, а не от длины")
        # Зона видимости обязана красить и КОРОБКИ, а не только землю. Пока наложение висело
        # только на шейдере местности, дом стоял ярким посреди затенённого поля — и это
        # читалось как «строение зону не перекрывает», хотя за ним ноль. Дом ставится
        # ОТДЕЛЬНО и крупно, чтобы полоса замера гарантированно пришлась на стены.
        one = [(500.0, 500.0, 120.0, 80.0, 0.0, 0.0, view3d.building_height_m(120.0, 80.0))]
        gl.set_buildings(one, key=("проба", 2))
        h_plain = np.asarray(gl.frame(near, whole)).astype(float)
        tint = np.zeros((64, 64, 4), dtype=np.uint8)
        tint[..., 2] = 255
        tint[..., 3] = 220
        gl.set_overlay(tint, 2550.0 / 64, origin=(0.0, 0.0))
        h_shaded = np.asarray(gl.frame(near, whole)).astype(float)
        band = (slice(120, 190), slice(150, 250))
        blue0 = float(h_plain[band][..., 2].mean() - h_plain[band][..., 0].mean())
        blue1 = float(h_shaded[band][..., 2].mean() - h_shaded[band][..., 0].mean())
        ok(blue1 > blue0 + 20.0,
           f"зона видимости красит и коробки домов: синева {blue0:.0f} -> {blue1:.0f}")
        gl.clear_overlay()
        gl.set_buildings([], key=("проба", 0))

        # то же самое, но КУСКАМИ: местность, собранная из четырёх кусков, обязана совпасть с
        # цельной. Щели и швы между уровнями подробности ловятся ровно здесь
        parts = []
        half = G // 2
        for i in (0, 1):
            for j in (0, 1):
                sub = mark[i * half:(i + 1) * half, j * half:(j + 1) * half]
                gl.upload_tile(("part", i, j),
                               view3d.surface_rgb(sub).transpose(1, 0, 2).astype(np.uint8))
                parts.append((i * half * cell_a, j * half * cell_a, half * cell_a, half * cell_a,
                              0.0, 0.0, 1.0, 1.0, ("part", i, j), 32))
        c = np.asarray(gl.frame(side, parts)).astype(float)
        ok(float(np.abs(a - c).mean()) < 3.0,
           f"местность кусками совпадает с цельной, швов нет "
           f"(расхождение {float(np.abs(a - c).mean()):.2f} из 255)")
    except Exception as ex:
        print(f"  --  видеокарта недоступна, объём остаётся программным ({ex})")

    # --- выход в главное меню: несохранённое должно спрашивать
    e = app.frame
    if not isinstance(e, ed.StartScreen):
        asked = []
        real = ed.messagebox.askyesno
        ed.messagebox.askyesno = lambda *a, **k: (asked.append(a), False)[1]
        e.push_undo()
        e.doc.shapes.append({"kind": "polygon", "type": "forest",
                             "points": [[10, 10], [60, 10], [60, 60]]})
        e._changed()
        e.back_to_menu()
        ok(asked and app.frame is e,
           "выход в меню с несохранённой правкой спрашивает и по «нет» остаётся в редакторе")
        ed.messagebox.askyesno = lambda *a, **k: True
        e.back_to_menu()
        ok(isinstance(app.frame, ed.StartScreen), "по «да» уходит в главное меню")
        ed.messagebox.askyesno = real

    app.show_start()
    root.update()
    ok(isinstance(app.frame, ed.StartScreen), "«в главное меню» возвращает на стартовый экран")

    for p in (NAME + ".vector.json", NAME + ".fields.npz", NAME + ".map.json",
              os.path.join("preview", NAME + ".png")):
        f = os.path.join(P.MAPS, p)
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(sc_path):
        os.remove(sc_path)

    # --- объём В ОКНЕ: своё окно со своим редактором, чтобы не мешать проверкам выше
    ed._gl_widget_class = real_gl_widget          # виджету — свои проверки, с настоящим классом
    try:
        w_root = tk.Toplevel(root)
        w_root.geometry("1000x760")
        w_app = ed.App(w_root)
        w_root.update()
        w_app.show_editor(ed.Doc(vectormap.new_doc((P.ARENA_M, P.ARENA_M)), P.CELL_M, "окно"))
        w_root.update()
        we = w_app.frame
        ok(we._glw is not None, "виджет с настоящим GL-контекстом заводится")
        if we._glw is not None:
            we.fit_view()
            we.mode3d.set(True)
            we._toggle_3d()
            w_root.update()
            ok(we._in_widget and we._gl is not None and getattr(we._gl, "to_screen", False),
               "в объёме кадр идёт ПРЯМО В ОКНО, без чтения обратно в память")
            for _ in range(6):
                we.draw_3d()
                w_root.update()
            ok(True, "кадры в окне рисуются без ошибок")
            we.mode3d.set(False)
            we._toggle_3d()
            w_root.update()
            ok(not we._in_widget, "на плане возвращается холст")
            we.mode3d.set(True)
            we._toggle_3d()
            we.draw_3d()
            w_root.update()
            ok(we._in_widget and we._gl is not None,
               "повторный вход в объём работает: контекст виджета пересоздаётся")

            # Ориентация кадра в окне проверяется ОТДЕЛЬНЫМ прогоном:
            #     py -3.9 editor/check_gl_window.py
            # Здесь нельзя: к этому месту живут сразу несколько контекстов OpenGL, и чтение
            # кадра из окна перестаёт быть надёжным — читается пустота, хотя нарисовано верно.
            # Примета такой беды приметная: расхождение с образцом и с его зеркалом выходит
            # ОДИНАКОВЫМ, потому что перевернуть однотонное поле нельзя.
        w_root.destroy()
    except Exception as ex:                                  # noqa: BLE001
        ok(False, "объём в окне: %s: %s" % (type(ex).__name__, ex))

    root.destroy()
    print("\nПРОВАЛОВ: " + (str(len(FAILED)) + " — " + "; ".join(FAILED) if FAILED else "нет"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
