"""Тот же объёмный вид, но на видеокарте, и с кусочной местностью.

Почему это возможно в tkinter, у которого нет GL-холста: мы не встраиваем контекст в окно, а
рисуем ЗАКАДРОВО и кладём готовую картинку на обычный холст. Тк остаётся показывалкой картинок,
как и был, — а рисование забирает видеокарта.

Замер на этой машине (RTX 4060, боевая карта 170x170, окно 1100x700):

    программно, coarse=1 ss=2 со срезом грунта   273 мс
    видеокартой, полная подробность, MSAA x8       1.1 мс + показ

Как устроена местность. Геометрия — это ОДНА единичная сетка, растянутая по куску карты
уравнением в вершинном шейдере, а высота берётся выборкой из общего поля высот (текстура). Из
этого следует главное: сколько бы уровней подробности ни было, все они читают высоту из одного
места, поэтому щелей на стыках не бывает в принципе. Тайлу остаётся только своя картинка
местности — сетка типов в его клетке.

Вид (цвета, солнце, дымка, преувеличение высот, срез грунта) берётся ИЗ view3d — иначе два
рисовальщика разойдутся, и одна и та же карта будет выглядеть по-разному в зависимости от того,
нашлась видеокарта или нет.
"""
import math

import numpy as np

import view3d
from view3d import COLORS, FOG, GROUND_COLOR, GROUND_DEEPEN, GROUND_ROWS, GROUND_SHARE, \
    SKY_HORIZON, SKY_TOP, SUN, VSCALE

VERT = """
#version 330
uniform mat4 mvp;
uniform vec4 patch;        // x0, y0, сторона по x и по y в метрах (у края карты обрезана)
uniform vec4 texrect;      // u0, v0, du, dv — окно в картинке местности
uniform vec4 huv;          // масштаб и сдвиг мира в координаты поля высот
uniform float hcell;       // клетка поля высот в метрах
uniform sampler2D hmap;
in vec2 in_uv;
out vec3 v_norm;
out vec2 v_uv;
out vec2 v_world;
out float v_depth;

float ground(vec2 world) {
    return textureLod(hmap, world * huv.xy + huv.zw, 0.0).r;
}

void main() {
    vec2 world = patch.xy + in_uv * patch.zw;
    float z = ground(world);
    // Нормаль считаем по соседям в ПОЛЕ ВЫСОТ, а не по своей сетке: тогда освещение одинаково
    // на всех уровнях подробности и не мигает при их смене.
    float hx = ground(world + vec2(hcell, 0.0)) - ground(world - vec2(hcell, 0.0));
    float hy = ground(world + vec2(0.0, hcell)) - ground(world - vec2(0.0, hcell));
    v_norm = normalize(vec3(-hx / (2.0 * hcell), -hy / (2.0 * hcell), 1.0 / VS));
    v_uv = texrect.xy + in_uv * texrect.zw;
    v_world = world;
    vec4 p = mvp * vec4(world, z * VS, 1.0);
    gl_Position = p;
    v_depth = p.w;
}
""".replace("VS", "%.4f" % VSCALE)

FRAG = """
#version 330
uniform sampler2D tex;
uniform vec3 sun;
uniform vec3 fog_color;
uniform float fog_far;
uniform float fog_amount;
uniform sampler2D over;    // наложение: зона видимости и прочее плоское поверх местности
uniform vec4 over_uv;      // масштаб и сдвиг мира в координаты наложения
uniform float over_on;
in vec3 v_norm;
in vec2 v_uv;
in vec2 v_world;
in float v_depth;
out vec4 f_color;
void main() {
    vec3 base = texture(tex, v_uv).rgb;
    if (over_on > 0.5) {
        // Наложение ложится по МИРОВЫМ координатам, а не по кускам: у каждого куска своя
        // картинка местности со своим окном, и вешать на неё общую маску было бы нечем.
        vec2 ouv = v_world * over_uv.xy + over_uv.zw;
        if (ouv.x >= 0.0 && ouv.x <= 1.0 && ouv.y >= 0.0 && ouv.y <= 1.0) {
            vec4 o = texture(over, ouv);
            base = mix(base, o.rgb, o.a);
        }
    }
    // тот же зажим, что в программной отмывке: теневой склон должен читаться, а не проваливаться
    float lam = clamp(dot(normalize(v_norm), sun), 0.58, 1.34);
    // дымка в метрах: fog_far — настоящая дальность видимости, одна и та же на любом удалении
    float fog = pow(clamp(v_depth / fog_far, 0.0, 1.0), FOGP) * fog_amount;
    f_color = vec4(mix(base * lam, fog_color, fog), 1.0);
}
""".replace("FOGP", "%.3f" % view3d.FOG_POW)

# срез грунта: свет уже вложен в цвет вершины (стенка плоская, считать на пиксель нечего)
VERT_SOLID = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
out vec2 v_world;
out float v_depth;
void main() {
    vec4 p = mvp * vec4(in_pos, 1.0);
    gl_Position = p;
    v_color = in_color;
    v_world = in_pos.xy;
    v_depth = p.w;
}
"""

FRAG_SOLID = """
#version 330
uniform vec3 fog_color;
uniform float fog_far;
uniform float fog_amount;
uniform sampler2D over;    // то же наложение, что на местности: зона видимости
uniform vec4 over_uv;
uniform float over_on;     // срезу грунта наложение ни к чему, домам — нужно
in vec3 v_color;
in vec2 v_world;
in float v_depth;
out vec4 f_color;
void main() {
    vec3 col = v_color;
    // Дом обязан затеняться зоной видимости наравне с землёй под ним. Пока наложение висело
    // только на шейдере местности, коробка стояла ЯРКОЙ посреди затенённого поля, и читалось
    // это как «строение зону не перекрывает» — хотя перекрывает: за ним ноль.
    if (over_on > 0.5) {
        vec2 ouv = v_world * over_uv.xy + over_uv.zw;
        if (ouv.x >= 0.0 && ouv.x <= 1.0 && ouv.y >= 0.0 && ouv.y <= 1.0) {
            vec4 o = texture(over, ouv);
            col = mix(col, o.rgb, o.a);
        }
    }
    float fog = pow(clamp(v_depth / fog_far, 0.0, 1.0), FOGP) * fog_amount;
    f_color = vec4(mix(col, fog_color, fog), 1.0);
}
""".replace("FOGP", "%.3f" % view3d.FOG_POW)

# небо — градиент во весь кадр, тот же, что в программном виде
VERT_SKY = """
#version 330
in vec2 in_pos;
out float v_t;
void main() {
    gl_Position = vec4(in_pos, 0.999999, 1.0);
    v_t = in_pos.y;
}
"""

FRAG_SKY = """
#version 330
uniform vec3 top;
uniform vec3 horizon;
in float v_t;
out vec4 f_color;
// v_t берётся с минусом: вертикаль в матрице перевёрнута, чтобы снимок читался сверху вниз, и
// небо без этого встаёт вверх ногами — светлый горизонт оказывается наверху, тёмный зенит внизу
void main() { f_color = vec4(mix(horizon, top, clamp(-v_t * 0.5 + 0.5, 0.0, 1.0)), 1.0); }
"""


def _mvp(cam, size):
    """Матрица мира-в-экран. Камера та же, что в view3d.project: фокус равен ширине кадра —
    иначе на видеокарте карта выглядит мельче, и переключение рисовальщика видно как скачок.
    Вертикаль перевёрнута нарочно: тогда снимок читается сразу сверху вниз, без переворота."""
    w, h = size
    yaw, pitch = math.radians(cam.yaw), math.radians(cam.pitch)
    # Знак поворота — как в view3d.project (там мир вращается на -yaw). Со знаком наоборот
    # картинка получается зеркальной, и при переключении рисовальщика карта прыгает.
    tz = cam.tz * VSCALE                 # точка, вокруг которой ходит камера, лежит на земле
    eye = np.array([cam.tx - cam.dist * math.cos(pitch) * math.sin(yaw),
                    cam.ty - cam.dist * math.cos(pitch) * math.cos(yaw),
                    tz + cam.dist * math.sin(pitch)], dtype=np.float32)
    target = np.array([cam.tx, cam.ty, tz], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.eye(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ eye
    near, far = 2.0, cam.dist * 4.0 + 40000.0
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0
    proj[1, 1] = -2.0 * w / h
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1.0
    return np.ascontiguousarray((proj @ view).T, dtype="f4")   # OpenGL ждёт по столбцам


def _grid_mesh(n):
    """Единичная сетка n x n клеток в координатах 0..1. Одна на все куски карты: в мир её
    переводит вершинный шейдер, поэтому буфер заводится по разу на каждую подробность, а не на
    каждый тайл."""
    ix, iy = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    verts = np.stack([ix / n, iy / n], axis=-1).astype("f4")
    a = np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]
    b = a + (n + 1)
    idx = np.stack([a, b, b + 1, a, b + 1, a + 1], axis=-1).astype("i4").ravel()
    return verts.reshape(-1, 2), idx


def _ground_mesh(height_m, cell_m):
    """Срез грунта: те же четыре стенки, что в view3d, но геометрией, а не заливкой. Цвет кладём
    в вершины — видеокарта растянет его плавно, и крап получается без квадратов."""
    Gx, Gy = height_m.shape
    hs = np.asarray(height_m, dtype=np.float32) * VSCALE
    hbase = float(hs.min())
    # Кромка среза идёт до Gx*cell, а не до последнего УЗЛА высоты: местность нарисована на всю
    # карту, и последняя её клетка берёт высоту крайнего узла (текстура зажата по краю). Считать
    # по узлам значило укоротить блок на клетку — на театре это сорок метров, и вблизи земля
    # заметно свисала над срезом.
    W, H = Gx * cell_m, Gy * cell_m
    extent = max(W, H)
    depth = float(np.clip(GROUND_SHARE * extent, 180.0, 900.0)) * VSCALE
    base = np.asarray(GROUND_COLOR, dtype=np.float32) / 255.0
    xs = np.append(np.arange(Gx) * cell_m, W).astype(np.float32)
    ys = np.append(np.arange(Gy) * cell_m, H).astype(np.float32)
    verts, idx, off = [], [], 0
    edges = ((xs, np.zeros(Gx + 1, "f4"), np.append(hs[:, 0], hs[-1, 0]), (0.0, -1.0)),
             (xs, np.full(Gx + 1, H, "f4"), np.append(hs[:, -1], hs[-1, -1]), (0.0, 1.0)),
             (np.zeros(Gy + 1, "f4"), ys, np.append(hs[0, :], hs[0, -1]), (-1.0, 0.0)),
             (np.full(Gy + 1, W, "f4"), ys, np.append(hs[-1, :], hs[-1, -1]), (1.0, 0.0)))
    for X, Y, H_, n in edges:
        top = np.asarray(H_, dtype=np.float32)
        lam = float(np.clip(0.88 + 0.45 * (n[0] * SUN[0] + n[1] * SUN[1]), 0.62, 1.30))
        along = (X + Y).astype(np.float32)
        N = len(X)
        bottom = np.full(N, hbase - depth, dtype=np.float32)
        for r in range(GROUND_ROWS + 1):
            t = r / GROUND_ROWS
            z = top + (bottom - top) * t
            tint = view3d._earth_tint(along, np.full(N, t * depth / VSCALE, dtype=np.float32))
            col = base[None, :] * (lam * (1.0 - GROUND_DEEPEN * t)) * tint[:, None]
            verts.append(np.concatenate([np.stack([X, Y, z], axis=-1), col], axis=-1))
        for r in range(GROUND_ROWS):
            a = off + r * N + np.arange(N - 1)
            b = a + N
            idx.append(np.stack([a, b, b + 1, a, b + 1, a + 1], axis=-1).ravel())
        off += (GROUND_ROWS + 1) * N
    return np.concatenate(verts).astype("f4"), np.concatenate(idx).astype("i4")


SPARE_TEXTURES = 48        # столько освободившихся текстур держим про запас на каждый размер:
#                            хватает на churn при полёте (за кадр меняется единицы), а 48 штук
#                            по 1.2 МБ — это около 58 МБ памяти видеокарты


class GLView:
    """Живой рисовальщик: контекст, буферы, поле высот и картинки тайлов держатся между кадрами.

    Ради этого класс и существует. Создание контекста стоит 400 мс, поле высот — ещё десяток, и
    если делать это на каждый кадр (как делал первый набросок), от выигрыша видеокарты не
    остаётся ничего: 40 мс вместо одного."""

    def __init__(self, size=(1100, 700), samples=8):
        import moderngl
        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.prog_solid = self.ctx.program(vertex_shader=VERT_SOLID, fragment_shader=FRAG_SOLID)
        self.prog_sky = self.ctx.program(vertex_shader=VERT_SKY, fragment_shader=FRAG_SKY)
        self.prog_sky["top"].value = tuple(float(v) for v in SKY_TOP / 255.0)
        self.prog_sky["horizon"].value = tuple(float(v) for v in SKY_HORIZON / 255.0)
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self.vao_sky = self.ctx.vertex_array(
            self.prog_sky, [(self.ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self.samples = int(samples)
        self.renderer = self.ctx.info.get("GL_RENDERER", "?")
        self._size = None
        self._grids = {}                 # число делений -> (vao, число индексов)
        self._tiles = {}                 # ключ тайла -> текстура местности
        self._spare = {}                 # размер -> запас освободившихся текстур, см. upload_tile
        self._vao_bld = None             # коробки домов одной сеткой
        self._bld_key = None
        self._hkey = None
        self._hmap = None
        self._over = None
        self._over_uv = (0.0, 0.0, 0.0, 0.0)
        self._vao_ground = None
        self.resize(size)

    # --- буферы

    def resize(self, size):
        size = (max(8, int(size[0])), max(8, int(size[1])))
        if size == self._size:
            return
        self._size = size
        self.msaa = self.ctx.framebuffer(self.ctx.renderbuffer(size, samples=self.samples),
                                         self.ctx.depth_renderbuffer(size, samples=self.samples))
        self.plain = self.ctx.framebuffer(self.ctx.renderbuffer(size),
                                          self.ctx.depth_renderbuffer(size))

    def _grid(self, n):
        got = self._grids.get(n)
        if got is None:
            verts, idx = _grid_mesh(n)
            vbo = self.ctx.buffer(verts.tobytes())
            ibo = self.ctx.buffer(idx.tobytes())
            got = (self.ctx.vertex_array(self.prog, [(vbo, "2f", "in_uv")], ibo), len(idx))
            self._grids[n] = got
        return got[0]

    def set_height(self, height_m, cell_m, key):
        """Поле высот — общее для всей карты и для всех уровней подробности. Отсюда берут высоту
        и геометрия, и нормали, поэтому уровни сходятся на стыках точно."""
        if key == self._hkey:
            return
        self._hkey = key
        h = np.ascontiguousarray(np.asarray(height_m, dtype="f4").T)   # строки — это y
        Gy, Gx = h.shape
        if self._hmap is not None:
            self._hmap.release()
        self._hmap = self.ctx.texture((Gx, Gy), 1, h.tobytes(), dtype="f4")
        self._hmap.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        self._hmap.repeat_x = self._hmap.repeat_y = False
        self.h_cell = float(cell_m)
        self.h_uv = (1.0 / (cell_m * Gx), 1.0 / (cell_m * Gy), 0.5 / Gx, 0.5 / Gy)

        gv, gi = _ground_mesh(np.asarray(height_m, dtype=np.float32), cell_m)
        self._vao_ground = self.ctx.vertex_array(
            self.prog_solid, [(self.ctx.buffer(gv.tobytes()), "3f 3f", "in_pos", "in_color")],
            self.ctx.buffer(gi.tobytes()))

    def set_buildings(self, boxes, key=None):
        """Коробки домов одной сеткой. Пересобираем только при смене карты: строений на театре
        сотни, а меняются они правкой, а не кадром.

        Геометрию берём из view3d.building_faces — общую с программным рисовальщиком, иначе
        сверка их кадров в check_gui разошлась бы на первом же доме."""
        if key is not None and key == self._bld_key:
            return
        self._bld_key = key
        if self._vao_bld is not None:
            self._vao_bld.release()
            self._vao_bld = None
        if not boxes:
            return
        verts, idx, off = [], [], 0
        for b in boxes:
            for quad, col in view3d.building_faces(*b):
                c = np.repeat((np.asarray(col, dtype=np.float32) / 255.0)[None, :], 4, axis=0)
                verts.append(np.hstack([quad, c]).astype("f4"))
                idx.append(np.array([off, off + 1, off + 2, off, off + 2, off + 3], dtype="i4"))
                off += 4
        v = np.concatenate(verts)
        i = np.concatenate(idx)
        self._vao_bld = self.ctx.vertex_array(
            self.prog_solid, [(self.ctx.buffer(v.tobytes()), "3f 3f", "in_pos", "in_color")],
            self.ctx.buffer(i.tobytes()))

    # --- картинки тайлов

    def has_tile(self, key):
        return key in self._tiles

    def upload_tile(self, key, rgb):
        """Картинка куска местности -> текстура. rgb — (py, px, 3), строка 0 это низ куска.

        Мипмапы обязательны: без них дальний кусок, у которого на точку картинки приходится
        меньше пикселя экрана, шипит зерном при каждом движении камеры.

        Текстура берётся ИЗ ЗАПАСА, а не заводится заново. Разница не косметическая: завести
        новую и уничтожить прежнюю стоит 2.06 мс на кусок, а перезаписать готовую — 0.35 мс
        (замер под нагрузкой, когда в кадре 163 куска и часть заменяется). Ошибка была тихой:
        поодиночке создание текстуры стоит те же 0.35 мс, и дорогим оно становится только при
        потоке создание-уничтожение, то есть ровно при полёте у земли. При четырёх новых кусках
        за кадр набегало под семь миллисекунд — заметная доля кадра.

        Место в запасе считается ПО РАЗМЕРУ: векторная картинка 544x544, клеточная 136x136,
        и перезаписать одну другой нельзя."""
        img = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        size = (img.shape[1], img.shape[0])
        old = self._tiles.pop(key, None)
        if old is not None:
            self._retire(old)
        spare = self._spare.get(size)
        if spare:
            tex = spare.pop()
            tex.write(img)
        else:
            tex = self.ctx.texture(size, 3, img.tobytes())
            tex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            tex.anisotropy = 16.0
            tex.repeat_x = tex.repeat_y = False
        tex.build_mipmaps()
        self._tiles[key] = tex

    def _retire(self, tex):
        """Освободившуюся текстуру — в запас, а не в утиль. Запас ограничен: держать их без счёта
        значит не отдавать память видеокарты вовсе, а картинка куска весит около 1.2 МБ."""
        size = tuple(tex.size)
        spare = self._spare.setdefault(size, [])
        if len(spare) < SPARE_TEXTURES:
            spare.append(tex)
        else:
            tex.release()

    def set_overlay(self, rgba, cell_m, origin=(0.0, 0.0)):
        """Картинка поверх всей местности: зона видимости, а дальше что угодно плоское.

        rgba — (py, px, 4), строка 0 это y=0. Держим ОДНУ на карту: маска общая, а куски у
        каждого уровня свои, и раздавать её по кускам было бы нечем."""
        img = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8))
        if self._over is not None:
            self._over.release()
        self._over = self.ctx.texture((img.shape[1], img.shape[0]), 4, img.tobytes())
        self._over.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        self._over.repeat_x = self._over.repeat_y = False
        gy, gx = img.shape[0], img.shape[1]
        # наложение лежит не на всю карту, а на своё окно: видимость считается вокруг точки
        self._over_uv = (1.0 / (cell_m * gx), 1.0 / (cell_m * gy),
                         0.5 / gx - origin[0] / (cell_m * gx),
                         0.5 / gy - origin[1] / (cell_m * gy))

    def clear_overlay(self):
        if self._over is not None:
            self._over.release()
            self._over = None

    def keep_tiles(self, keys, limit=400):
        """Освободить картинки тайлов, которых нет в списке. Не сразу: пока их меньше предела,
        держим все — при облёте камера возвращается на прежние места, и перезаливать одно и то же
        дороже, чем помнить. Картинка тайла — 67 КБ с мипмапами, четыреста штук это 27 МБ."""
        if len(self._tiles) <= limit:
            return
        keep = set(keys)
        for key in [k for k in self._tiles if k not in keep]:
            self._retire(self._tiles.pop(key))

    # --- кадр

    def frame(self, cam, draws, size=None, ground=True):
        """draws — список (x0, y0, сторона_x, сторона_y, u0, v0, du, dv, ключ_картинки, делений).

        Стороны разные, потому что у края карты тайл обрезан: без обрезки местность свисает за
        срез грунта губой в десяток метров."""
        from PIL import Image
        if size is not None:
            self.resize(size)
        mvp = _mvp(cam, self._size)
        far = view3d.fog_far(cam.dist)       # дальность видимости, а не «сколько влезло в кадр»
        for p in (self.prog, self.prog_solid):
            p["mvp"].write(mvp)
            p["fog_color"].value = tuple(float(v) for v in SKY_HORIZON / 255.0)
            p["fog_far"].value = far
            p["fog_amount"].value = float(FOG)
        self.msaa.use()
        self.ctx.clear(*(float(v) for v in SKY_HORIZON / 255.0))
        self.ctx.disable(self.mgl.DEPTH_TEST)
        self.vao_sky.render()
        self.ctx.enable(self.mgl.DEPTH_TEST)
        # Срез грунта наложением не красим: он ниже местности и к зоне видимости не относится
        self.prog_solid["over_on"].value = 0.0
        if ground and self._vao_ground is not None:
            self._vao_ground.render()
        if self._vao_bld is not None:
            if self._over is not None:
                self._over.use(2)
                self.prog_solid["over"].value = 2
                self.prog_solid["over_uv"].value = self._over_uv
                self.prog_solid["over_on"].value = 1.0
            self._vao_bld.render()
            self.prog_solid["over_on"].value = 0.0
        if self._hmap is not None and draws:
            self._hmap.use(1)
            self.prog["hmap"].value = 1
            self.prog["tex"].value = 0
            self.prog["over_on"].value = 1.0 if self._over is not None else 0.0
            if self._over is not None:
                self._over.use(2)
                self.prog["over"].value = 2
                self.prog["over_uv"].value = self._over_uv
            self.prog["sun"].value = tuple(float(v) for v in SUN)
            self.prog["huv"].value = self.h_uv
            self.prog["hcell"].value = self.h_cell
            for x0, y0, spx, spy, u0, v0, du, dv, key, segs in draws:
                tex = self._tiles.get(key)
                if tex is None:
                    continue
                tex.use(0)
                self.prog["patch"].value = (float(x0), float(y0), float(spx), float(spy))
                self.prog["texrect"].value = (float(u0), float(v0), float(du), float(dv))
                self._grid(segs).render()
        self.ctx.copy_framebuffer(self.plain, self.msaa)
        return Image.frombytes("RGB", self._size, self.plain.read(components=3))


def render(surface, height_m, cell_m, cam, size=(1400, 900), ground=True, samples=8):
    """Разовый кадр в файл: вся карта одним куском, без уровней подробности. Для окна редактора
    так делать нельзя — держите GLView и кормите его тайлами."""
    gl = GLView(size, samples=samples)
    gl.set_height(height_m, cell_m, key="once")
    gl.upload_tile("all", view3d.surface_rgb(surface).transpose(1, 0, 2).astype(np.uint8))
    Gx, Gy = surface.shape
    segs = int(np.clip(max(Gx, Gy), 8, 256))
    draws = [(0.0, 0.0, Gx * cell_m, Gy * cell_m, 0.0, 0.0, 1.0, 1.0, "all", segs)]
    return gl.frame(cam, draws, ground=ground)
