"""Объёмный вид карты: поверхность рельефа под свободной камерой.

Зачем он в редакторе, который рисует сверху. Высота на плане не читается: отмывка и горизонтали
говорят «здесь выше», но не отвечают на вопрос, ради которого рельеф вообще заводился, — что
откуда видно. Поставить камеру на скат и посмотреть глазами — отвечает сразу.

Рисуем программно, без видеокарты, поэтому платим разрешением: при вращении сетка грубее, на
отпущенной мыши — полная. Замер на этой машине: 42x42 клетки — 22 мс на кадр, 85x85 — 66 мс,
170x170 — 282 мс. Отсюда и LOD.

Это ПРОСМОТР, а не редактор: рисование остаётся на плане, здесь только камера.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

# Цвета те же, что на плане (play.py TERRAIN_COLORS) — иначе одна и та же местность выглядит
# по-разному в двух окнах, и глаз перестаёт им верить.
COLORS = {0: (92, 96, 62), 1: (26, 58, 30), 2: (78, 76, 84), 3: (28, 52, 88), 4: (140, 124, 74),
          5: (92, 112, 116)}
SUN = np.array([-0.55, 0.35, 0.76], dtype=np.float32)      # направление на солнце
#   Свет ОБЯЗАН падать слева-сверху (северо-запад). При свете снизу глаз читает отмывку
#   наизнанку — известная иллюзия: холмы становятся ямами. Так печатают все карты.
SKY_TOP = np.array([26, 31, 44], dtype=np.float32)         # небо: сверху темнее, к горизонту светлее
SKY_HORIZON = np.array([74, 82, 96], dtype=np.float32)
# ДЫМКА В АБСОЛЮТНЫХ МЕТРАХ. Раньше она считалась от глубины кадра: дальний край выцветал
# одинаково — что он в двух километрах, что в десяти. Из-за этого расстояние на глаз не читалось
# вовсе, и карта 10 км ощущалась такой же, как 2.5. Теперь дымка привязана к настоящей дальности
# видимости: на пяти километрах местность заметно бледнеет, на пятнадцати сливается с горизонтом.
VISIBILITY_M = 14000.0     # дальность, на которой местность полностью уходит в дымку


def fog_far(cam_dist):
    """Дальность дымки для кадра.

    Вблизи она АБСОЛЮТНАЯ — только так по выцветанию читается расстояние. Но когда камеру
    отводят, чтобы посмотреть карту целиком, честная дымка закрашивает её целиком: с девятнадцати
    километров десятикилометровый театр превращается в серый лист. Поэтому дальше определённого
    удаления дымка отпускается — это тот же приём, что в варгеймах: на оперативном виде воздуха
    нет, он мешает читать обстановку."""
    return max(VISIBILITY_M, 2.0 * float(cam_dist))
FOG = 0.90                 # предельная доля дымки на этой дальности
FOG_POW = 1.0              # набор дымки с расстоянием; крутее единицы забивает и близкое
VSCALE = 1.8               # ПОКАЗНОЕ преувеличение высот. Сорок метров на два с половиной
#                            километра — уклон в полтора градуса, глазом это плоскость. Карты
#                            рельефа рисуют с преувеличением по той же причине. На бой не влияет:
#                            логика читает поле высот, а не картинку.


# Срез грунта под картой. Карта висела в пустоте, и низ у неё был обрывом в никуда: глазу не за
# что зацепиться, чтобы понять, где ноль высоты и насколько глубока лощина. Срез даёт опору.
# На логику не влияет вовсе, это подставка.
# Земля ОДНА, без слоёв: полосатый блок читался как учебник геологии и спорил с картой за
# внимание. Объём даёт крап и потемнение книзу, а не смена материала.
GROUND_COLOR = (118, 92, 62)
GROUND_ROWS = 12           # на сколько рядов бьётся стенка: рядами идёт потемнение и крап
GROUND_SHARE = 0.09        # какую долю от стороны карты занимает подставка: на карте 10 км
#                            двести метров — щель, её просто не видно
GROUND_DEEPEN = 0.50       # насколько темнее низ среза, чем верх
_NOISE = np.random.default_rng(20240).random((64, 64)).astype(np.float32)


def _earth_tint(along_m, depth_m):
    """Крап земли: два масштаба шума. Считается от МИРОВЫХ координат, а не от номера
    четырёхугольника, — иначе при смене подробности (LOD на вращении) крап пересыпался бы, и
    стенка мигала на каждое движение мыши."""
    n = (0.55 * _NOISE[(along_m / 130.0).astype(np.int32) % 64, (depth_m / 55.0).astype(np.int32) % 64]
         + 0.45 * _NOISE[(along_m / 24.0).astype(np.int32) % 64, (depth_m / 13.0).astype(np.int32) % 64])
    # Размах нарочно небольшой: мельче четырёхугольника крап всё равно не станет (он и есть
    # «пиксель» этой отрисовки), а на большом размахе стенка рассыпается в шашечки.
    return 0.92 + 0.15 * n


def _ground_quads(xs, ys, hh, cam, size):
    """Четырёхугольники боковых стенок подставки: по кромке карты, ряд за рядом вниз.

    Верхний ряд повторяет рельеф, нижний лежит плоско, между ними переход — так срез не выглядит
    штампованной плитой под холмом."""
    hbase = float(hh.min()) * VSCALE
    extent = max(float(xs.max()), float(ys.max()))
    depth = float(np.clip(GROUND_SHARE * extent, 180.0, 900.0)) * VSCALE
    base = np.asarray(GROUND_COLOR, dtype=np.float32)
    out = []
    edges = ((xs[:, 0], ys[:, 0], hh[:, 0], (0.0, -1.0)),
             (xs[:, -1], ys[:, -1], hh[:, -1], (0.0, 1.0)),
             (xs[0, :], ys[0, :], hh[0, :], (-1.0, 0.0)),
             (xs[-1, :], ys[-1, :], hh[-1, :], (1.0, 0.0)))
    for X, Y, H, n in edges:
        top = np.asarray(H, dtype=np.float32) * VSCALE
        bottom = np.full_like(top, hbase - depth)
        # стенка светлеет, если смотрит на солнце: без этого подставка выглядит плоской наклейкой
        lam = float(np.clip(0.88 + 0.45 * (n[0] * SUN[0] + n[1] * SUN[1]), 0.62, 1.30))
        levels = [top + (bottom - top) * (r / GROUND_ROWS) for r in range(GROUND_ROWS + 1)]
        proj = [project(X, Y, lv, cam, size) for lv in levels]
        along = np.asarray(X, dtype=np.float32) + np.asarray(Y, dtype=np.float32)
        mid = 0.5 * (along[:-1] + along[1:])
        for r in range(GROUND_ROWS):
            (ax, ay, az), (bx, by, bz) = proj[r], proj[r + 1]
            qx = np.stack([ax[:-1], ax[1:], bx[1:], bx[:-1]], axis=1)
            qy = np.stack([ay[:-1], ay[1:], by[1:], by[:-1]], axis=1)
            qd = 0.25 * (az[:-1] + az[1:] + bz[1:] + bz[:-1])
            t = (r + 0.5) / GROUND_ROWS
            tint = _earth_tint(mid, np.full_like(mid, t * depth / VSCALE))
            qc = base[None, :] * (lam * (1.0 - GROUND_DEEPEN * t)) * tint[:, None]
            out.append((qx, qy, qc, qd))
    return out


def surface_rgb(surface):
    """Сетка типов -> цвет клетки. Одна на оба рисовальщика: цвет местности не должен зависеть
    от того, нашлась видеокарта или нет.

    Края типов размываем на полклетки: лес обрывается в поле ступенькой ровно по сетке, и именно
    эта лесенка читается как грязь."""
    lut = np.zeros((max(COLORS) + 1, 3), dtype=np.float32)
    for k, c in COLORS.items():
        lut[k] = c
    base = lut[np.clip(surface, 0, max(COLORS))]
    return 0.62 * base + 0.38 * np.stack([_box_blur(base[..., k]) for k in range(3)], axis=-1)


def _box_blur(a):
    """Три-на-три среднее без scipy — им размываются края типов и гасится лесенка сетки."""
    out = a.copy()
    out[1:] += a[:-1]; out[:-1] += a[1:]
    out[:, 1:] += a[:, :-1]; out[:, :-1] += a[:, 1:]
    out[1:, 1:] += a[:-1, :-1]; out[:-1, :-1] += a[1:, 1:]
    out[1:, :-1] += a[:-1, 1:]; out[:-1, 1:] += a[1:, :-1]
    return out / 9.0


# Наклон камеры ведётся ЗА удалением, как в варгеймах: издали смотрим почти сверху (так читается
# обстановка), вблизи — почти с земли (так читается местность). Ручная поправка складывается с
# этим ходом, а не отменяет его.
PITCH_NEAR, PITCH_FAR = 13.0, 62.0
DIST_NEAR, DIST_FAR = 220.0, 9000.0


class Camera:
    """Камера над картой: точка, на которую смотрим, удаление, поворот, наклон.

    Наклон обычно не задаётся руками — он функция удаления (auto=True). Ручное вращение колёсиком
    и перетаскиванием земли считается ЧЕРЕЗ обратную проекцию, а не приращениями углов: только так
    точка под курсором остаётся под курсором, а без этого карта уезжает из-под пальца и большая
    карта не ощущается местностью."""

    def __init__(self, target=(1275.0, 1275.0), dist=3200.0, yaw=35.0, pitch=40.0,
                 bounds=None, auto=False):
        self.tx, self.ty = target
        self.tz = 0.0                    # высота точки, вокруг которой ходит камера, В МЕТРАХ
        self.dist = float(dist)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.bias = 0.0                  # ручная поправка наклона поверх автоматического
        self.auto = bool(auto)           # наклон ведёт себя сам
        self.bounds = bounds             # (w_m, h_m) карты: не даём уехать в пустоту

    # --- углы

    def auto_pitch(self):
        lo, hi = math.log(DIST_NEAR), math.log(DIST_FAR)
        t = (math.log(max(self.dist, 1.0)) - lo) / (hi - lo)
        return PITCH_NEAR + (PITCH_FAR - PITCH_NEAR) * min(max(t, 0.0), 1.0)

    def clamp(self):
        if self.auto:
            self.pitch = self.auto_pitch() + self.bias
        self.pitch = float(np.clip(self.pitch, 6.0, 88.0))
        self.dist = float(np.clip(self.dist, 60.0, 60000.0))
        self.yaw = (self.yaw + 180.0) % 360.0 - 180.0
        if self.bounds:                  # цель держим над картой с небольшим полем
            w, h = self.bounds
            self.tx = float(np.clip(self.tx, -0.15 * w, 1.15 * w))
            self.ty = float(np.clip(self.ty, -0.15 * h, 1.15 * h))

    # --- обратная проекция

    def ground_at(self, sx, sy, size, h0=0.0):
        """Точка карты под пикселем экрана, на высоте h0 метров. None — если луч ушёл выше
        горизонта (там земли нет, и хвататься не за что)."""
        w, h = size
        a = (sx - w / 2) / w
        b = (h / 2 - sy) / w
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        hs = (h0 - self.tz) * VSCALE
        den = sp - b * cp
        if abs(den) < 1e-6:
            return None
        yr = (b * self.dist - hs * (cp + b * sp)) / den
        zc = yr * cp - hs * sp + self.dist
        if zc <= 1.0:
            return None
        xr = a * zc
        cy, sy_ = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        dx = xr * cy + yr * sy_          # обратный поворот: матрица ортогональна
        dy = -xr * sy_ + yr * cy
        return self.tx + dx, self.ty + dy

    def zoom_at(self, factor, sx, sy, size, h0=0.0):
        """Приблизить/отдалить, оставив точку под курсором на месте. Наклон при этом меняется
        сам, поэтому пересчитывать надо ПОСЛЕ него, иначе земля прыгает."""
        before = self.ground_at(sx, sy, size, h0)
        self.dist *= factor
        self.clamp()
        after = self.ground_at(sx, sy, size, h0)
        if before and after:
            self.tx += before[0] - after[0]
            self.ty += before[1] - after[1]
            self.clamp()

    def drag_to(self, grab, sx, sy, size, h0=0.0):
        """Тащить землю: точка grab (в метрах), взятая при нажатии, идёт под курсор."""
        now = self.ground_at(sx, sy, size, h0)
        if now is None:
            return
        self.tx += grab[0] - now[0]
        self.ty += grab[1] - now[1]
        self.clamp()


def project(xs, ys, hs, cam, size, fov_px=None):
    """Мир (метры, высота в метрах) -> экран. Возвращает (x, y, глубина)."""
    w, h = size
    fov_px = fov_px or w                       # угол обзора один и тот же при любом разрешении
    cy, sy = math.cos(math.radians(cam.yaw)), math.sin(math.radians(cam.yaw))
    cp, sp = math.cos(math.radians(cam.pitch)), math.sin(math.radians(cam.pitch))
    dx = xs - cam.tx
    dy = ys - cam.ty
    hs = hs - cam.tz * VSCALE            # камера ходит вокруг точки НА ЗЕМЛЕ, а не над нулём:
    #                                      иначе вблизи на холмистой карте она уходит под грунт
    xr = dx * cy - dy * sy
    yr = dx * sy + dy * cy
    # Наклон. Камера стоит на орбите: вверх у неё (0, sin p, cos p), вперёд — (0, cos p, -sin p).
    # Отсюда экранная вертикаль и глубина. Раньше эти две строки стояли наоборот, и высота
    # уезжала в глубину вместо верха: холм отодвигался и уменьшался, то есть выглядел ямой,
    # а яма — холмом. На плоской карте ошибка не видна (меняется только видимый наклон),
    # поэтому и дожила до рельефа.
    yc = yr * sp + hs * cp
    zc = yr * cp - hs * sp + cam.dist
    zc = np.maximum(zc, 1.0)
    f = fov_px / zc
    return w / 2 + xr * f, h / 2 - yc * f, zc


def render(surface, height_m, cell_m, cam, size, coarse=1, markers=None, ss=1,
           ground=True):
    """Кадр объёмного вида. surface — сетка типов (Gx,Gy), height_m — высоты в метрах.

    ss — надвыборка: рисуем в ss раз крупнее и ужимаем. Ступеньки на краях четырёхугольников
    именно этим и лечатся; стоит вчетверо дороже, поэтому включается только на неподвижном
    кадре, когда мышь отпущена."""
    if ss > 1:
        img = render(surface, height_m, cell_m, cam,
                     (size[0] * ss, size[1] * ss), coarse, markers, ss=1, ground=ground)
        return img.resize(size, Image.LANCZOS)
    if coarse > 1:
        surface = surface[::coarse, ::coarse]
        height_m = height_m[::coarse, ::coarse]
        cell_m = cell_m * coarse
    Gx, Gy = surface.shape

    ix, iy = np.meshgrid(np.arange(Gx + 1), np.arange(Gy + 1), indexing="ij")
    xs = ix * cell_m
    ys = iy * cell_m
    hh = np.zeros((Gx + 1, Gy + 1), dtype=np.float32)
    hh[:Gx, :Gy] = height_m
    hh[Gx, :Gy] = height_m[-1, :]
    hh[:Gx, Gy] = height_m[:, -1]
    hh[Gx, Gy] = height_m[-1, -1]

    px, py, pz = project(xs, ys, hh * VSCALE, cam, size)

    # отмывка: наклон площадки к солнцу. Без неё поверхность одного цвета читается как плоскость,
    # даже когда она не плоская
    gx = (hh[1:, :-1] - hh[:-1, :-1]) / cell_m
    gy = (hh[:-1, 1:] - hh[:-1, :-1]) / cell_m
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    # Нижняя граница высокая нарочно: теневой склон должен читаться, а не проваливаться в чёрное.
    # С порогом 0.25 гребень выглядел рваной чёрной полосой, и форма земли терялась.
    lam = np.clip((-gx * SUN[0] - gy * SUN[1] + SUN[2]) * nz, 0.58, 1.34)

    lut = np.zeros((max(COLORS) + 1, 3), dtype=np.float32)
    for k, c in COLORS.items():
        lut[k] = c
    base = lut[np.clip(surface, 0, max(COLORS))]
    # Края типов размываем на полклетки: лес обрывается в поле ступенькой ровно по сетке, и
    # именно эта лесенка читается как грязь. Одного прохода хватает — двумя дорога исчезает.
    base = 0.62 * base + 0.38 * np.stack([_box_blur(base[..., k]) for k in range(3)], axis=-1)
    shade = base * lam[..., None]

    # Местность и подставка идут одним списком: их нельзя красить и сортировать порознь, иначе
    # ближняя стенка перекроет холм перед собой или наоборот.
    qx = [np.stack([px[:-1, :-1].ravel(), px[1:, :-1].ravel(),
                    px[1:, 1:].ravel(), px[:-1, 1:].ravel()], axis=1)]
    qy = [np.stack([py[:-1, :-1].ravel(), py[1:, :-1].ravel(),
                    py[1:, 1:].ravel(), py[:-1, 1:].ravel()], axis=1)]
    qc = [shade.reshape(-1, 3)]
    qd = [(0.25 * (pz[:-1, :-1] + pz[1:, :-1] + pz[1:, 1:] + pz[:-1, 1:])).ravel()]
    if ground:
        for gx_, gy_, gc_, gd_ in _ground_quads(xs, ys, hh, cam, size):
            qx.append(gx_); qy.append(gy_); qc.append(gc_); qd.append(gd_)
    QX = np.concatenate(qx); QY = np.concatenate(qy)
    QC = np.concatenate(qc); QD = np.concatenate(qd)

    # воздушная дымка в МЕТРАХ, а не в долях кадра: только так по ней читается расстояние
    t = (np.clip(QD / fog_far(cam.dist), 0.0, 1.0) ** FOG_POW)[:, None] * FOG
    QC = np.clip(QC * (1 - t) + SKY_HORIZON * t, 0, 255).astype(np.uint8)

    order = np.argsort(QD)[::-1]                        # дальние сначала — художник

    sky = np.linspace(0.0, 1.0, size[1], dtype=np.float32)[:, None, None]
    sky = (SKY_TOP * (1 - sky) + SKY_HORIZON * sky).astype(np.uint8)
    img = Image.fromarray(np.repeat(sky, size[0], axis=1))
    d = ImageDraw.Draw(img)
    w, h = size
    for k in order:
        ax, bx, cx_, dx_ = QX[k]
        ay, by, cy_, dy_ = QY[k]
        if max(ax, bx, cx_, dx_) < 0 or min(ax, bx, cx_, dx_) > w:
            continue
        if max(ay, by, cy_, dy_) < 0 or min(ay, by, cy_, dy_) > h:
            continue
        c = QC[k]
        d.polygon([(ax, ay), (bx, by), (cx_, cy_), (dx_, dy_)], fill=(int(c[0]), int(c[1]), int(c[2])))

    for kind, pts, col in (markers or []):
        for p in pts:
            hx = height_m[min(int(p[0] / cell_m), Gx - 1), min(int(p[1] / cell_m), Gy - 1)]
            sx, sy_, _ = project(np.array([p[0]]), np.array([p[1]]),
                                 np.array([hx + 3.0], dtype=np.float32), cam, size)
            d.ellipse([sx[0] - 5, sy_[0] - 5, sx[0] + 5, sy_[0] + 5], fill=col)
    return img
