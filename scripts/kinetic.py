"""贴脸逐词字幕 + 运镜。

坐标：cam 的 x、y 是原片铺满画布后的位置（画布坐标）；words 的 y、side T 的 stag 是成片屏幕上的位置；
side L / R 的 stag 是离头边缘再往外错开多少像素。字的 x 由头的位置自动算，不用写。

一段 kinetic = 若干镜头块 + 若干词：
- 镜头块 {"t0": 急推开始, "land": 落位, "cam": [取景中心 x, y, 放大倍数, 旋转角度]}
  坐标按画布像素（原片铺满画布后的位置）。块和块之间是一刀急推，上一块的字随之清掉。
  可选：smooth/ease（缓推代替弹簧）、push（落位后持续慢推速度）、drift/drift_t（落位后漂移）。
- 词 {"t": 出现时间, "text", "kind": "s" 小词 | "b" 重点词, "size": 最大字号, "color": "white" | "accent" | [r,g,b],
      "side": "L" 贴头左 | "R" 贴头右 | "T" 头顶（用 stag 给中心 x）, "y": 中线 y, "stag": 阶梯错位,
      "fx": null | "hit" 重字砸下 | "flip" 翻牌计数（items=[[t, 文字, 颜色], ...]） | "zh" 中文小标题（num=编号）,
      "until": 可选，指定清掉的时间}
- behind：放到人身后的词（被头发和身体挡住）；shakes / punches / nudges：镜头震动、砸下推一档、往字那边带一下。

规则见 references/04-贴脸字幕运镜.md。所有手感参数在 presets/kinetic.json。
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import core
from core import clamp, ease, ease_out5, lerp, memo, place, sp


def color_of(c, style):
    if isinstance(c, str):
        return tuple(style["colors"][c])
    return tuple(c)


def word_img(text, kind, size, color, style):
    """平面字 + 很淡的阴影和柔光。返回 (图, (墨迹左, 墨迹右, 中线))。"""
    def make():
        f = core.font(style["fonts"]["big" if kind == "b" else "small"], size)
        x0, y0, x1, y1 = f.getbbox(text)
        pad = int(size * .45)
        w, h = x1 - x0 + pad * 2, y1 - y0 + pad * 2
        txt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((pad - x0, pad - y0), text, font=f, fill=color + (255,))
        return _finish(txt, size, color, style), (pad, pad + (x1 - x0), pad + (y1 - y0) / 2)
    return memo(("kw", text, kind, size, color, id(style)), make)


def _finish(txt, size, color, style):
    wd = style.get("word", {})
    a = txt.getchannel("A")
    w, h = txt.size
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if wd.get("shadow", 95):
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sh.paste((0, 0, 0, 255), (0, int(size * .03) + 2), a.point(lambda v: v * wd.get("shadow", 95) // 255))
        out = Image.alpha_composite(out, sh.filter(ImageFilter.GaussianBlur(max(3, size * .07))))
    if wd.get("glow", 70):
        g = Image.new("RGBA", (w, h), color + (0,))
        g.putalpha(a.filter(ImageFilter.GaussianBlur(max(4, size * .09))).point(lambda v: v * wd.get("glow", 70) // 255))
        out = Image.alpha_composite(out, g)
    soft = wd.get("soft", .35)
    return Image.alpha_composite(out, txt.filter(ImageFilter.GaussianBlur(soft)) if soft else txt)


def zh_title_img(text, size, num, style):
    """中文小标题：编号用重点词字体 + 强调色在上，中文标题字在下。"""
    def make():
        f = core.font(style["fonts"]["zh"], size)
        white, accent = color_of("white", style), color_of("accent", style)
        x0, y0, x1, y1 = f.getbbox(text)
        nf = core.font(style["fonts"]["big"], int(size * .48))
        nb = nf.getbbox(num) if num else (0, 0, 0, 0)
        nh = nb[3] - nb[1]
        pad = int(size * .4)
        w = x1 - x0 + 2 * pad
        h = y1 - y0 + 2 * pad + (nh + int(size * .12) if num else 0)
        txt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(txt)
        top = pad
        if num:
            d.text((pad - nb[0], top - nb[1]), num, font=nf, fill=accent + (255,))
            top += nh + int(size * .12)
        d.text((pad - x0, top - y0), text, font=f, fill=white + (255,))
        return _finish(txt, size, white, style), (pad, pad + x1 - x0, h / 2)
    return memo(("zht", text, size, num, id(style)), make)


def fit_cam(cx, cy, z, roll):
    """取景不能露出画面外的黑边。"""
    W, H = core.W, core.H
    r = math.radians(abs(roll))
    ex = (W / 2 * math.cos(r) + H / 2 * math.sin(r)) / z
    ey = (W / 2 * math.sin(r) + H / 2 * math.cos(r)) / z
    cx = clamp(cx, ex, W - ex) if ex < W / 2 else W / 2
    cy = clamp(cy, ey, H - ey) if ey < H / 2 else H / 2
    return cx, cy, z, roll


def shoot(im, cam, resample=Image.BICUBIC):
    """按机位取景。im 可以是高清解码（比画布大），坐标仍按画布算。"""
    W, H = core.W, core.H
    cx, cy, z, roll = cam
    r = math.radians(roll)
    c, s = math.cos(r) / z, math.sin(r) / z
    coef = (c, -s, cx - c * W / 2 + s * H / 2, s, c, cy - s * W / 2 - c * H / 2)
    k = im.width / W
    if k != 1:
        coef = tuple(v * k for v in coef)
    return im.transform((W, H), Image.AFFINE, coef, resample)


def to_screen(p, cam):
    W, H = core.W, core.H
    cx, cy, z, roll = cam
    r = math.radians(roll)
    dx, dy = (p[0] - cx) * z, (p[1] - cy) * z
    c, s = math.cos(r), math.sin(r)
    return dx * c + dy * s + W / 2, -dx * s + dy * c + H / 2


class Kinetic:
    def __init__(self, seg, video, masks, style):
        self.st = style
        self.video, self.masks = video, masks
        self.B = seg["blocks"]
        self.W = [{"stag": 0, "fx": None, "color": "white", **w} for w in seg["words"]]
        self.start = tuple(seg.get("start_cam", (core.W / 2, core.H / 2, 1.0, 0.0)))
        self.nudges = seg.get("nudges", [])
        self.shakes = seg.get("shakes", [])
        self.punches = seg.get("punches", [])
        self.end = seg["t1"]
        self.behind = set(seg.get("behind", []))
        self._place, self._cam = {}, {}

    # ------------------------------------------------------------ 镜头
    def block_of(self, t):
        k = -1
        for i, b in enumerate(self.B):
            if t >= b["t0"]:
                k = i
        return k

    def camera(self, t):
        key = round(t, 5)
        if key in self._cam:
            return self._cam[key]
        cs = self.st.get("camera", {})
        k = self.block_of(t)
        if k < 0:
            out = fit_cam(*self.start)
        else:
            b = self.B[k]
            prev = self.camera(b["t0"] - 1e-4) if k > 0 else self.start
            if b.get("smooth"):
                p = ease_out5((t - b["t0"]) / b.get("ease", .5))
            else:
                p = sp(t - b["t0"], *cs.get("spring", (4.6, .58)))
            cx, cy, z, roll = (lerp(a, c, p) for a, c in zip(prev, b["cam"]))
            lt = t - b["land"]
            if lt > 0:
                z *= 1 + b.get("push", cs.get("push", .028)) * lt
                dx, dy, dr = b.get("drift", (0, 0, 0))
                u = clamp(lt / max(b.get("drift_t", 1.0), .01))
                cx += dx * u
                cy += dy * u
                roll += dr * u
            for tn, s in self.nudges:
                if b["t0"] <= tn:
                    cx += s * cs.get("nudge", 22) * ease_out5((t - tn) / .2)
            for tp, a in self.punches:
                if t > tp:
                    z *= 1 + a * (1 - sp(t - tp, 3.2, .5)) * min(1, (t - tp) / .02)
            for ts, a in self.shakes:
                q = t - ts
                if 0 < q < .4:
                    e = a * math.exp(-q * 14)
                    cx += e * math.sin(q * 95) / z
                    cy += e * math.cos(q * 71) / z
            out = fit_cam(cx, cy, z, roll)
        if len(self._cam) > 4000:
            self._cam.clear()
        self._cam[key] = out
        return out

    def whip_speed(self, t):
        a, b = self.camera(t - 1 / 60), self.camera(t + 1 / 60)
        return abs(a[0] - b[0]) * a[2] + abs(a[1] - b[1]) * a[2] + abs(a[2] - b[2]) * 900 + abs(a[3] - b[3]) * 30

    # ------------------------------------------------------------ 字
    def word_end(self, i):
        w = self.W[i]
        if "until" in w:
            return w["until"]
        k = self.block_of(w["t"] + 1e-4)
        return self.B[k + 1]["t0"] if k + 1 < len(self.B) else self.end

    def head_screen(self, t, cam):
        xl, xr, hx, top = self.masks.face_band(t, self.st["layout"].get("head_w", 470))
        y = top + 260
        return to_screen((xl, y), cam)[0], to_screen((xr, y), cam)[0], to_screen((hx, y), cam)

    def land_of(self, t0):
        k = self.block_of(t0 + 1e-4)
        return max(t0, self.B[k]["land"] + .05) if k >= 0 else t0

    def ink_w(self, w, size):
        f = self.st["fonts"]
        if w["fx"] == "flip":
            fb = core.font(f["big"], size)
            return max(fb.getbbox(s)[2] - fb.getbbox(s)[0] for _, s, _ in w["items"])
        spec = f["zh"] if w["fx"] == "zh" else f["big" if w["kind"] == "b" else "small"]
        x0, _, x1, _ = core.font(spec, size).getbbox(w["text"])
        return x1 - x0

    def placement(self, i):
        """锚点 x 和实际字号：字太宽就缩到放得下，小词有最小字号。"""
        if i in self._place:
            return self._place[i]
        L = self.st["layout"]
        W = core.W
        margin, tuck_in, gap = L.get("margin", 110), L.get("tuck", 26), L.get("gap", 22)
        w = self.W[i]
        size, side, stag = w["size"], w["side"], w["stag"]
        land = self.land_of(w["t"])
        xl, xr, _ = self.head_screen(land, self.camera(land))
        iw = self.ink_w(w, size)
        tuck = tuck_in if w["text"] in self.behind else -gap
        if side == "L":
            edge = xl + tuck - stag
            room = edge - margin
        elif side == "R":
            edge = xr - tuck + stag
            room = W - margin - edge
        else:
            edge = stag or W / 2
            room = 2 * min(edge, W - edge) - 2 * margin
        fit = int(min(size, size * max(room, 1) / max(iw, 1)))
        if w["kind"] == "s":
            fit = max(fit, L.get("min_small", 50))
        self._place[i] = (edge, fit)
        return self._place[i]

    def draw(self, fr, t, layer, which):
        st = self.st
        margin = st["layout"].get("margin", 110)
        cam = self.camera(t)
        par, flo = st.get("parallax", {}), st.get("float", {})
        sm, bg, hit = st.get("small", {}), st.get("big", {}), st.get("hit", {})
        for i, w in enumerate(self.W):
            behind_ = w["text"] in self.behind
            if t < w["t"] or t >= self.word_end(i) or (which == "b") != behind_:
                continue
            edge, size = self.placement(i)
            land = self.land_of(w["t"])
            _, _, H0 = self.head_screen(land, self.camera(land))
            _, _, H1 = self.head_screen(t, cam)
            k = par.get("small", .5) if w["kind"] == "s" else par.get("big", .8)
            sx = (H1[0] - H0[0]) * k
            sy = (H1[1] - H0[1]) * k * .6
            ph = i * 1.7
            sx += flo.get("x", 3.2) * math.sin(1.35 * t + ph)
            sy += flo.get("y", 4.0) * math.sin(1.05 * t + ph * 1.3)
            lt = t - w["t"]
            target = layer if behind_ else fr
            side = w["side"]
            color = color_of(w["color"], st)
            if w["fx"] == "flip":
                self.draw_flip(target, t, w, edge, size, sx, sy)
                continue
            if w["fx"] == "zh":
                im, (lx, rx, my) = zh_title_img(w["text"], size, w.get("num"), st)
            else:
                im, (lx, rx, my) = word_img(w["text"], w["kind"], size, color, st)
            if w["kind"] == "s":                      # 小词：两帧淡入、轻轻上浮
                a = clamp(lt / sm.get("fade", .067))
                sy += sm.get("rise", 9) * (1 - ease(lt / sm.get("rise_t", .14)))
                s = 1.0
            elif w["fx"] == "hit":                    # 重字：从大砸下，带一点抖
                a = clamp(lt / .04)
                s = lerp(hit.get("from_scale", 1.35), 1.0, sp(lt, *hit.get("spring", (3.0, .5))))
                j = hit.get("jitter", 7) * math.exp(-lt * 16)
                sx += j * math.sin(lt * 120)
                sy += j * math.cos(lt * 97)
            else:                                     # 重点词：第一帧放大偏上，很快落位
                a = 1.0
                e = ease(lt / bg.get("settle", .09))
                s = lerp(bg.get("from_scale", 1.16), 1.0, e)
                sx += bg.get("dx", 8) * (1 - e) * (1 if side == "R" else -1)
                sy -= bg.get("dy", 16) * (1 - e)
            if side == "L":
                cx = edge + sx - (rx - im.width / 2)
            elif side == "R":
                cx = edge + sx + (im.width / 2 - lx)
            else:
                cx = edge + sx
            cy = w["y"] + sy + (im.height / 2 - my)
            half = (rx - lx) / 2 * s
            ink_c = cx + ((lx + rx) / 2 - im.width / 2) * s
            ink_c = clamp(ink_c, margin + half, core.W - margin - half)     # 墨迹不出安全边
            cx = ink_c - ((lx + rx) / 2 - im.width / 2) * s
            place(target, im, cx, cy, s, alpha=a)

    def draw_flip(self, fr, t, w, edge, size, sx, sy):
        """翻牌计数：旧字往上滑走，新字从下面顶上来；最后一个用重字落位。"""
        items = w["items"]
        cur = max((j for j, it in enumerate(items) if t >= it[0]), default=None)
        if cur is None:
            return
        for j in (cur - 1, cur):
            if j < 0:
                continue
            _, s_, col = items[j]
            q = t - items[cur][0]
            im, (lx, rx, my) = word_img(s_, "b", size, color_of(col, self.st), self.st)
            e = ease_out5(q / .09)
            if j == cur:
                dy, a = 60 * (1 - e), clamp(q / .05)
                sc = lerp(1.25, 1.0, sp(q, 3.0, .5)) if j == len(items) - 1 else 1.0
            else:
                dy, a, sc = -60 * e, 1 - e, 1.0
                if a <= 0:
                    continue
            cx = edge + sx - (rx - im.width / 2) if w["side"] == "L" else edge + sx + (im.width / 2 - lx)
            place(fr, im, cx, w["y"] + sy + dy + (im.height / 2 - my), sc, alpha=a)

    # ------------------------------------------------------------ 整帧
    def frame(self, t):
        fa = self.video.at(t)
        cam = self.camera(t)
        wp = self.st.get("whip", {})
        v = self.whip_speed(t)
        if v > wp.get("threshold", 40):               # 急推时的动态模糊：同一帧里叠几个机位
            n = int(clamp(v / 20, 3, wp.get("max_samples", 12)))
            acc = None
            for j in range(n):
                im = np.asarray(shoot(fa, self.camera(t - 1 / 45 * j / max(n - 1, 1))), np.float32)
                acc = im if acc is None else acc + im
            fr = Image.fromarray((acc / n + .5).astype(np.uint8))
        else:
            fr = shoot(fa, cam)
        layer = Image.new("RGBA", (core.W, core.H), (0, 0, 0, 0))
        self.draw(fr, t, layer, "b")
        if layer.getbbox():                            # 人身后的字：先画字，再把人按遮罩盖回去
            m = self.masks.at(t)
            person = fr.copy()
            fr.paste(layer, (0, 0), layer)
            if m is not None:
                fr.paste(person, (0, 0), shoot(m, cam, Image.BILINEAR))
        self.draw(fr, t, None, "s")
        return fr
