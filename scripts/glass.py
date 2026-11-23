"""液态玻璃：材质引擎 + 组件（胶囊、卡片、玻璃字、章节转场）。

材质依据 Apple HIG Materials 与 WWDC25「Meet Liquid Glass」：
- 靠折射弯曲光线，不靠模糊：模糊很轻，白色染色很淡。
- 折射按圆弧厚度曲线分布：边缘陡、往里迅速变平，中心几乎不变形。
- 按背后亮度自适应：亮背景上轻压暗、阴影变浅；暗背景上轻提亮、阴影加深。
- 高光是细边光 + 沿边缘两处光斑，光的方向可以随时间转。
- 玻璃里的字用 vibrancy：按背后亮度选深字或白字，不用死黑死白。

所有组件的样式都从预设 presets/glass.json 读，调用时传 style（dict）。
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import core
from core import clamp, ease, ease_io, lerp, memo, place, sp, with_alpha

LUMA = np.array((.2126, .7152, .0722), np.float32)
DEFAULT_LIGHT = math.atan2(-.83, -.55)   # 左上方来光


# ================================================================ 材质引擎
def rrect_sdf(X, Y, cx, cy, hw, hh, r):
    qx, qy = np.abs(X - cx) - (hw - r), np.abs(Y - cy) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def smin(a, b, k):
    """平滑并集：两块玻璃靠近时像水滴一样连起来。"""
    h = np.clip(.5 + .5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _bilinear(a, x, y):
    h, w = a.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    if a.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
            + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)


def _glass(fr, box, d, ds, cov, tint, amount, bezel, refract, blur, sat, bright, shadow, alpha, mag, rim,
           center, light, adapt, glow, glow_at, glow_r, dim):
    x0, y0, x1, y1 = box
    crop = fr.crop(box)
    src = np.asarray(crop, np.float32)
    back = np.asarray(crop.filter(ImageFilter.GaussianBlur(blur)), np.float32) if blur > .3 else src
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32) + .5
    gy, gx = np.gradient(d)
    n = np.hypot(gx, gy) + 1e-5
    gx, gy = gx / n, gy / n

    # 圆弧厚度曲线：u 是离边缘的相对深度，边缘最陡
    u = np.clip(-d / bezel, 0, 1)
    slope = (1 - u) / np.sqrt(np.maximum(1 - (1 - u) ** 2, .015))
    disp = refract * np.clip(slope / 3.2, 0, 1)
    sx = X - x0 - gx * disp
    sy = Y - y0 - gy * disp
    if mag != 1:
        sx = center[0] - x0 + (sx - (center[0] - x0)) / mag
        sy = center[1] - y0 + (sy - (center[1] - y0)) / mag
    px = _bilinear(back, sx, sy)
    k = disp * .045   # 色散只出现在最外缘
    px[..., 0] = _bilinear(back[..., 0], sx - gx * k, sy - gy * k)
    px[..., 2] = _bilinear(back[..., 2], sx + gx * k, sy + gy * k)

    # 背后亮度自适应
    w = cov > .5
    L = float((src[w] @ LUMA).mean()) if w.any() else 128.0
    a = clamp((L - 135) / 90, -1, 1) * adapt
    lum = (px @ LUMA)[..., None]
    px = lum + (px - lum) * sat
    px = px * (bright * (1 - .07 * max(a, 0)) * (1 + .09 * max(-a, 0)))
    px = px + (np.array(tint, np.float32) - px) * amount * (1 + .6 * max(-a, 0))
    if dim > 0:
        px = px * (1 - dim)

    # 高光：细边光 + 沿边缘两处光斑
    L2 = np.array((math.cos(light), math.sin(light)), np.float32)
    lit = gx * L2[0] + gy * L2[1]
    edge = np.exp(-((d + .9) / 1.05) ** 2)
    soft = np.exp(-((d + 3.2) / 3.4) ** 2)
    hot = np.maximum(lit, 0) ** 3
    cold = np.maximum(-lit, 0) ** 3
    spec = (edge * (.22 + 1.05 * hot + .5 * cold) + soft * (.06 + .34 * hot + .14 * cold)) * rim
    inner = np.exp(np.minimum(d, 0) / 9) * (.02 + .1 * hot) * rim
    px = px + (255 - px) * np.clip(spec + inner, 0, 1)[..., None]

    if glow > 0:
        gr = np.hypot(X - glow_at[0], Y - glow_at[1])
        g = glow * np.exp(-(gr / glow_r) ** 2) * np.clip(-d / 6, 0, 1)
        px = px + (255 - px) * np.clip(g * .55, 0, 1)[..., None]

    out = src
    if shadow > 0:
        sh_k = shadow * alpha * (1.25 - .55 * clamp((L - 90) / 130))
        sh = sh_k * np.maximum(np.exp(-np.maximum(ds, 0) / 22) - .04, 0) * np.clip(.5 + ds / 14, 0, 1)
        out = src * (1 - sh[..., None]) + np.array((18, 22, 38), np.float32) * sh[..., None]
    c = (cov * alpha)[..., None]
    res = out * (1 - c) + px * c
    fr.paste(Image.fromarray(np.clip(res + .5, 0, 255).astype(np.uint8)), (x0, y0))


def _box(fr, x0, y0, x1, y1, pad):
    return (max(int(x0) - pad, 0), max(int(y0) - pad, 0), min(int(x1) + pad, fr.width), min(int(y1) + pad, fr.height))


MATERIAL_KEYS = ("tint", "amount", "bezel", "refract", "blur", "sat", "bright", "shadow", "alpha", "mag", "rim",
                 "light", "adapt", "glow", "glow_at", "glow_r", "dim")


def _defaults(kw):
    o = dict(tint=(255, 255, 255), amount=.05, bezel=16, refract=22, blur=2.4, sat=1.22, bright=1.02, shadow=.22,
             alpha=1.0, mag=1.0, rim=1.0, light=DEFAULT_LIGHT, adapt=1.0, glow=0.0, glow_at=(0, 0), glow_r=80.0,
             dim=0.0)
    o.update({k: v for k, v in kw.items() if k in MATERIAL_KEYS})
    o["tint"] = tuple(o["tint"])
    return o


def material(style):
    """从预设里取材质参数。"""
    return {k: v for k, v in (style or {}).get("material", {}).items() if k in MATERIAL_KEYS}


def glass(fr, cx, cy, w, h, r=None, **kw):
    """一块圆角玻璃。r 默认是胶囊（高的一半）。"""
    o = _defaults(kw)
    if w < 2 or h < 2 or o["alpha"] <= 0:
        return
    r = min(r if r is not None else h / 2, w / 2, h / 2)
    glass_field(fr, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                lambda X, Y: rrect_sdf(X, Y, cx, cy, w / 2, h / 2, r), center=(cx, cy), **o)


def glass_field(fr, bounds, fn, center=(0, 0), **kw):
    """任意距离场的玻璃。fn(X, Y) 返回到边缘的有符号距离（里面为负）。"""
    o = _defaults(kw)
    if o["alpha"] <= 0:
        return
    box = _box(fr, *bounds, int(max(o["refract"], 70)) + 6)
    if box[2] - box[0] < 3 or box[3] - box[1] < 3:
        return
    Y, X = np.mgrid[box[1]:box[3], box[0]:box[2]].astype(np.float32) + .5
    d = fn(X, Y)
    _glass(fr, box, d, fn(X, Y - 16) - 3, np.clip(.5 - d, 0, 1), center=center, **o)


def glass_glyphs(fr, mask, x, y, soft=4.0, **kw):
    """把文字本身做成玻璃。mask 是 L 图，(x, y) 是它在画面里的左上角。"""
    o = _defaults(kw)
    if o["alpha"] <= 0:
        return
    cv = Image.new("L", fr.size, 0)
    cv.paste(mask, (int(x), int(y)))
    box = _box(fr, x, y, x + mask.width, y + mask.height, 80)
    c = cv.crop(box)
    rng = soft * 2.5
    d = (.5 - np.asarray(c.filter(ImageFilter.GaussianBlur(soft)), np.float32) / 255) * 2 * rng
    sh = Image.new("L", c.size, 0)
    sh.paste(c, (0, 14))
    ds = (.5 - np.asarray(sh.filter(ImageFilter.GaussianBlur(16)), np.float32) / 255) * 160
    _glass(fr, box, d, ds, np.asarray(c, np.float32) / 255, center=(0, 0), **o)


# ================================================================ 文字
def lum_at(fr, box):
    box = tuple(int(v) for v in box)
    a = np.asarray(fr.crop(box), np.float32).reshape(-1, 3).mean(0)
    return float(a @ LUMA)


def ink_for(fr, box, style):
    """vibrancy：亮背景用深字，暗背景用白字。"""
    v = style.get("vibrancy", {})
    return tuple(v.get("dark", (22, 24, 32))) if lum_at(fr, box) > v.get("threshold", 150) \
        else tuple(v.get("light", (255, 255, 255)))


def glyph(ch, spec, size, fill):
    def make():
        f = core.font(spec, size)
        w, h = int(f.getlength(ch)) + 12, int(size * 1.5)
        im = Image.new("RGBA", (max(w, 1), h), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((6, h / 2), ch, font=f, fill=tuple(fill), anchor="lm")
        return im
    return memo(("glyph", ch, spec, int(size), tuple(fill)), make)


def reveal(fr, s, cx, cy, size, ink, lt, style, alpha=1.0):
    """逐字浮现：轻微上移 + 淡入。每个字比上一个晚 style.reveal.stagger 秒。"""
    rv = style.get("reveal", {})
    stagger, dur, rise = rv.get("stagger", .022), rv.get("duration", .26), rv.get("rise", 10)
    spec = style["font"]
    f = core.font(spec, size)
    x = cx - f.getlength(s) / 2
    for i, ch in enumerate(s):
        adv = f.getlength(s[:i + 1]) - f.getlength(s[:i])
        e = ease((lt - i * stagger) / dur)
        if e > 0 and ch != " ":
            g = with_alpha(glyph(ch, spec, size, ink), e * alpha)
            fr.paste(g, (int(x) - 6, int(cy - g.height / 2 + rise * (1 - e))), g)
        x += adv


# ================================================================ 组件
def morph(lt, dur, w, h, style):
    """玻璃生长：先成小圆，长宽两根不同的弹簧拉开；最后 out 秒收回。"""
    m = style.get("morph", {})
    fw, zw = m.get("spring_w", (1.7, .58))
    fh, zh = m.get("spring_h", (2.6, .5))
    start, out_len = m.get("start", .56), m.get("out", .2)
    a, b = sp(lt, fw, zw), sp(lt, fh, zh)
    out = ease_io((lt - dur + out_len) / out_len)
    s = h * start
    gw, gh = lerp(s, w, a), lerp(s, h, b)
    gw, gh = lerp(gw, s * .6, out), lerp(gh, s * .6, out)
    return max(gw, 1), max(gh, 1), clamp(lt / .06) * (1 - out)


def capsule(fr, cx, cy, text, lt, dur, style, size=None, pad=None, h=None, alpha=1.0, ink=None):
    """玻璃胶囊：弹簧生长 + 文字逐字浮现。lt 是从出现开始的时间。"""
    if lt <= 0:
        return
    c = style.get("capsule", {})
    size = size or c.get("size", 44)
    pad = c.get("pad", 34) if pad is None else pad
    w = core.font(style["font"], size).getlength(text) + pad * 2
    h = h or size * c.get("height", 2.0)
    gw, gh, al = morph(lt, dur, w, h, style)
    al *= alpha
    if al <= 0:
        return
    ink = ink or ink_for(fr, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), style)
    glass(fr, cx, cy, gw, gh, alpha=al, **material(style))
    reveal(fr, text, cx, cy, size, ink, lt - .14, style, al)


def card(fr, im, cx, cy, w, h, lt, dur, style, alpha=1.0):
    """玻璃边框卡片：把录屏、成片、截图放进去。im 是 RGB/RGBA 图，会等比缩放填满卡片。"""
    if lt <= 0:
        return
    c = style.get("card", {})
    r, border = c.get("radius", 36), c.get("border", 6)
    e = sp(lt, *c.get("spring", (2.2, .7)))
    out = ease_io((lt - dur + .25) / .25)
    s = lerp(c.get("from_scale", .9), 1.0, e) * lerp(1, .94, out)
    al = clamp(lt / .12) * (1 - out) * alpha
    if al <= 0:
        return
    gw, gh = w * s, h * s
    glass(fr, cx, cy, gw + border * 2, gh + border * 2, r=r + border, alpha=al, **material(style))
    iw, ih = int(gw), int(gh)
    k = max(iw / im.width, ih / im.height)
    tw, th = int(im.width * k), int(im.height * k)
    pic = im.convert("RGBA").resize((tw, th), Image.LANCZOS).crop(((tw - iw) // 2, (th - ih) // 2,
                                                                   (tw - iw) // 2 + iw, (th - ih) // 2 + ih))
    m = Image.new("L", (iw * 3, ih * 3), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, iw * 3 - 1, ih * 3 - 1), int(r * s) * 3, fill=255)
    m = m.resize((iw, ih), Image.LANCZOS)
    if al < 1:
        m = m.point(lambda v: int(v * al))
    fr.paste(pic, (int(cx - iw / 2), int(cy - ih / 2)), m)


def text_mask(text, spec, size):
    def make():
        f = core.font(spec, size)
        x0, y0, x1, y1 = f.getbbox(text)
        pad = int(size * .2)
        m = Image.new("L", (x1 - x0 + pad * 2, y1 - y0 + pad * 2), 0)
        ImageDraw.Draw(m).text((pad - x0, pad - y0), text, font=f, fill=255)
        return m
    return memo(("tmask", text, spec, int(size)), make)


def glass_text(fr, text, cx, cy, size, lt, style, alpha=1.0):
    """玻璃大字：文字本身做成玻璃，折射身后的画面。用于需要冲击力的标题。"""
    if lt <= 0:
        return
    g = style.get("glass_text", {})
    spec = g.get("font", style["font"])
    e = sp(lt, *g.get("spring", (2.3, .55)))
    s = lerp(g.get("from_scale", 1.35), 1.0, e)
    m = text_mask(text, spec, int(size * s))
    mat = {**material(style), **g.get("material", {})}
    glass_glyphs(fr, m, cx - m.width / 2, cy - m.height / 2, soft=g.get("soft", 4.0),
                 alpha=clamp(lt / .06) * alpha, **mat)


def defocus(fr, e, dim=.28):
    """失焦 + 压暗，e 从 0 到 1。"""
    if e <= 0:
        return fr
    W, H = core.W, core.H
    b = fr.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(10 * e)).resize((W, H), Image.BICUBIC)
    b = Image.blend(b, Image.new("RGB", (W, H), (20, 22, 34)), dim * e)
    return Image.blend(fr, b, clamp(e * 3))


def chapter(fr, t, t0, t1, title, style, sub=None):
    """章节转场：画面失焦，玻璃大字出现，再对焦回来。跟口播原速走，不按住画面。"""
    ch = style.get("chapter", {})
    lt, rest = t - t0, t1 - t
    fin, fout = ch.get("in", .25), ch.get("out", .3)
    e = ease(lt / fin) * ease(rest / fout)
    fr = defocus(fr, e, ch.get("dim", .28))
    cy = ch.get("y", .42) * core.H
    glass_text(fr, title, core.W / 2, cy, ch.get("size", 190), lt, style, alpha=ease(rest / fout))
    if sub:
        capsule(fr, core.W / 2, cy + ch.get("size", 190) * .9, sub, lt - .2, t1 - t0 - .2, style)
    return fr
