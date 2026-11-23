"""对标拆解：把「我想要这个感觉」测成数字。只依赖 numpy + pillow + ffmpeg；人物占比需要 mediapipe（可选）。

先看全貌，再测：
  python3 analyze_ref.py sheet 对标.mp4                       # 每秒 4 张的总览图，找要拆的那几秒
  python3 analyze_ref.py sheet 对标.mp4 --at 2.2 --dur 1      # 按原帧率逐帧，带帧号（看入场几帧落位）
宏观（整条视频）：
  python3 analyze_ref.py cuts 对标.mp4                        # 切点、镜头数、中位镜头时长、每分钟切换数
  python3 analyze_ref.py sfx 对标.mp4                         # 音效起音和切点的关系（和随机位置对比）
  python3 analyze_ref.py people 对标.mp4 --model selfie_multiclass_256x256.tflite   # 人物占比、最长连续出镜
微观（用户指定的那几秒），两步：
  1) 扫候选：python3 analyze_ref.py text 对标.mp4 --range 0 8.8 [--crop ...] [--words words.json]
     自动找新出现的字，每个候选出一张证据条。背景在动、人在动的素材会有误报和漏报，所以只当候选。
  2) 精测：看证据条和带网格的总览（sheet --grid），把确认是字的写进清单，逐个测：
     python3 analyze_ref.py measure 对标.mp4 --list 字.json --words words.json
     清单：[{"text": "EXACT", "t": 3.3, "box": [0.62, 0.33, 0.96, 0.5]}, ...]（t 是大致出现时间，box 框住它出现时的位置）
     测：出现帧和开口差几帧、字高、颜色、每帧缩放（入场放大多少、几帧落位）、首帧偏移、虚焦、
     描边阴影、停留期间跟着画面漂移多少、什么时候清掉、是否随切镜清掉；汇总出层级比、一词一行比例等。
一次全跑：
  python3 analyze_ref.py all 对标.mp4 --range 0 8.8 --out 对标拆解/

--crop 是画面里要分析的那一块（左 上 右 下，比例）。对标是录屏、画面套在卡片里时用它，
否则卡片外的背景、播放器按钮都会被算进去。
所有尺寸都换算成「1080 宽竖屏画布上的像素」，可以直接对照 presets 里的参数。
结果写 JSON；字的证据帧存成图片，报告里引用。协议见 references/02-对标拆解协议.md。
"""
import argparse
import collections
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


# ================================================================ 解码
def info(src):
    out = subprocess.run([core.tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", src],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return dict(w=s["width"], h=s["height"], fps=float(num) / float(den), duration=float(d["format"]["duration"]))


def frames(src, t0=0.0, t1=None, width=None, fps=None, crop=None):
    """逐帧读 RGB（numpy uint8）。fps=None 按原帧率；crop=(x0,y0,x1,y1) 比例；width 是裁完后的宽。
    返回 (帧生成器, 尺寸, 实际帧率)。"""
    inf = info(src)
    vf = []
    cw, ch = inf["w"], inf["h"]
    if crop:
        x0, y0, x1, y1 = crop
        cw, ch = int(inf["w"] * (x1 - x0)) // 2 * 2, int(inf["h"] * (y1 - y0)) // 2 * 2
        vf.append(f"crop={cw}:{ch}:{int(inf['w'] * x0)}:{int(inf['h'] * y0)}")
    if width and width != cw:
        h = int(round(ch * width / cw / 2)) * 2
        vf.append(f"scale={width}:{h}:flags=area")
        cw, ch = width, h
    if fps:
        vf.append(f"fps={fps}")
    rate = fps or inf["fps"]
    cmd = [core.tool("ffmpeg"), "-v", "error", "-ss", f"{t0:.3f}"] + (["-t", f"{t1 - t0:.3f}"] if t1 else []) + \
          ["-i", src, "-an"] + (["-vf", ",".join(vf)] if vf else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    def gen():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        n = cw * ch * 3
        try:
            while True:
                b = p.stdout.read(n)
                if len(b) < n:
                    break
                yield np.frombuffer(b, np.uint8).reshape(ch, cw, 3)
        finally:
            p.stdout.close()
            p.kill()
            p.wait()
    return gen(), (cw, ch), rate


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


# ================================================================ 总览图
def sheet(src, out, at=None, dur=None, fps=None, cols=None, crop=None, tile=240, grid=False):
    """at/dur 给了就按原帧率逐帧（标帧号），没给就整条按 fps（默认 4）抽。"""
    native = at is not None
    t0 = at or 0.0
    t1 = t0 + dur if dur else None
    it, (w, h), rate = frames(src, t0, t1, width=tile, fps=None if native else (fps or 4), crop=crop)
    ims = list(it)
    cols = cols or (8 if native else 9)
    rows = (len(ims) + cols - 1) // cols
    th = int(tile * h / w)
    im = Image.new("RGB", (cols * tile, rows * (th + 22)), (20, 20, 24))
    d = ImageDraw.Draw(im)
    f = core.font("SmileySans-Oblique.ttf", 18)
    for i, a in enumerate(ims):
        x, y = (i % cols) * tile, (i // cols) * (th + 22)
        im.paste(Image.fromarray(a), (x, y))
        if grid:                                       # 0.1 网格，方便写 measure 的框
            for k in range(1, 10):
                d.line((x + k * tile // 10, y, x + k * tile // 10, y + th), fill=(0, 200, 90))
                d.line((x, y + k * th // 10, x + tile, y + k * th // 10), fill=(0, 200, 90))
        t = t0 + i / rate
        label = f"{fmt(t)}  #{int(round(t * rate))}" if native else fmt(t)
        d.text((x + 4, y + th + 2), label, font=f, fill=(240, 220, 120))
    if not os.path.splitext(out)[1]:                 # 给的是目录
        out = os.path.join(out, "总览.jpg" if at is None else f"逐帧_{at:g}s.jpg")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    im.save(out, quality=88)
    return out


# ================================================================ 切点
def cuts(src, crop=None, thr=28.0, ratio=3.0, min_shot=.15):
    """相邻帧 32×32 缩略图的平均色差，明显高于周围几帧的算一刀。"""
    it, _, rate = frames(src, width=64, crop=crop)
    d, prev = [], None
    for a in it:
        s = np.asarray(Image.fromarray(a).resize((32, 32), Image.BOX), np.float32)
        d.append(0.0 if prev is None else float(np.abs(s - prev).mean()))
        prev = s
    d = np.array(d)
    n = len(d)
    cut = []
    for i in range(1, n):
        lo, hi = max(1, i - 8), min(n, i + 9)
        around = np.concatenate([d[lo:i], d[i + 1:hi]])
        base = np.median(around) if len(around) else 0
        if d[i] > thr and d[i] > ratio * (base + 1.5) and d[i] == d[max(1, i - 2):i + 3].max():
            if cut and i / rate - cut[-1] < min_shot:    # 转场动画会连着触发两次，算一刀
                continue
            cut.append(round(i / rate, 3))
    dur = n / rate
    still = float(np.mean(d[1:] < .4)) if n > 1 else 0.0       # 和上一帧几乎一样的帧
    content_fps = round(rate * (1 - still), 1)
    edges = [0.0] + cut + [dur]
    shots = np.diff(edges)
    per_min = collections.Counter(int(c // 60) for c in cut)
    return {
        "时长": round(dur, 2), "帧率": round(rate, 3), "切点": cut, "镜头数": len(shots),
        "平均镜头时长": round(float(shots.mean()), 2), "中位镜头时长": round(float(np.median(shots)), 2),
        "最长镜头": round(float(shots.max()), 2), "最短镜头": round(float(shots.min()), 2),
        "每分钟切换": [per_min.get(m, 0) for m in range(int(dur // 60) + 1)],
        "实际内容帧率": content_fps,
        "说明_帧率": "实际内容帧率明显低于文件帧率 = 有重复帧（比如 30 帧的片装在 60 帧的录屏里），数入场帧数时按实际内容帧率算",
    }


# ================================================================ 音效和切点
def onsets(src, sr=16000):
    raw = subprocess.run([core.tool("ffmpeg"), "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le",
                          "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop, win = 256, 1024
    if len(x) < win * 2:
        return [], hop / sr
    idx = np.arange(0, len(x) - win, hop)
    spec = np.abs(np.fft.rfft(x[idx[:, None] + np.arange(win)] * np.hanning(win), axis=1))
    flux = np.maximum(np.diff(np.log1p(spec * 10), axis=0), 0).sum(1)
    flux = np.concatenate([[0], flux])
    med = np.convolve(flux, np.ones(31) / 31, "same")
    mad = np.median(np.abs(flux - np.median(flux))) + 1e-6
    pk = [i for i in range(1, len(flux) - 1)
          if flux[i] > med[i] + 6 * mad and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]]
    keep, last = [], -1e9
    for i in pk:                                       # 50 ms 内只算一个
        if (i - last) * hop / sr > .05:
            keep.append(i)
            last = i
    return [round(i * hop / sr, 3) for i in keep], hop / sr


def sfx(src, cut_list=None, crop=None, win=.25):
    c = cut_list if cut_list is not None else cuts(src, crop)["切点"]
    on, _ = onsets(src)
    dur = info(src)["duration"]
    near = lambda t: any(abs(t - o) <= win for o in on)
    rng = np.random.default_rng(7)
    rand = rng.uniform(win, max(dur - win, win * 2), 2000)
    p_cut = sum(near(t) for t in c) / len(c) if c else None
    p_rand = float(np.mean([near(t) for t in rand]))
    verdict = None
    if p_cut is not None:
        verdict = ("音效卡在切点上" if p_cut > p_rand + .25 else
                   "音效和切点没有关系（不要每刀配声）" if abs(p_cut - p_rand) <= .15 else "有一部分切点配了音效")
    return {"起音数": len(on), "每秒起音": round(len(on) / dur, 2), "切点附近有起音的比例": None if p_cut is None else round(p_cut, 2),
            "随机位置有起音的比例": round(p_rand, 2), "判断": verdict, "窗口秒": win}


# ================================================================ 人物
def people(src, model, crop=None, fps=2.0, face_min=.002):
    mp, vision, BaseOptions = core.mediapipe()
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model), output_category_mask=True))
    it, _, rate = frames(src, width=256, fps=fps, crop=crop)
    on = []
    for a in it:
        m = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(a))).category_mask.numpy_view()
        on.append(float((m == 3).mean()) > face_min)     # 3 = 脸部皮肤
    on = np.array(on)
    runs, cur = [], 0
    for v in on:
        cur = cur + 1 if v else 0
        runs.append(cur)
    first_off = next((i / rate for i, v in enumerate(on) if not v), None)
    return {"人物出镜占比": round(float(on.mean()), 2) if len(on) else None,
            "最长连续出镜秒": round(max(runs, default=0) / rate, 1),
            "第一次切离人物": None if first_off is None else round(first_off, 1),
            "采样": f"{fps} 帧/秒，看得到脸算出镜"}


# ================================================================ 字
def _dilate(m, rx, ry=None):
    ry = rx if ry is None else ry
    out = m.copy()
    for dx in range(1, rx + 1):
        out[:, dx:] |= m[:, :-dx]
        out[:, :-dx] |= m[:, dx:]
    m2 = out.copy()
    for dy in range(1, ry + 1):
        out[dy:] |= m2[:-dy]
        out[:-dy] |= m2[dy:]
    return out


def _label(m):
    """4 邻接连通域，返回 [像素坐标数组 (ys, xs)]。只在真像素上走，稀疏时很快。"""
    seen = np.zeros_like(m)
    ys, xs = np.nonzero(m)
    h, w = m.shape
    comps = []
    for y, x in zip(ys, xs):
        if seen[y, x]:
            continue
        q = collections.deque([(y, x)])
        seen[y, x] = True
        py, px = [], []
        while q:
            cy, cx = q.popleft()
            py.append(cy)
            px.append(cx)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comps.append((np.array(py), np.array(px)))
    return comps


def _texty(a):
    """像字的像素：偏白的平面色，或者很鲜艳的颜色；并且紧挨着锐利的边缘（虚化的灯、背景不算）。
    边缘按颜色差算，不只看亮度：深红字压在暗背景上亮度差很小，颜色差很大。"""
    a16 = a.astype(np.int16)
    mx, mn = a16.max(2), a16.min(2)
    col = ((mn >= 165) & (mx - mn <= 60)) | ((mx - mn >= 100) & (mx >= 130))
    g = np.zeros(mx.shape, bool)                      # 隔一个像素比：轻微虚焦的字边也算
    gx = np.abs(a16[:, 2:] - a16[:, :-2]).sum(2) > 120
    gy = np.abs(a16[2:] - a16[:-2]).sum(2) > 120
    g[:, 1:-1] |= gx
    g[1:-1] |= gy
    return col & _dilate(g, 1)


def _texty_soft(a):
    a16 = a.astype(np.int16)
    mx, mn = a16.max(2), a16.min(2)
    return ((mn >= 120) & (mx - mn <= 60)) | ((mx - mn >= 60) & (mx >= 100))


def _bbox(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _name_color(rgb):
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    if mn >= 175 and mx - mn <= 75:
        return "白"
    if r >= g and r >= b:
        return "红" if g < r * .45 else "橙" if g < r * .75 else "黄"
    if g >= r and g >= b:
        return "绿"
    return "蓝"


def _hue_sat(c):
    r, g, b = (v / 255 for v in c)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0, 0.0
    d = mx - mn
    h = ((g - b) / d % 6) if mx == r else ((b - r) / d + 2) if mx == g else ((r - g) / d + 4)
    return h * 60, d / mx


def _color_name(col, ring_px):
    """字色名。和背景同色相、更亮、但饱和度明显更低的，多半是半透明白字或细笔画和背景混色。"""
    name = _name_color(col)
    if ring_px is not None and len(ring_px) and name != "白":
        bg = tuple(int(v) for v in np.median(ring_px, 0))
        (h1, s1), (h2, s2) = _hue_sat(col), _hue_sat(bg)
        dh = min(abs(h1 - h2), 360 - abs(h1 - h2))
        if dh < 30 and sum(col) > sum(bg) + 60 and s1 < s2 * .7:
            return "白（半透明或和背景混色）"
    return name


def _comps(tm, rx, ry, min_px):
    """把一帧里像字的像素按词聚成块：在 2 倍下采样的格子上膨胀后找连通域。"""
    h, w = tm.shape
    sm = tm[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2).any((1, 3))
    out = []
    for ys, xs in _label(_dilate(sm, max(1, rx // 2), max(1, ry // 2))):
        x0, y0, x1, y1 = xs.min() * 2, ys.min() * 2, xs.max() * 2 + 2, ys.max() * 2 + 2
        cnt = int(tm[y0:y1, x0:x1].sum())
        if cnt >= min_px and y1 - y0 >= 8 and (x1 - x0) * (y1 - y0) < .25 * w * h:
            out.append(((x0, y0, x1, y1), cnt))
    return out


def _shift(m, dx, dy):
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = m[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
    return out


def _match(a, b, r=6):
    """a 平移多少能和 b 最重合（允许 1 像素误差）。返回 (dx, dy, 重合度)。"""
    if not a.any() or not b.any():
        return 0, 0, 0.0
    bd = _dilate(b, 1)
    best = (0, 0, -1.0)
    for dy in range(-r, r + 1, 2):
        for dx in range(-r, r + 1, 2):
            sa = _shift(a, dx, dy)
            sc = min((sa & bd).sum() / a.sum(), (b & _dilate(sa, 1)).sum() / b.sum())
            if sc > best[2]:
                best = (dx, dy, sc)
    dx0, dy0, _ = best
    for dy in (dy0 - 1, dy0, dy0 + 1):
        for dx in (dx0 - 1, dx0, dx0 + 1):
            sa = _shift(a, dx, dy)
            sc = min((sa & bd).sum() / a.sum(), (b & _dilate(sa, 1)).sum() / b.sum())
            if sc > best[2]:
                best = (dx, dy, sc)
    return best


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0


def _track(seq, final, bb, chg, K):
    """拿落位后的字形当模板，从第 K 帧往回逐帧找它的缩放和位置。
    返回每帧 [(缩放, dx, dy, 重合度)]，下标 0 是出现帧。字边入场边跟着画面移动也能测准。"""
    x0, y0, x1, y1 = bb
    tw, th = x1 - x0, y1 - y0
    T = Image.fromarray((final[y0:y1, x0:x1] * 255).astype(np.uint8))
    h, w = final.shape
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cache = {}

    def tmpl(sc):
        k = round(sc, 3)
        if k not in cache:
            cache[k] = np.asarray(T.resize((max(1, int(round(tw * sc))), max(1, int(round(th * sc)))), Image.NEAREST)) > 127
        return cache[k]

    def score(M, Md, sc, dx, dy):
        t = tmpl(sc)
        hh, ww = t.shape
        ax, ay = int(round(cx + dx - ww / 2)), int(round(cy + dy - hh / 2))
        if ax < 0 or ay < 0 or ax + ww > w or ay + hh > h:
            return 0.0
        a = t.sum()
        hit = (Md[ay:ay + hh, ax:ax + ww] & t).sum()
        got = (M[ay:ay + hh, ax:ax + ww] & _dilate(t, 1)).sum()
        return 2 * min(hit, a) / (a + max(got, 1)) if a else 0.0

    out = [None] * (K + 1)
    cur = (1.0, 0, 0)
    for j in range(K + 1, 0, -1):
        fr, tm = seq[j]
        M = tm & chg(fr)
        Md = _dilate(M, 1)
        s0, dx0, dy0 = cur
        best = (-1.0, s0, dx0, dy0)
        for sc in np.arange(max(.5, s0 - .14), s0 + .15, .02):
            for dy in range(dy0 - 8, dy0 + 9, 2):
                for dx in range(dx0 - 8, dx0 + 9, 2):
                    v = score(M, Md, sc, dx, dy)
                    if v > best[0]:
                        best = (v, sc, dx, dy)
        v, sc, dx, dy = best
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                vv = score(M, Md, sc, dx + ddx, dy + ddy)
                if vv > v:
                    v, best = vv, (vv, sc, dx + ddx, dy + ddy)
        _, sc, dx, dy = best
        out[j - 1] = (round(float(sc), 3), dx, dy, round(float(v), 2))
        if v > .35:
            cur = (sc, dx, dy)
    return out


def _measure(seq, o, t0, bb0, w, h, rate, K, min_px, rej=None, reg=None):
    """测一个字。seq 是 [(帧, 字掩码)]，seq[0] 是出现前一帧，seq[1+K] 是出现后第 K 帧；o 是出现帧（相对 t0）。
    rej 不为 None 时按三条规则过滤（自动扫描用）；指定了 reg（手动框）就不过滤。"""
    k1080 = 1080 / w
    strict = reg is None
    rej = rej or (lambda *a_: None)
    pre, pre_fr = seq[0][1], seq[0][0].astype(np.int16)
    fin_fr, fin_tm = seq[1 + K][0], seq[1 + K][1]
    chg = lambda fr: np.abs(fr.astype(np.int16) - pre_fr).sum(2) > 90    # 颜色真的变了（字边上的背景不变）
    if reg is None:
        pad = int(max(bb0[2] - bb0[0], bb0[3] - bb0[1]) * .25) + 4
        reg = np.zeros_like(pre)
        reg[max(0, bb0[1] - pad):bb0[3] + pad, max(0, bb0[0] - pad):bb0[2] + pad] = True
    final = fin_tm & ~pre & reg & chg(fin_fr)
    bb = _bbox(final)
    if bb is None or final.sum() < min_px:
        return None
    x0, y0, x1, y1 = bb
    hf, cyf = y1 - y0, (y0 + y1) / 2
    px = fin_fr[final]
    ring = _dilate(final, 7) & ~_dilate(final, 4)
    if ring.any():                                    # 字色取笔画中心：离背景色最远的那一半像素（细字的边是和背景混出来的）
        dist = np.abs(px.astype(np.int16) - np.median(fin_fr[ring], 0)).sum(1)
        core_px = px[dist >= np.median(dist)]
    else:
        core_px = px
    col = tuple(int(v) for v in np.median(core_px, 0))
    # 1) 落位后形状稳定（允许整体平移：字可能跟着镜头漂；允许 1 像素误差）
    mid = seq[K - 3][1] & ~pre & reg & chg(seq[K - 3][0])
    ys_, xs_ = np.nonzero(reg)
    sl = (slice(ys_.min(), ys_.max() + 1), slice(xs_.min(), xs_.max() + 1))
    ddx, ddy, sc = _match(mid[sl], final[sl])
    if strict and sc < .75:
        rej("落位后还在变形", o, bb0, f"{sc:.2f}")
        return None
    # 2) 字是「贴上去」的：外面一圈是别的颜色。大片墙、天空、衣服贴着人边的那一条，外面还是同一种颜色
    same = float((np.abs(fin_fr[ring].astype(np.int16) - np.array(col)).sum(1) < 70).mean()) if ring.any() else 1.0
    if strict and same > .3:
        rej("和周围反差不够", o, bb0, f"外圈同色 {same:.2f}")
        return None
    # 3) 字有笔画，不是实心块
    if strict and final.sum() / max((x1 - x0) * (y1 - y0), 1) > .75:
        rej("实心块", o, bb0)
        return None
    tr = _track(seq, final, bb, chg, K)
    first = next((x for x in tr if x and x[3] > .35), None)
    scale0 = first[0] if first else None
    sc_ = [x[0] if x and x[3] > .35 else None for x in tr]
    sm_ = []                                           # 3 帧中值，去掉模板匹配的抖动（精度约 ±4%）
    for i in range(len(sc_)):
        win_ = [v for v in sc_[max(0, i - 1):i + 2] if v is not None]
        sm_.append(float(np.median(win_)) if win_ else None)
    settle = 0
    for i in range(len(sm_) - 1, -1, -1):
        if sm_[i] is None or abs(sm_[i] - 1) > .06:
            settle = i + 1
            break
    # 漂移：落位后平均每帧移动；首帧偏移：首帧位置减去按漂移倒推的位置（正 = 偏下）
    after = [x for x in tr[settle:] if x and x[3] > .35]
    vx = vy = 0.0
    if len(after) >= 2:
        n_ = len(after) - 1
        vx, vy = (after[-1][1] - after[0][1]) / n_, (after[-1][2] - after[0][2]) / n_
    off = None
    if first and after:
        k_ = tr.index(first)
        off = (first[2] - (after[-1][2] - vy * (len(tr) - 1 - k_))) * k1080
    ddx, ddy = vx * 4, vy * 4
    r1 = _dilate(final, 2) & ~final
    l1 = float(fin_fr[r1].mean()) if r1.any() else 0
    l2 = float(fin_fr[ring].mean()) if ring.any() else 0
    return dict(frame=int(round(t0 * rate)) + o, t=round(t0 + o / rate, 3),
                bbox=[round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
                字高_1080=round(hf * k1080, 1), 宽_1080=round((x1 - x0) * k1080, 1),
                颜色=col, 颜色名=_color_name(col, fin_fr[ring] if ring.any() else None),
                首帧放大=None if scale0 is None else round(scale0, 3),
                首帧偏移_1080=None if off is None else round(off, 1),
                落位帧数=settle, 落位秒=round(settle / rate, 3),
                每帧缩放=[x[0] if x else None for x in tr],
                字边暗圈=round(l1 / max(l2, 1), 2),
                落位后每帧漂移_1080=round(float(np.hypot(ddx, ddy)) / 4 * k1080, 1),
                虚焦=round(float(np.mean(_dilate(final, 2) & ~final & _texty_soft(fin_fr) & ~fin_tm)) * 10, 2),
                _mask=final, _bb=bb)


def _strip(seq, e, path):
    """证据条：出现前一帧、出现帧、之后每帧直到落位后两帧，框出字。帧号是原片帧号。"""
    x0, y0, x1, y1 = e["_bb"]
    h, w = seq[0][1].shape
    n_show = min(len(seq), e["落位帧数"] + 3)
    pad2 = int(max(x1 - x0, y1 - y0) * .5) + 10
    box = (max(0, x0 - pad2), max(0, y0 - pad2), min(w, x1 + pad2), min(h, y1 + pad2))
    tiles = []
    for j in range(n_show):
        t = Image.fromarray(seq[j][0]).crop(box)
        ImageDraw.Draw(t).rectangle((x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1]), outline=(0, 255, 120))
        tiles.append(t)
    tw, th = tiles[0].size
    sc = min(1.0, 200 / th)
    tw, th = max(1, int(tw * sc)), max(1, int(th * sc))
    strip = Image.new("RGB", (len(tiles) * (tw + 4), th + 22), (20, 20, 24))
    d = ImageDraw.Draw(strip)
    f = core.font("SmileySans-Oblique.ttf", 16)
    for j, t in enumerate(tiles):
        strip.paste(t.resize((tw, th)), (j * (tw + 4), 0))
        d.text((j * (tw + 4) + 3, th + 2), f"#{e['frame'] - 1 + j}", font=f, fill=(240, 220, 120))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    strip.save(path, quality=88)
    return path


def word(src, t, box, crop=None, width=540, K=12, words=None, cut_list=None, evidence=None, look=.35):
    """手动指定一个字：大致出现时间 t（前后 look 秒内找）和框 box（左 上 右 下，比例，相对 --crop 之后的画面）。
    精确测它的出现帧、入场、颜色、虚焦、漂移和停留。自动扫描漏掉或者误判的字用这个补测。"""
    nxt = min([c for c in (cut_list or []) if c > t + .05], default=t + 6)
    prv = max([c for c in (cut_list or []) if c <= t + .05], default=-1.0)
    end = min(nxt + .05, t + 6)
    t0 = max(0.0, t - look, prv + .04)                # 不跨过上一刀
    it, (w, h), rate = frames(src, t0, end, width=width, crop=crop)
    seq = [(a, _texty(a)) for a in it]
    reg = np.zeros((h, w), bool)
    reg[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)] = True
    f0 = seq[0][0].astype(np.int16)
    base = seq[0][1] & reg
    cnt = np.array([(tm & reg & ~base & (np.abs(a.astype(np.int16) - f0).sum(2) > 90)).sum() for a, tm in seq])
    win = int((2 * look) * rate) + K
    peak = cnt[:win].max() if len(cnt) else 0
    if peak == 0:
        return {"说明": "框里没找到新出现的字，换个时间或把框画大一点"}
    o = int(np.argmax(cnt[:win] > .3 * peak))
    o = max(o, 1)
    if o + K + 1 >= len(seq):
        return {"说明": "这个字离切镜或片尾太近，测不完入场"}
    e = _measure(seq[o - 1:o + K + 1], o, t0, None, w, h, rate, K, 1, reg=reg)
    if not e:
        return {"说明": "没测出来"}
    m = e["_mask"]
    ys_, xs_ = np.nonzero(m)
    gone, dx, dy, path = None, 0, 0, []
    for j in range(o + K + 1, len(seq)):            # 跟着字走：字可能贴在场景里随镜头移动
        tmj = seq[j][1]
        best = (0.0, dx, dy)
        for ddy in range(dy - 3, dy + 4):
            for ddx in range(dx - 3, dx + 4):
                yy, xx = ys_ + ddy, xs_ + ddx
                ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                sc = tmj[yy[ok], xx[ok]].sum() / len(ys_)
                if sc > best[0]:
                    best = (sc, ddx, ddy)
        if best[0] < .3:
            gone = j
            break
        _, dx, dy = best
        path.append((dx, dy))
    if path:
        e["停留期间位移_1080"] = round(float(np.hypot(*path[-1])) * 1080 / w, 1)
        e["停留期间每秒位移_1080"] = round(float(np.hypot(*path[-1])) * 1080 / w / (len(path) / rate), 1)
    e["清掉"] = round(t0 + (gone if gone is not None else len(seq)) / rate, 3)
    e["停留秒"] = round(e["清掉"] - e["t"], 2)
    e["随切镜清掉"] = gone is None or abs(e["清掉"] - nxt) <= 3 / rate
    if words:
        near = min((x["t0"] for x in words), key=lambda s_: abs(s_ - e["t"]))
        e["最近的词开口"] = round(near, 3)
        e["比开口晚帧数"] = round((e["t"] - near) * rate, 1)
    if evidence:
        e["证据"] = _strip(seq[o - 1:o + K + 1], e, evidence)
    e.pop("_mask")
    e.pop("_bb")
    return e


def text(src, t0=0.0, t1=None, crop=None, width=540, K=12, words=None, evidence=None, cut_list=None, max_evidence=40,
         debug=False):
    """逐帧找新出现的字块：上一帧这里没有字，之后 K 帧（默认 12，60fps 下 0.2 秒）还在。逐个测入场和停留。"""
    it, (w, h), rate = frames(src, t0, t1, width=width, crop=crop)
    k1080 = 1080 / w                                    # 换算到 1080 宽画布
    cut_frames = set()
    for c in (cut_list or []):
        f = int(round((c - t0) * rate))
        cut_frames.update(range(f - 2, f + 3))          # 切镜那几帧整个画面都是新的，跳过（和切镜同一帧出现的字测不到）
    rx, ry = max(4, int(w * .016)), max(2, int(w * .006))
    cut_rel = sorted((c - t0) * rate for c in (cut_list or []))
    shot_of = lambda f: sum(c <= f + 2 for c in cut_rel)
    min_px = max(12, int(w * w * 6e-5))
    buf = collections.deque(maxlen=K + 2)              # (帧, 字掩码)
    events, n_read = [], 0
    rejects = collections.Counter()                    # 候选被哪条规则滤掉，调阈值时看
    detail = []

    def rej(why, o, bb, extra=""):
        rejects[why] += 1
        if debug:
            detail.append(f"{t0 + o / rate:.3f} {[round(v / w, 2) for v in bb]} {why} {extra}")
    if evidence:
        os.makedirs(evidence, exist_ok=True)

    def measure(o, bb0):
        e = _measure(list(buf), o, t0, bb0, w, h, rate, K, min_px, rej=rej)
        if e and evidence and len(events) < max_evidence:
            e["证据"] = _strip(list(buf), e, os.path.join(evidence, f"字{len(events) + 1:02d}_{e['t']:.2f}s.jpg"))
        return e

    for a in it:
        tm = _texty(a)
        buf.append((a, tm))
        n_read += 1
        now = n_read - 1
        for e in events:                               # 正在显示的字：什么时候消失
            if e["_shot"] != shot_of(now):
                if e.get("清掉") is None:
                    e["清掉"] = round(t0 + (now - 1) / rate, 3)
                continue
            keep = (tm & e["_mask"]).sum() / max(e["_mask"].sum(), 1)
            if e.get("清掉") is None:
                e["_miss"] = e.get("_miss", 0) + 1 if keep < .3 else 0
                if e["_miss"] >= 2:
                    e["清掉"] = round(t0 + (now - 1) / rate, 3)
            elif keep > .6:                            # 被人挡住又露出来，不算消失
                e["清掉"], e["_miss"] = None, 0
        if len(buf) < K + 2:
            continue
        o = n_read - len(buf) + 1                      # 候选出现帧 = buf[1]
        if o in cut_frames:
            continue
        pre_tm, cur_tm, fin_tm = buf[0][1], buf[1][1], buf[1 + K][1]
        new = cur_tm & ~_dilate(pre_tm, 1)             # 这一帧新增、上一帧没有的字像素
        if new.sum() < min_px or new.mean() > .2:      # 太少是噪点；太多是整个画面在变（转场、甩镜）
            continue
        for bb, cnt in _comps(new, rx, ry, min_px):
            x0, y0, x1, y1 = bb
            nm = new[y0:y1, x0:x1]
            stay = (fin_tm[y0:y1, x0:x1] & _dilate(nm, 6)).sum()
            if stay < .5 * cnt:                        # 没撑过 K 帧，不是字（人在动、背景在变）
                rej("没撑过 K 帧", o, bb, f"{stay}/{cnt}")
                continue
            if any(e["_shot"] == shot_of(o) and (_dilate(e["_mask"][y0:y1, x0:x1], 2) & nm).sum() > .3 * cnt
                   for e in events):
                continue                               # 同一个镜头里已经记过的字（入场还没落位、被挡住又露出来）
            e = measure(o, bb)
            if e:
                e["_shot"] = shot_of(o)
                events.append(e)

    end = round(t0 + n_read / rate, 3)
    for e in events:
        if e.get("清掉") is None:
            e["清掉"] = end
            e["到结尾还在"] = True
        e["停留秒"] = round(e["清掉"] - e["t"], 2)
        e["随切镜清掉"] = bool(cut_list) and any(abs(e["清掉"] - c) <= 3 / rate for c in cut_list)
        for k in ("_mask", "_miss", "_shot", "_bb"):
            e.pop(k, None)
    if words:
        starts = [x["t0"] for x in words]
        for e in events:
            near = min(starts, key=lambda s_: abs(s_ - e["t"]))
            e["最近的词开口"] = round(near, 3)
            e["比开口晚帧数"] = round((e["t"] - near) * rate, 1)
    return {"帧率": round(rate, 3), "分析宽": w, "字": events, "汇总": summarize(events, rate, cut_list),
            "滤掉的候选": dict(rejects), **({"候选明细": detail} if debug else {})}


def summarize(ev, rate, cut_list):
    if not ev:
        return {"说明": "没找到新出现的字。检查 --crop 是否框住了画面，或者字是不是一直在动（这类要人工看逐帧图）。"}
    hs = np.array(sorted(e["字高_1080"] for e in ev))
    small, big, split = hs, np.array([]), None
    if len(hs) >= 4:                                   # 对数字高做二分聚类：小词一堆、重点词一堆
        lg = np.log(hs)
        c = np.array([lg.min(), lg.max()])
        for _ in range(20):
            lab = np.abs(lg[:, None] - c[None]).argmin(1)
            c = np.array([lg[lab == k].mean() if (lab == k).any() else c[k] for k in (0, 1)])
        if np.exp(c[1] - c[0]) >= 1.6 and (lab == 0).any() and (lab == 1).any():
            small, big = hs[lab == 0], hs[lab == 1]
            split = small.max()
    grp = lambda e: "重点词" if split is not None and e["字高_1080"] > split else "小词"

    def entry(es):
        es = [e for e in es if e["首帧放大"]]
        if not es:
            return None
        return {"个数": len(es),
                "首帧放大中位数": round(float(np.median([e["首帧放大"] for e in es])), 3),
                "首帧偏移中位数_1080": round(float(np.median([e["首帧偏移_1080"] for e in es if e["首帧偏移_1080"] is not None] or [0])), 1),
                "落位帧数中位数": float(np.median([e["落位帧数"] for e in es])),
                "落位秒中位数": round(float(np.median([e["落位秒"] for e in es])), 3),
                "首帧放大超过 5% 的比例": round(float(np.mean([e["首帧放大"] > 1.05 for e in es])), 2),
                **({"注意": "字高中位数不到 45 像素，模板匹配误差约 ±10%，入场以证据条为准"}
                   if np.median([e["字高_1080"] for e in es]) < 45 else {})}
    cols = collections.Counter(e["颜色名"] for e in ev)
    # 排版：同一屏（没清掉的）里，后出现的词和前一个词的关系
    one_line, stag, sides = [], [], collections.Counter()
    for a, b in zip(ev, ev[1:]):
        if b["t"] >= a["清掉"]:
            continue
        ay0, ay1, by0, by1 = a["bbox"][1], a["bbox"][3], b["bbox"][1], b["bbox"][3]
        overlap = max(0, min(ay1, by1) - max(ay0, by0)) / max(min(ay1 - ay0, by1 - by0), 1e-6)
        one_line.append(overlap < .3)
        if overlap < .3 and abs(by0 - ay1) < .15:
            stag.append(round(((b["bbox"][0] + b["bbox"][2]) - (a["bbox"][0] + a["bbox"][2])) / 2 * 1080, 1))
    for e in ev:
        cx = (e["bbox"][0] + e["bbox"][2]) / 2
        sides["左" if cx < .42 else "右" if cx > .58 else "中"] += 1
    lags = [e["比开口晚帧数"] for e in ev if "比开口晚帧数" in e]
    edge = [e["字边暗圈"] for e in ev]
    out = {
        "字数": len(ev),
        "出现间隔中位数秒": round(float(np.median(np.diff([e["t"] for e in ev]))), 2) if len(ev) > 1 else None,
        "层级": {"小词字高中位数_1080": round(float(np.median(small)), 1),
                 "重点词字高中位数_1080": round(float(np.median(big)), 1) if len(big) else None,
                 "大小比": round(float(np.median(big) / np.median(small)), 2) if len(big) else None},
        "入场_小词": entry([e for e in ev if grp(e) == "小词"]),
        "入场_重点词": entry([e for e in ev if grp(e) == "重点词"]),
        "颜色": dict(cols),
        "颜色样本": {k: [int(v) for v in np.median([e["颜色"] for e in ev if e["颜色名"] == k], 0)] for k in cols},
        "说明_颜色": "颜色是成片里的样子，带着对标的调色；转参数时按色相取，亮度按自己片子的调色校",
        "描边或阴影": f"{sum(x < .7 for x in edge)}/{len(edge)} 个字边缘有一圈明显变暗",
        "停留": {"停留秒中位数": round(float(np.median([e["停留秒"] for e in ev])), 2),
                 "随切镜清掉的比例": round(float(np.mean([e["随切镜清掉"] for e in ev])), 2) if cut_list else None,
                 "说明": "随切镜清掉比例高 = 字出来后一直叠着，镜头一切整块清掉"},
        "排版": {"同屏相邻两词分行的比例": round(float(np.mean(one_line)), 2) if one_line else None,
                 "换行时的左右错位中位数_1080": float(np.median(stag)) if stag else None,
                 "位置分布（画面左中右）": dict(sides)},
    }
    if lags:
        out["时机"] = {"比开口晚帧数中位数": float(np.median(lags)),
                       "说明": "接近 0 = 词说出口那一帧就出现；明显为负 = 提前出现"}
    return out


# ================================================================ 命令行
def main():
    ap = argparse.ArgumentParser(description="对标拆解")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, rng=False):
        p.add_argument("src")
        p.add_argument("--crop", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
        if rng:
            p.add_argument("--range", nargs=2, type=float, metavar=("T0", "T1"))
        p.add_argument("--out")
    p = sub.add_parser("sheet", help="总览图 / 逐帧图")
    common(p)
    p.add_argument("--at", type=float)
    p.add_argument("--dur", type=float, default=1.0)
    p.add_argument("--fps", type=float)
    p.add_argument("--grid", action="store_true", help="叠 0.1 网格，方便写 measure 的框")
    p.add_argument("--tile", type=int, default=240)
    common(sub.add_parser("cuts", help="切点和镜头时长"))
    common(sub.add_parser("sfx", help="音效和切点的关系"))
    p = sub.add_parser("people", help="人物占比（需要 mediapipe）")
    common(p)
    p.add_argument("--model", required=True)
    p = sub.add_parser("measure", help="逐个测指定的字（扫描漏掉、误判的用这个）")
    common(p)
    p.add_argument("--list", required=True, help='JSON：[{"text": "EXACT", "t": 3.37, "box": [x0, y0, x1, y1]}, ...]')
    p.add_argument("--words")
    p.add_argument("--width", type=int, default=540)
    for name in ("text", "all"):
        p = sub.add_parser(name, help="逐帧测字" if name == "text" else "全部跑一遍，写进 --out 目录")
        common(p, rng=True)
        p.add_argument("--words", help="对标音频的逐字时间轴（transcribe.py 输出），用来测出字时机")
        p.add_argument("--width", type=int, default=540)
        p.add_argument("--model", help="可选：mediapipe 人像模型，给了就测人物占比")
    a = ap.parse_args()
    crop = tuple(a.crop) if a.crop else None
    dump = lambda d: print(json.dumps(d, ensure_ascii=False, indent=1))

    if a.cmd == "sheet":
        out = a.out or os.path.splitext(a.src)[0] + ("_逐帧.jpg" if a.at is not None else "_总览.jpg")
        print(sheet(a.src, out, a.at, a.dur if a.at is not None else None, a.fps, crop=crop, tile=a.tile, grid=a.grid))
    elif a.cmd == "cuts":
        dump(cuts(a.src, crop))
    elif a.cmd == "sfx":
        dump(sfx(a.src, crop=crop))
    elif a.cmd == "people":
        dump(people(a.src, a.model, crop))
    elif a.cmd == "measure":
        with open(a.list, encoding="utf-8") as f:
            todo = json.load(f)
        words = json.load(open(a.words, encoding="utf-8"))["words"] if a.words else None
        out = a.out or os.path.splitext(a.list)[0] + "_测量"
        cl = cuts(a.src, crop)["切点"]
        ev = []
        for i, x in enumerate(todo):
            e = word(a.src, x["t"], x["box"], crop, a.width, words=words, cut_list=cl,
                     evidence=os.path.join(out, f"字{i + 1:02d}_{x.get('text', '')}.jpg"))
            e["text"] = x.get("text")
            ev.append(e)
            print(x.get("text"), {k: e.get(k) for k in ("t", "字高_1080", "首帧放大", "落位帧数", "颜色名", "比开口晚帧数")})
        ok = [e for e in ev if "t" in e]
        res = {"字": ev, "汇总": summarize(ok, info(a.src)["fps"], cl)}
        with open(os.path.join(out, "测量.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        dump(res["汇总"])
        print("完整结果：", os.path.join(out, "测量.json"))
    else:
        words = None
        if a.words:
            with open(a.words, encoding="utf-8") as f:
                words = json.load(f)["words"]
        t0, t1 = a.range if a.range else (0.0, None)
        out = a.out or os.path.splitext(a.src)[0] + "_拆解"
        os.makedirs(out, exist_ok=True)
        c = cuts(a.src, crop)
        res = {"cuts": c} if a.cmd == "all" else {}
        if a.cmd == "all":
            res["sfx"] = sfx(a.src, c["切点"])
            if a.model:
                res["people"] = people(a.src, a.model, crop)
            sheet(a.src, os.path.join(out, "总览.jpg"), crop=crop)
        res["text"] = text(a.src, t0, t1, crop, a.width, words=words, evidence=os.path.join(out, "证据帧"),
                           cut_list=c["切点"])
        with open(os.path.join(out, "拆解.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        dump({k: (v["汇总"] if k == "text" else v) for k, v in res.items()})
        print("完整结果：", os.path.join(out, "拆解.json"))


if __name__ == "__main__":
    main()
