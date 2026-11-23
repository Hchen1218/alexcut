"""时间轴 JSON：整个 skill 的接口。对标报告转出的参数、分镜、两套动效、字幕、配乐都读写这一份。

最小例子（路径相对时间轴文件所在目录）：
{
  "canvas": {"w": 1080, "h": 1920, "fps": 30},
  "sources": {"aroll": {"path": "口播.mov", "scale": 1.5, "offset": 0}},
  "masks": {"aroll": "work/masks"},
  "audio": "口播.mov",
  "banned": ["不能上屏的词"],
  "segments": [
    {"t0": 0, "t1": 7.15, "type": "kinetic", "source": "aroll", "blocks": [...], "words": [...]},
    {"t0": 7.15, "t1": 12, "type": "aroll", "source": "aroll", "zoom": [1.1, 0.5, 0.4]},
    {"t0": 12, "t1": 20, "type": "glass", "base": {"source": "aroll"}, "items": [...]},
    {"t0": 20, "t1": 21.6, "type": "chapter", "source": "aroll", "title": "节奏", "sub": "第一件事"}
  ],
  "subtitles": {"items": [{"t0": 0.1, "t1": 1.3, "text": "我现在所有的视频"}], "hide_in": ["kinetic", "chapter"],
                "style": {"size": 72, "y": 0.672}},
  "music": {...}
}

顶层字段：
- sources.*.offset：时间轴时间 + offset = 原片时间。只用原片中间一段时写它（比如 60），时间轴就从 0 开始。
  遮罩（masks.py --range）按原片时间生成，读取时自动按 offset 换算。
- audio / audio_offset：成片的音轨。audio 和某个 source 是同一个文件时，audio_offset 自动等于那个 source 的 offset；
  不是同一个文件时才需要自己写。
- banned：不能出现在画面上的词（品牌名、客户名、账号……），check 会在字幕、贴脸字、胶囊、章节里查。
- subtitles.style：覆盖 presets/glass.json 里的 subtitle（字体、字号、y、颜色、阴影、max_chars）。

段落类型：
- aroll：原片，可选 zoom [倍数, 中心 x 比例, 中心 y 比例]、flat（略灰略平，表示「还没剪」）。
- kinetic：贴脸逐词字幕运镜，字段见 kinetic.py。start_cam [x, y, 倍数, 角度] 是第一块之前的取景。
- glass：在底图上叠玻璃组件。base 是 {"source": key} 或 {"color": [r,g,b]} 或 {"image": 路径}。
  items：capsule {t, dur, text, x, y, size?}
        / card {t, dur, x, y, w, h, source 或 image, src_start?, crop? [x0,y0,x1,y1] 比例}
          （横屏录屏的素材写 "fit": "native", "scale": 2，再用 crop 裁出要看的部分）
        / glass_text {t, text, x, y, size} / defocus {t, dur, amount}。
- chapter：失焦 + 玻璃大字章节转场，title、sub 可选。
每段都可以写 "style": {...} 覆盖预设里的同名字段，"preset": "文件名" 换整套预设。
每段都可以写 "overlays": [{"module": "fx/xxx.py", "fn": "draw", "t": 秒, "dur": 秒, ...}]，
挂预设外新写的效果，函数签名 draw(fr, t, lt, item, style)，见 references/04 末尾。
"""
import copy
import json
import os

import core

TYPES = {"aroll", "kinetic", "glass", "chapter"}
DEFAULT_PRESET = {"kinetic": "kinetic", "glass": "glass", "chapter": "glass", "aroll": "glass"}


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


_presets = {}


def preset(name):
    if name not in _presets:
        path = name if os.path.isabs(name) else os.path.join(core.PRESET_DIR, name + ".json")
        with open(path, encoding="utf-8") as f:
            _presets[name] = json.load(f)
    return _presets[name]


def style_of(seg):
    return deep_merge(preset(seg.get("preset", DEFAULT_PRESET[seg["type"]])), seg.get("style"))


def load(path):
    with open(path, encoding="utf-8") as f:
        tl = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    tl["_base"] = base

    def fix(p):
        return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(base, os.path.expanduser(p)))

    c = tl.setdefault("canvas", {})
    core.set_canvas(c.get("w", 1080), c.get("h", 1920), c.get("fps", 30))
    for s in tl.get("sources", {}).values():
        s["path"] = fix(s["path"])
    tl["masks"] = {k: fix(v) for k, v in tl.get("masks", {}).items()}
    if tl.get("audio"):
        tl["audio"] = fix(tl["audio"])
        if "audio_offset" not in tl:              # 音轨默认跟着同一个文件的 source 的 offset 走
            same = [s.get("offset", 0.0) for s in tl.get("sources", {}).values()
                    if os.path.realpath(s["path"]) == os.path.realpath(tl["audio"])]
            tl["audio_offset"] = same[0] if same else 0.0
    for seg in tl.get("segments", []):
        for it in seg.get("items", []):
            if it.get("image"):
                it["image"] = fix(it["image"])
        if seg.get("base", {}).get("image"):
            seg["base"]["image"] = fix(seg["base"]["image"])
    validate(tl)
    return tl


def validate(tl):
    errs = []
    segs = tl.get("segments", [])
    if not segs:
        errs.append("segments 为空")
    srcs = tl.get("sources", {})
    for k, s in srcs.items():
        if not os.path.exists(s["path"]):
            errs.append(f"素材 {k} 不存在：{s['path']}")
    prev = None
    for i, seg in enumerate(segs):
        tag = f"第 {i + 1} 段（{seg.get('t0')}–{seg.get('t1')}）"
        if seg.get("type") not in TYPES:
            errs.append(f"{tag} 类型 {seg.get('type')} 不认识，可选 {sorted(TYPES)}")
            continue
        if seg["t1"] <= seg["t0"]:
            errs.append(f"{tag} 结束时间不晚于开始时间")
        if prev is not None and seg["t0"] < prev - 1e-6:
            errs.append(f"{tag} 和上一段重叠（上一段到 {prev}）")
        prev = seg["t1"]
        src = seg.get("source") or seg.get("base", {}).get("source")
        if src and src not in srcs:
            errs.append(f"{tag} 用了不存在的素材 {src}")
        for it in seg.get("items", []):
            if it.get("source") and it["source"] not in srcs:
                errs.append(f"{tag} 的组件用了不存在的素材 {it['source']}")
        if seg["type"] == "kinetic":
            if not seg.get("blocks"):
                errs.append(f"{tag} kinetic 至少要一个镜头块")
            for w in seg.get("words", []):
                if not seg["t0"] <= w["t"] < seg["t1"]:
                    errs.append(f"{tag} 词「{w.get('text')}」的时间 {w['t']} 不在本段内")
    if errs:
        raise SystemExit("时间轴有问题：\n- " + "\n- ".join(errs))


def segment_at(tl, t):
    for seg in tl["segments"]:
        if seg["t0"] <= t < seg["t1"]:
            return seg
    return None


def end_time(tl):
    return max(s["t1"] for s in tl["segments"])


# ---------------------------------------------------------------- 规则检查（不渲染，几秒出结果）
def missing_glyphs(spec, text):
    """字体里没有的字（会画成方块或空白）。"""
    f = core.font(spec, 60)
    ref = f.getmask("\U0010FFFD")
    ref = (ref.size, bytes(ref))
    miss = []
    for ch in dict.fromkeys(text):
        if ch.isspace():
            continue
        m = f.getmask(ch)
        if m.getbbox() is None or (m.size, bytes(m)) == ref:
            miss.append(ch)
    return miss


def texts(tl):
    """时间轴里所有会上屏的文字：(时间, 位置说明, 文字)。"""
    for it in tl.get("subtitles", {}).get("items", []):
        yield it["t0"], "字幕", it["text"]
    for seg in tl.get("segments", []):
        for w in seg.get("words", []):
            yield w["t"], "贴脸字", w["text"]
            for x in w.get("items", []) if w.get("fx") == "flip" else []:
                yield x[0], "翻牌", x[1]
        for it in seg.get("items", []):
            if it.get("text"):
                yield it["t"], it["kind"], it["text"]
        for k in ("title", "sub"):
            if seg.get(k):
                yield seg["t0"], "章节", seg[k]


def lint(tl):
    """把 references 里能自动判的规则跑一遍，返回 [(级别, 说明)]。级别：错（必须改）/ 提醒（看情况）。"""
    out = []
    W, H = core.W, core.H
    k = W / 1080
    x0, x1, y0, y1 = 110 * k, 970 * k, 240 * H / 1920, 1500 * H / 1920
    sub = deep_merge(preset("glass").get("subtitle", {}), tl.get("subtitles", {}).get("style"))
    yc = sub.get("y", .672) * H
    band = (yc - 50 * H / 1920, yc + 50 * H / 1920)
    segs = tl["segments"]
    dur = end_time(tl)
    fmt = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"

    for seg in segs:
        st = style_of(seg)
        if st.get("grain", 0):
            out.append(("提醒", f"{fmt(seg['t0'])} 段开了胶片颗粒 grain={st['grain']}（07：不加颗粒）"))
        vis = []
        for it in seg.get("items", []):
            if it["kind"] not in ("capsule", "card"):
                continue
            if it["kind"] == "card":
                bx = (it["x"] - it["w"] / 2, it["y"] - it["h"] / 2, it["x"] + it["w"] / 2, it["y"] + it["h"] / 2)
            else:
                c = st.get("capsule", {})
                size = it.get("size") or c.get("size", 44)
                hh = size * c.get("height", 2.0) / 2
                ww = core.font(st["font"], size).getlength(it["text"]) / 2 + c.get("pad", 34)
                bx = (it["x"] - ww, it["y"] - hh, it["x"] + ww, it["y"] + hh)
            name = f"{fmt(it['t'])} {it['kind']}「{it.get('text', it.get('source', ''))}」"
            if bx[0] < x0 - 1 or bx[2] > x1 + 1 or bx[1] < y0 - 1 or bx[3] > y1 + 1:
                out.append(("错", f"{name} 出了安全区 x 110–970 / y 240–1500：{tuple(int(v) for v in bx)}"))
            if bx[1] < band[1] and bx[3] > band[0]:
                out.append(("错", f"{name} 压进字幕带 y {int(band[0])}–{int(band[1])}"))
            vis.append((it["kind"], it["t"], it["t"] + it.get("dur", seg["t1"] - it["t"]), name))
        for kind in ("capsule", "card"):                # 讲课版式 = 一张卡片 + 一个标签胶囊，算一组
            v = sorted(x[1:] for x in vis if x[0] == kind)
            for a, b in zip(v, v[1:]):
                if b[0] < a[1] - .05:
                    out.append(("提醒", f"{b[2]} 和 {a[2]} 同时在屏上（03：同屏最多一张卡片加一个胶囊）"))
        if seg["type"] == "glass" and any(it["kind"] == "card" for it in seg.get("items", [])):
            if seg["t1"] - seg["t0"] < 4:
                out.append(("提醒", f"{fmt(seg['t0'])} 讲课版式只停了 {seg['t1'] - seg['t0']:.1f} 秒（01：至少 4 秒）"))
        if seg["type"] == "kinetic":
            bl = sorted(b["t0"] for b in seg.get("blocks", [])) + [seg["t1"]]
            for i in range(len(bl) - 1):
                n = [w for w in seg.get("words", []) if bl[i] <= w["t"] < bl[i + 1] and w.get("color") == "accent"]
                if len(n) > 1:
                    out.append(("提醒", f"{fmt(bl[i])} 这一块有 {len(n)} 个强调色词（04：每块最多 1 个）"))
            for w in seg.get("words", []):
                if w.get("kind") == "s" and w.get("size", 58) < st["layout"].get("min_small", 50):
                    out.append(("错", f"{fmt(w['t'])} 小词「{w['text']}」只有 {w['size']} 像素（04：不小于 50）"))
                fx = w.get("fx")
                spec = st["fonts"]["zh"] if fx == "zh" else st["fonts"]["big" if w.get("kind") == "b" else "small"]
                miss = missing_glyphs(spec, w["text"])
                if miss:
                    out.append(("错", f"{fmt(w['t'])} 贴脸字「{w['text']}」里的「{''.join(miss)}」字体里没有，会画成方块。"
                                     f"中文贴脸字要把 style.fonts.big / small 换成中文字体（例如 SmileySans-Oblique.ttf），见 04 第 2 节"))
            out += _kinetic_layout(tl, seg, st, fmt, x0, x1, y0, y1, band)
        if seg["type"] == "chapter" and seg["t1"] - seg["t0"] < 1.2:
            out.append(("提醒", f"{fmt(seg['t0'])} 章节转场只有 {seg['t1'] - seg['t0']:.1f} 秒，玻璃字来不及看清（03：至少 1.2 秒）"))
        for it in seg.get("items", []):
            if it["kind"] == "capsule" and len(it["text"].replace(" ", "")) > 10:
                out.append(("提醒", f"{fmt(it['t'])} 胶囊「{it['text']}」超过 10 个字（03：一行说完）"))

    mx = sub.get("max_chars", 12)
    for it in tl.get("subtitles", {}).get("items", []):
        n = len(it["text"].replace(" ", ""))
        if n > mx + 2:
            out.append(("提醒", f"{fmt(it['t0'])} 字幕「{it['text']}」{n} 个字（07：超过约 {mx} 字拆成两条）"))

    for word in tl.get("banned", []):
        for t, where, txt in texts(tl):
            if word.lower() in txt.lower():
                out.append(("错", f"{fmt(t)} {where}「{txt}」里有不能出现的词「{word}」"))

    if tl.get("audio") and tl.get("audio_offset", 0) == 0:
        offs = {k: s.get("offset", 0) for k, s in tl.get("sources", {}).items() if s.get("offset")}
        same = [os.path.realpath(s["path"]) for s in tl.get("sources", {}).values()]
        if offs and os.path.realpath(tl["audio"]) not in same:
            out.append(("提醒", f"画面截了原片的一段（{offs}），音轨从 0 秒开始。音轨和画面不是同一个文件时，要写 audio_offset"))

    ev = set()
    for seg in segs:
        if seg["t0"] < 3:
            ev.add(round(seg["t0"], 1))
        for b in seg.get("blocks", []):
            ev.add(round(b["t0"], 1))
        for w in seg.get("words", []):
            ev.add(round(w["t"], 1))
        for it in seg.get("items", []):
            ev.add(round(it["t"], 1))
    n3 = sum(t < 3 for t in ev)
    if segs and segs[0]["t0"] == 0 and n3 < 4:
        out.append(("提醒", f"前 3 秒只有 {n3} 个视觉事件（01：至少 4 个：字、镜头动作、元素）"))

    if dur < 60:
        out.append(("提醒", f"片长 {dur:.0f} 秒，不到 60 秒，节奏类检查（人物占比、中段露脸时长、章节数、前后密度）没有跑"))
    if dur >= 60:
        a_kinds = ("kinetic", "aroll")
        share = sum(s["t1"] - s["t0"] for s in segs if s["type"] in a_kinds) / dur
        if not .2 <= share <= .45:
            out.append(("提醒", f"人物全屏占 {share:.0%}（01：建议 25–35%）"))
        for s in segs:
            if s["type"] in a_kinds and .24 * dur <= s["t0"] < .83 * dur and s["t1"] - s["t0"] > 5.5:
                out.append(("提醒", f"{fmt(s['t0'])} 中段人物全屏连续 {s['t1'] - s['t0']:.1f} 秒（01：每次不超过 5 秒）"))
        nch = sum(s["type"] == "chapter" for s in segs)
        if not 3 <= nch <= 7:
            out.append(("提醒", f"章节转场 {nch} 个（01：4–6 个）"))
        front = sum(s["t0"] < dur / 3 for s in segs)
        back = sum(s["t0"] >= dur * 2 / 3 for s in segs)
        if back and front < 1.5 * back:
            out.append(("提醒", f"前三分之一 {front} 段，后三分之一 {back} 段（01：开头要明显更密）"))
    return out


def _kinetic_layout(tl, seg, st, fmt, x0, x1, y0, y1, band):
    """按渲染时的同一套算法摆一遍贴脸字：字被脸挤小、压脸、出安全区、压字幕带、同块里叠字、藏进头发太多。"""
    from kinetic import Kinetic, fit_cam, to_screen
    from masks import Masks
    src = seg.get("source")
    off = tl.get("sources", {}).get(src, {}).get("offset", 0.0)
    k = Kinetic(seg, None, Masks(tl.get("masks", {}).get(src), off), st)
    hide = seg["type"] in tl.get("subtitles", {}).get("hide_in", ["kinetic", "chapter"])
    min_big = st["layout"].get("min_big", 110)
    out, boxes = [], []

    def room_hint(w, land):
        cx, cy, z, roll = k.camera(land)
        lo = fit_cam(-1e4, cy, z, roll)[0]
        hi = fit_cam(1e4, cy, z, roll)[0]
        if w["side"] == "L":
            more = cx - lo
            tip = "cam 的 x 往小改（往左），脸会被推到右边" + (f"，这一块还能再挪 {int(more)}" if more > 5 else "，已经挪到头了")
        else:
            more = hi - cx
            tip = "cam 的 x 往大改（往右），脸会被推到左边" + (f"，这一块还能再挪 {int(more)}" if more > 5 else "，已经挪到头了")
        return tip + f"；放大倍数越大，能挪的越多（现在 {z:.2f}）；或者换短一点的词、改 side: \"T\" 放头顶"

    for i, w in enumerate(k.W):
        edge, fit = k.placement(i)
        land = k.land_of(w["t"])
        name = f"{fmt(w['t'])} 贴脸字「{w['text']}」"
        if w["kind"] == "b" and w["side"] in ("L", "R") and (fit < .6 * w["size"] or fit < min_big):
            out.append(("错", f"{name} 头旁放不下，实际只有 {fit} 像素（要 {w['size']}，重点词至少 {min_big}）。{room_hint(w, land)}"))
            continue
        iw = k.ink_w(w, fit)
        spec = st["fonts"]["zh"] if w["fx"] == "zh" else st["fonts"]["big" if w["kind"] == "b" else "small"]
        bb = core.font(spec, fit).getbbox(w["text"])
        ih = bb[3] - bb[1]
        a, b = (edge - iw, edge) if w["side"] == "L" else (edge, edge + iw) if w["side"] == "R" else (edge - iw / 2, edge + iw / 2)
        y = w.get("y", 0)
        ya, yb = y - ih / 2, y + ih / 2
        if iw > x1 - x0 or ya < y0 - 2 or yb > y1 + 2:
            out.append(("错", f"{name} 出了安全区：宽 {int(iw)}，y {int(ya)}–{int(yb)}"))
            continue
        if not hide and yb > band[0] and ya < band[1]:
            out.append(("错", f"{name} 压进字幕带"))
        sh = max(x0 - a, 0) - max(b - x1, 0)          # 渲染时会被安全边推回来的距离（正 = 往右推）
        a, b = a + sh, b + sh
        xl, xr, _ = k.head_screen(land, k.camera(land))
        if w["side"] in ("L", "R") and abs(sh) > 1 and w["text"] not in k.behind:
            over = (b - xl) if w["side"] == "L" else (xr - a)
            if over > 30:
                out.append(("提醒", f"{name} 头旁放不下，被安全边推回来约 {int(abs(sh))} 像素，会压到脸 {int(over)} 像素。{room_hint(w, land)}"))
        if w["text"] in k.behind and k.masks.dir:         # 藏进头发：头顶以下的部分太多就读不出来
            hx, top = k.masks.head(land)
            top_s = to_screen((hx, top), k.camera(land))[1]
            hidden = (yb - top_s) / max(ih, 1)
            if hidden > .15:
                out.append(("提醒", f"{name} 放进 behind 后约 {hidden:.0%} 的字高在头顶线以下，可能缺字母读不出来。"
                                   f"往上挪 {int((hidden - .1) * ih)} 像素，或者不放 behind"))
        boxes.append((k.block_of(w["t"] + 1e-4), w, (a, ya, b, yb), name))

    for i, (bk, w, r, name) in enumerate(boxes):          # 同一屏里字和字叠在一起
        for bk2, w2, r2, name2 in boxes[i + 1:]:
            if bk2 != bk:
                continue
            ox = min(r[2], r2[2]) - max(r[0], r2[0])
            oy = min(r[3], r2[3]) - max(r[1], r2[1])
            if ox > 6 and oy > .15 * min(r[3] - r[1], r2[3] - r2[1]):
                out.append(("错", f"{name2} 和「{w['text']}」叠在一起（约 {int(ox)}×{int(oy)} 像素）。拉开 y，或者换 side"))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "check":
        raise SystemExit("用法：python3 timeline.py check timeline.json")
    res = lint(load(sys.argv[2]))
    for lv, msg in res:
        print(("❌ " if lv == "错" else "⚠️ ") + msg)
    bad = sum(lv == "错" for lv, _ in res)
    print(f"\n{bad} 个必须改，{len(res) - bad} 个提醒。" if res else "规则检查全部通过。")
    sys.exit(1 if bad else 0)

