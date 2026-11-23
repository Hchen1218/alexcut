"""按时间轴渲染：静帧、片段、全片。

用法：
  python3 compose.py timeline.json --stills "1.0 3.2 8.5"        # 先看静帧（几秒钟出结果）
      加 --guides 画安全线和字幕带，加 --sheet 把这几张拼成一张手机尺寸的总览图
  python3 compose.py timeline.json --range 0 30 --out 样片.mp4    # 再出样片
  python3 compose.py timeline.json --out 成片.mp4                 # 最后出全片
改了某几段，只用 --range 重渲那几段，再用 ffmpeg concat 拼回去，不要整片重来。
"""
import argparse
import importlib.util
import os
import sys

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import glass as G
import timeline as TL
from kinetic import Kinetic
from masks import Masks


class Composer:
    def __init__(self, tl):
        self.tl = tl
        self.videos, self.masks, self.kin, self._items, self._fx = {}, {}, {}, {}, {}
        self.sub_style = TL.deep_merge(TL.preset("glass").get("subtitle", {}), tl.get("subtitles", {}).get("style"))

    # ------------------------------------------------------------ 素材
    def video(self, key):
        if key not in self.videos:
            s = self.tl["sources"][key]
            self.videos[key] = core.Video(s["path"], s.get("scale", 1.0), s.get("offset", 0.0), s.get("fit", "cover"))
        return self.videos[key]

    def item_video(self, seg_i, it_i, key):
        """卡片里的素材单独解码，避免和底图抢同一个解码器来回跳。"""
        k = (seg_i, it_i)
        if k not in self._items:
            s = self.tl["sources"][key]
            self._items[k] = core.Video(s["path"], s.get("scale", 1.0), s.get("offset", 0.0), s.get("fit", "cover"))
        return self._items[k]

    def mask(self, key):
        if key not in self.masks:
            off = self.tl["sources"].get(key, {}).get("offset", 0.0)
            self.masks[key] = Masks(self.tl["masks"].get(key), off)
        return self.masks[key]

    def kinetic(self, i, seg):
        if i not in self.kin:
            st = TL.style_of(seg)
            self.kin[i] = Kinetic(seg, self.video(seg["source"]), self.mask(seg["source"]), st)
        return self.kin[i]

    def base(self, seg, t):
        b = seg.get("base") or {"source": seg.get("source")}
        if b.get("source"):
            fr = self.video(b["source"]).at(t)
            z = seg.get("zoom") or b.get("zoom")
            fr = core.zoomed(fr, *(z or (1.0,)))
        elif b.get("image"):
            fr = core.memo(("img", b["image"]), lambda: Image.open(b["image"]).convert("RGB")).copy()
            fr = core.zoomed(fr, 1.0)
        else:
            fr = Image.new("RGB", (core.W, core.H), tuple(b.get("color", (18, 18, 22))))
        if seg.get("flat") or b.get("flat"):
            fr = ImageEnhance.Contrast(ImageEnhance.Color(fr).enhance(.72)).enhance(.88)
        return fr

    # ------------------------------------------------------------ 一帧
    def frame(self, t):
        segs = self.tl["segments"]
        i = next((k for k, s in enumerate(segs) if s["t0"] <= t < s["t1"]), None)
        if i is None:
            return Image.new("RGB", (core.W, core.H))
        seg = segs[i]
        st = TL.style_of(seg)
        kind = seg["type"]
        if kind == "kinetic":
            fr = self.kinetic(i, seg).frame(t)
        elif kind == "chapter":
            fr = G.chapter(self.base(seg, t), t, seg["t0"], seg["t1"], seg["title"], st, seg.get("sub"))
        else:
            fr = self.base(seg, t)
            if kind == "glass":
                fr = self.glass_items(fr, t, i, seg, st)
        fr = self.overlays(fr, t, seg, st)
        fr = core.grain(fr, t, st.get("grain", 0))
        self.subtitle(fr, t, kind, getattr(self, "_busy", []) if kind == "glass" else [])
        return fr

    def glass_items(self, fr, t, seg_i, seg, st):
        self._busy = []                                # 这一帧被玻璃组件占住的纵向范围，字幕要让开
        for j, it in enumerate(seg.get("items", [])):
            lt = t - it["t"]
            dur = it.get("dur", seg["t1"] - it["t"])
            if lt <= 0 or lt > dur:
                continue
            k = it["kind"]
            if k == "capsule":
                G.capsule(fr, it["x"], it["y"], it["text"], lt, dur, st, size=it.get("size"))
                hh = (it.get("size") or st["capsule"]["size"]) * st["capsule"]["height"] / 2
                self._busy.append((it["y"] - hh, it["y"] + hh))
            elif k == "card":
                if it.get("source"):
                    tt = it["src_start"] + lt if "src_start" in it else t
                    im = self.item_video(seg_i, j, it["source"]).at(tt)
                else:
                    im = core.memo(("img", it["image"]), lambda: Image.open(it["image"]).convert("RGB"))
                if it.get("crop"):                     # 按比例裁出要看的那一块，例如录屏里的对话栏
                    x0, y0, x1, y1 = it["crop"]
                    im = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
                G.card(fr, im, it["x"], it["y"], it["w"], it["h"], lt, dur, st)
                self._busy.append((it["y"] - it["h"] / 2, it["y"] + it["h"] / 2))
            elif k == "glass_text":
                G.glass_text(fr, it["text"], it["x"], it["y"], it.get("size", 180), lt, st)
            elif k == "defocus":
                e = core.ease(lt / .25) * core.ease((dur - lt) / .3)
                fr = G.defocus(fr, e * it.get("amount", 1.0))
        return fr

    # ------------------------------------------------------------ 预设外的效果
    def plugin(self, module, fn):
        """加载项目里新写的效果：module 是 .py 路径（相对时间轴），fn 是里面的函数名。"""
        key = (module, fn)
        if key not in self._fx:
            path = module if os.path.isabs(module) else os.path.join(self.tl["_base"], module)
            spec = importlib.util.spec_from_file_location("fx_" + str(len(self._fx)), path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._fx[key] = getattr(mod, fn)
        return self._fx[key]

    def overlays(self, fr, t, seg, st):
        """段落里的 overlays：[{"module": "fx/letters.py", "fn": "draw", "t": 3.2, "dur": 1.5, ...}]。
        函数签名 draw(fr, t, lt, item, style)，返回新帧或 None（None 表示原地画在 fr 上）。"""
        for ov in seg.get("overlays", []):
            lt = t - ov.get("t", seg["t0"])
            if lt < 0 or lt > ov.get("dur", seg["t1"] - ov.get("t", seg["t0"])):
                continue
            out = self.plugin(ov["module"], ov.get("fn", "draw"))(fr, t, lt, ov, st)
            fr = out if out is not None else fr
        return fr

    # ------------------------------------------------------------ 底部字幕
    def subtitle(self, fr, t, kind, busy=()):
        sub = self.tl.get("subtitles")
        if not sub or kind in sub.get("hide_in", ["kinetic", "chapter"]):
            return
        for it in sub.get("items", []):
            if it["t0"] <= t < it["t1"]:
                s = self.sub_style
                im = core.memo(("sub", it["text"], id(s)), lambda: self._sub_img(it["text"], s))
                y = s.get("y", .672) * core.H
                for a, b in sorted(busy):                # 和玻璃组件撞了就挪到它下面
                    if a - im.height / 2 < y < b + im.height / 2:
                        y = b + im.height / 2 + 8
                fr.paste(im, (int(core.W / 2 - im.width / 2), int(y - im.height / 2)), im)
                return

    @staticmethod
    def _sub_img(text, s):
        f = core.font(s["font"], s.get("size", 72))
        size = s.get("size", 72)
        w, h = int(f.getlength(text)) + 80, int(size * 1.9)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((w / 2, h / 2 + 2), text, font=f, fill=(0, 0, 0, s.get("shadow", 150)), anchor="mm")
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(max(4, size * .09))))
        ImageDraw.Draw(im).text((w / 2, h / 2), text, font=f, fill=tuple(s.get("color", (255, 255, 255))) + (240,),
                                anchor="mm")
        return im

    def close(self):
        for v in list(self.videos.values()) + list(self._items.values()):
            v.close()


def guides(fr, sub_style):
    """安全区：x 110–970，y 240–1500；字幕带画成半透明条。"""
    W, H = core.W, core.H
    k = W / 1080
    ov = Image.new("RGBA", fr.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    yc = sub_style.get("y", .672) * H
    hh = sub_style.get("size", 72) * .7
    d.rectangle((0, yc - hh, W, yc + hh), fill=(255, 210, 0, 60))
    for x in (110 * k, 970 * k):
        d.line((x, 0, x, H), fill=(255, 40, 40, 230), width=3)
    for y in (240 * H / 1920, 1500 * H / 1920):
        d.line((0, y, W, y), fill=(255, 40, 40, 230), width=3)
    fr.paste(ov, (0, 0), ov)


def sheet(shots, out, cols=4, w=360):
    """静帧总览：每张缩到手机上大致看到的大小，下面标时间。"""
    h = int(w * core.H / core.W)
    rows = (len(shots) + cols - 1) // cols
    im = Image.new("RGB", (cols * (w + 12) + 12, rows * (h + 46) + 12), (24, 24, 28))
    d = ImageDraw.Draw(im)
    f = core.font("SmileySans-Oblique.ttf", 28)
    for i, (t, fr) in enumerate(shots):
        x, y = 12 + (i % cols) * (w + 12), 12 + (i // cols) * (h + 46)
        im.paste(fr.resize((w, h), Image.LANCZOS), (x, y))
        d.text((x + 4, y + h + 6), f"{int(t // 60)}:{t % 60:05.2f}", font=f, fill=(235, 235, 235))
    im.save(out, quality=90)
    return out


def main():
    ap = argparse.ArgumentParser(description="按时间轴渲染")
    ap.add_argument("timeline")
    ap.add_argument("--stills", help="空格分隔的时间点，出 jpg 静帧")
    ap.add_argument("--dir", help="静帧目录，默认 <时间轴目录>/stills")
    ap.add_argument("--range", nargs=2, type=float, metavar=("T0", "T1"))
    ap.add_argument("--out", help="输出 mp4，默认 <时间轴目录>/output/成片.mp4")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--guides", action="store_true", help="静帧上画安全区红线和字幕带")
    ap.add_argument("--sheet", action="store_true", help="静帧拼成一张总览图 sheet.jpg")
    a = ap.parse_args()
    tl = TL.load(a.timeline)
    c = Composer(tl)
    try:
        if a.stills:
            d = a.dir or os.path.join(tl["_base"], "stills")
            os.makedirs(d, exist_ok=True)
            shots = []
            for s in a.stills.split():
                p = os.path.join(d, f"t{float(s):07.2f}.jpg")
                fr = c.frame(float(s))
                if a.guides:
                    guides(fr, c.sub_style)
                fr.save(p, quality=92)
                shots.append((float(s), fr))
                print(p)
            if a.sheet:
                print(sheet(shots, os.path.join(d, "sheet.jpg")))
            return
        t0, t1 = a.range if a.range else (0.0, TL.end_time(tl))
        out = a.out or os.path.join(tl["_base"], "output", "成片.mp4")
        audio = None if a.no_audio else tl.get("audio")
        print(core.encode(c.frame, t0, t1, out, audio, tl.get("audio_offset", 0.0)))
    finally:
        c.close()


if __name__ == "__main__":
    main()
