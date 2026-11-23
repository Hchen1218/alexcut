"""alexcut 公共件：画布、缓动与弹簧、字体、亚像素摆放、解码和编码。

其他模块都通过 `core.W / core.H / core.FPS` 读画布尺寸，由 timeline 在加载时设置。
"""
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageFont

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(SKILL, "assets", "fonts")
PRESET_DIR = os.path.join(SKILL, "presets")

W, H, FPS = 1080, 1920, 30


def set_canvas(w, h, fps):
    global W, H, FPS
    W, H, FPS = int(w), int(h), int(fps)


def tool(name):
    """找 ffmpeg / ffprobe / whisper-cli。"""
    p = shutil.which(name)
    if p:
        return p
    for d in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        q = os.path.join(d, name)
        if os.path.exists(q):
            return q
    raise SystemExit(f"找不到 {name}。先运行 python3 scripts/check_env.py 看安装方法。")


def mediapipe():
    """导入 mediapipe。它的绘图工具会 import cv2 / matplotlib，分割用不到，没装就用空模块顶上。"""
    import importlib
    import sys
    import types
    for m in ("cv2", "matplotlib", "matplotlib.pyplot"):
        try:
            importlib.import_module(m)
        except Exception:
            sys.modules.setdefault(m, types.ModuleType(m))
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    return mp, vision, BaseOptions


# ---------------------------------------------------------------- 缓动与弹簧
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, p):
    return a + (b - a) * p


def ease(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_io(p):
    p = clamp(p)
    return 4 * p ** 3 if p < .5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out5(p):
    p = clamp(p)
    return 1 - (1 - p) ** 5


def sp(t, f=2.0, z=.62):
    """欠阻尼弹簧的阶跃响应。t 秒，f 频率（赫兹），z 阻尼比。"""
    if t <= 0:
        return 0.0
    w = 2 * math.pi * f
    if z >= 1:
        return 1 - math.exp(-w * t) * (1 + w * t)
    wd = w * math.sqrt(1 - z * z)
    return 1 - math.exp(-z * w * t) * (math.cos(wd * t) + z * w / wd * math.sin(wd * t))


# ---------------------------------------------------------------- 缓存
_cache = {}


def memo(key, fn):
    if key not in _cache:
        if len(_cache) > 3000:
            _cache.clear()
        _cache[key] = fn()
    return _cache[key]


# ---------------------------------------------------------------- 字体
_fonts = {}


def font(spec, size):
    """spec 写法：文件名（在 assets/fonts 或系统字体目录里找）、绝对路径，
    可变字体用「文件名#字重」，例如 Montserrat-VF.ttf#SemiBold。"""
    size = max(int(size), 1)
    key = (spec, size)
    if key in _fonts:
        return _fonts[key]
    name, _, var = spec.partition("#")
    path = name if os.path.isabs(name) else None
    if path is None:
        for d in (FONT_DIR, os.path.expanduser("~/Library/Fonts"), "/Library/Fonts", "/System/Library/Fonts",
                  "/usr/share/fonts", os.path.expanduser("~/.fonts")):
            q = os.path.join(d, name)
            if os.path.exists(q):
                path = q
                break
    if path is None:
        raise SystemExit(f"找不到字体 {name}：放进 {FONT_DIR}，或在预设里写绝对路径。")
    f = ImageFont.truetype(path, size)
    if var:
        f.set_variation_by_name(var.encode())
    _fonts[key] = f
    return f


# ---------------------------------------------------------------- 摆放
def with_alpha(im, a):
    if a >= 1:
        return im
    r = im.copy()
    r.putalpha(r.getchannel("A").point(lambda v: int(v * a)))
    return r


def place(fr, im, cx, cy, s=1.0, rot=0.0, alpha=1.0):
    """亚像素摆放 RGBA 图（中心 cx, cy），避免慢速移动时一格一格跳。"""
    if s <= .004 or alpha <= 0:
        return
    w, h = im.size
    c, n = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    ow, oh = int(w * s * (abs(c) + abs(n))) + 6, int(h * s * (abs(c) + abs(n))) + 6
    ox, oy = int(cx - ow / 2), int(cy - oh / 2)
    a, b = c / s, n / s
    tx, ty = cx - ox, cy - oy
    coef = (a, b, w / 2 - a * tx - b * ty, -b, a, h / 2 + b * tx - a * ty)
    if s < .6:
        small = (max(1, int(w * s * 1.5)), max(1, int(h * s * 1.5)))
        pm = im.convert("RGBa").resize(small, Image.BICUBIC)
        k = small[0] / w
        coef = tuple(v * k for v in coef)
    else:
        pm = im.convert("RGBa")
    t = pm.transform((ow, oh), Image.AFFINE, coef, Image.BICUBIC).convert("RGBA")
    if alpha < 1:
        t.putalpha(t.getchannel("A").point(lambda v: int(v * alpha)))
    fr.paste(t, (ox, oy), t)


def zoomed(im, z, cx=.5, cy=.5):
    """按画面比例位置 (cx, cy) 放大 z 倍，输出画布尺寸。"""
    sw, sh = im.size
    if abs(z - 1) < 1e-4 and (sw, sh) == (W, H):
        return im.copy()
    w, h = sw / z, sh / z
    x = clamp(cx * sw - w / 2, 0, sw - w)
    y = clamp(cy * sh - h / 2, 0, sh - h)
    return im.resize((W, H), Image.BICUBIC, box=(x, y, x + w, y + h))


# ---------------------------------------------------------------- 解码
def probe(path):
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "default=nw=1", path],
                         capture_output=True, text=True, check=True).stdout
    d = dict(line.split("=", 1) for line in out.split() if "=" in line)
    num, _, den = d.get("r_frame_rate", "30/1").partition("/")
    return dict(w=int(d["width"]), h=int(d["height"]), fps=float(num) / float(den or 1),
                duration=float(d.get("duration", 0) or 0))


class Video:
    """按时间取帧，铺满画布（等比放大后居中裁切）。

    scale > 1 时按更高分辨率解码（例如 4K 原片用 1.5），推镜放大时直接取高清像素。
    小幅前跳顺读丢帧，回跳或大跳才重新定位；越过结尾时停在最后一帧。
    """

    def __init__(self, path, scale=1.0, offset=0.0, fit="cover"):
        self.path, self.offset = path, offset
        w, h = int(round(W * scale / 2) * 2), int(round(H * scale / 2) * 2)
        if fit == "native":          # 保持原片比例，宽度 = 画布宽 × scale；给卡片里的横屏录屏用
            info = probe(path)
            h = int(round(info["h"] * w / info["w"] / 2) * 2)
        self.size = (w, h)
        if fit == "native":
            self.vf = f"scale={w}:{h}:flags=lanczos"
        elif fit == "contain":
            self.vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
        else:
            self.vf = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}"
        self.p, self.next_n, self.last, self.good, self.eof = None, None, None, None, False

    def _open(self, n):
        self.close()
        cmd = [tool("ffmpeg"), "-v", "error", "-ss", f"{max(n / FPS, 0):.4f}", "-i", self.path, "-an",
               "-vf", self.vf, "-r", str(FPS), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        self.next_n, self.eof = n, False

    def _read(self):
        n = self.size[0] * self.size[1] * 3
        b = b"" if self.eof else self.p.stdout.read(n)
        self.next_n += 1
        if len(b) == n:
            self.good = Image.frombuffer("RGB", self.size, b, "raw", "RGB", 0, 1)
            return self.good
        self.eof = True
        return self.good if self.good is not None else Image.new("RGB", self.size)

    def at(self, t):
        """t 是时间轴时间；实际取原片 t + offset。"""
        n = max(int(round((t + self.offset) * FPS)), 0)
        if self.last and self.last[0] == n:
            return self.last[1]
        if self.p is None or n < self.next_n or n - self.next_n > 40:
            self._open(n)
        while self.next_n < n:
            self._read()
        im = self._read()
        self.last = (n, im)
        return im

    def close(self):
        if self.p:
            self.p.stdout.close()
            self.p.kill()
            self.p.wait()
            self.p = None


# ---------------------------------------------------------------- 编码
def encode(frame_fn, t0, t1, out, audio=None, audio_offset=0.0, crf=14):
    """逐帧渲染 [t0, t1) 并编码成 mp4。audio 是音频来源文件（可选），按同一段时间截取。"""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-ss", f"{t0 + audio_offset:.3f}", "-t", f"{t1 - t0:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "256k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    for i, n in enumerate(range(n0, n1)):
        enc.stdin.write(frame_fn(n / FPS).convert("RGB").tobytes())
        if i % (FPS * 5) == 0:
            print(f"  {n / FPS:7.2f}s / {t1:.2f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    return out


# ---------------------------------------------------------------- 质感
_grain = None


def grain(fr, t, amount=4.2):
    """很轻的胶片颗粒，让字和画面融成一个质感。"""
    global _grain
    if amount <= 0:
        return fr
    if _grain is None or _grain[0].shape != (H // 2, W // 2):
        rng = np.random.default_rng(3)
        _grain = [rng.standard_normal((H // 2, W // 2)).astype(np.float32) for _ in range(8)]
    g = _grain[int(t * FPS) % len(_grain)] * amount
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[:H, :W, None]
    arr = np.asarray(fr).astype(np.float32) + g
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
