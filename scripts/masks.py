"""人像遮罩：让字躲到人身后、让贴脸字知道脸在哪。

- 生成：mediapipe 人像分割（可选依赖），按帧存成 PNG：<dir>/fNNNN.png，NNNN 是原片帧号。
- 读取：Masks(dir) 提供 at(t) 遮罩、head(t) 头的位置、face_band(t) 脸左右边。
  没有遮罩时退回「脸在画面中间」的估计，贴脸字照样能排，只是没有遮挡效果。

生成用法：
  python3 masks.py 原片.mov work/masks --model selfie_multiclass_256x256.tflite --range 0 30
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


class Masks:
    def __init__(self, folder=None, offset=0.0):
        """folder 为 None 时不做遮挡。offset：时间轴时间 + offset = 原片时间。"""
        self.dir = folder if folder and os.path.isdir(folder) else None
        self.offset = offset
        self._c = {}
        self.k = None          # 画布像素 / 遮罩像素

    def small(self, n):
        if self.dir is None:
            return None
        if n not in self._c:
            p = os.path.join(self.dir, f"f{n:04d}.png")
            a = np.asarray(Image.open(p).convert("L")) if os.path.exists(p) else None
            if a is not None and self.k is None:
                self.k = core.W / a.shape[1]
            self._c[n] = a
            if len(self._c) > 60:
                self._c.pop(next(iter(self._c)))
        return self._c[n]

    def frame_no(self, t):
        return int(round((t + self.offset) * core.FPS))

    def at(self, t):
        a = self.small(self.frame_no(t))
        return None if a is None else Image.fromarray(a).resize((core.W, core.H), Image.BILINEAR)

    def head(self, t):
        """(头中心 x, 头顶 y)，画布坐标；取前后 3 帧平均防抖。"""
        n = self.frame_no(t)
        pts = []
        for j in range(n - 3, n + 4):
            a = self.small(j)
            if a is None:
                continue
            rows = np.where(a.max(1) > 128)[0]
            if not len(rows):
                continue
            top = rows[0]
            band = a[top:top + int(110 / self.k * 2)] > 128
            pts.append((np.nonzero(band)[1].mean() * self.k, top * self.k))
        return tuple(np.mean(pts, 0)) if pts else (core.W / 2, core.H * .16)

    def face_band(self, t, head_w=470):
        """脸（含头发、眼镜）的左右边，画布坐标。只看头顶往下一小段，手和身体不算。"""
        hx, top = self.head(t)
        a = self.small(self.frame_no(t))
        if a is not None:
            k = self.k
            band = a[int((top + 40) / k):int((top + 220) / k)] > 100
            cols = np.where(band.mean(0) > .25)[0] if band.size else []
            if len(cols):
                return cols[0] * k - 26, cols[-1] * k + 26, hx, top
        return hx - head_w / 2, hx + head_w / 2, hx, top


def generate(src, out, model, t0=0.0, t1=None, size=(540, 960)):
    """逐帧分割，存遮罩。需要 pip install mediapipe。"""
    try:
        mp, vision, BaseOptions = core.mediapipe()
    except ImportError:
        raise SystemExit("没装 mediapipe：pip install mediapipe。不装也能用，只是字不会躲到人身后。")
    os.makedirs(out, exist_ok=True)
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model), output_confidence_masks=True, output_category_mask=False))
    W, H, FPS = core.W, core.H, core.FPS
    t1 = t1 if t1 is not None else core.probe(src)["duration"]
    n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    p = subprocess.Popen([core.tool("ffmpeg"), "-v", "error", "-ss", f"{n0 / FPS:.3f}", "-i", src, "-an",
                          "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}",
                          "-r", str(FPS), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    prev = None
    for n in range(n0, n1):
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
        m = 1 - r.confidence_masks[0].numpy_view().squeeze()      # 第 0 类是背景
        m = np.clip((m - .3) / .4, 0, 1)                           # 边缘收紧
        m = m * m * (3 - 2 * m)
        if prev is not None:                                       # 时间平滑，减少边缘闪烁
            m = .7 * m + .3 * prev
        prev = m
        im = Image.fromarray((m * 255).astype(np.uint8)).resize(size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(.8))
        im.save(os.path.join(out, f"f{n:04d}.png"))
        if (n - n0) % (FPS * 5) == 0:
            print(f"  {n / FPS:.1f}s", flush=True)
    p.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成人像遮罩")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--model", required=True, help="mediapipe selfie_multiclass_256x256.tflite 的路径")
    ap.add_argument("--range", nargs=2, type=float, metavar=("T0", "T1"))
    ap.add_argument("--canvas", default="1080x1920x30", help="宽x高x帧率，要和时间轴一致")
    a = ap.parse_args()
    w, h, fps = (int(v) for v in a.canvas.split("x"))
    core.set_canvas(w, h, fps)
    t0, t1 = a.range if a.range else (0.0, None)
    generate(a.src, a.out, a.model, t0, t1)
