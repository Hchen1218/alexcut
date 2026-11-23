"""装好以后先跑这个：不用任何素材，生成一段假口播和假录屏，按示例时间轴出静帧，确认环境能用。

  python3 scripts/demo.py                  # 输出到当前目录下的 alexcut-demo/
  python3 scripts/demo.py --out 某个目录 [--video]   # --video 再渲染一段 4 秒的视频（约 1 分钟）

看 alexcut-demo/stills/sheet.jpg：有贴脸字、玻璃卡片、胶囊、章节大字，就说明装好了。
假口播里的「人」只是几个色块，所以贴脸字的位置是按画面中间估的，不代表真实效果。
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)


def fake_aroll(path, dur):
    """1080×1920 的假口播：暖色房间 + 一个色块小人。"""
    W, H = 1080, 1920
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        k = y / H
        d.line([(0, y), (W, y)], fill=(int(70 + 60 * k), int(58 + 40 * k), int(52 + 30 * k)))
    d.rectangle((0, 0, 260, 1300), fill=(150, 132, 112))            # 窗帘
    d.rectangle((820, 300, 1080, 1250), fill=(52, 50, 54))           # 柜子
    d.ellipse((230, 1150, 850, 2300), fill=(88, 70, 58))             # 肩膀
    d.rectangle((470, 980, 610, 1200), fill=(222, 182, 156))         # 脖子
    d.ellipse((380, 560, 700, 1010), fill=(232, 192, 166))           # 脸
    d.pieslice((360, 500, 720, 860), 180, 360, fill=(96, 62, 40))    # 头发
    d.rounded_rectangle((420, 760, 530, 830), 18, outline=(25, 25, 28), width=10)   # 眼镜
    d.rounded_rectangle((550, 760, 660, 830), 18, outline=(25, 25, 28), width=10)
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    png = path + ".png"
    im.save(png)
    subprocess.run([core.tool("ffmpeg"), "-v", "error", "-y", "-loop", "1", "-i", png,
                    "-f", "lavfi", "-i", f"sine=frequency=220:duration={dur}",
                    "-t", str(dur), "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path], check=True)
    os.remove(png)


def fake_screen(path, dur):
    """1920×1080 的假录屏：左边对话栏，右边文档。"""
    W, H = 1920, 1080
    im = Image.new("RGB", (W, H), (246, 246, 248))
    d = ImageDraw.Draw(im)
    f = core.font("SmileySans-Oblique.ttf", 30)
    fb = core.font("SmileySans-Oblique.ttf", 40)
    d.rectangle((0, 0, W, 64), fill=(232, 232, 236))
    d.rectangle((420, 64, 1260, H), fill=(255, 255, 255))
    d.text((460, 100), "示例：对标拆解", font=fb, fill=(30, 30, 34))
    lines = ["镜头数：82 个", "中位镜头时长：2.2 秒", "人物占比：约四分之一",
             "音效和切点：不卡点", "重点词：第一帧就是最终大小", "颜色：白 + 一个强调色"]
    for i, s in enumerate(lines):
        d.text((460, 190 + i * 70), "· " + s, font=f, fill=(50, 50, 56))
    for i in range(8):
        d.rounded_rectangle((30, 100 + i * 110, 380, 180 + i * 110), 14, fill=(236, 236, 240))
    png = path + ".png"
    im.save(png)
    subprocess.run([core.tool("ffmpeg"), "-v", "error", "-y", "-loop", "1", "-i", png, "-t", str(dur), "-r", "30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True)
    os.remove(png)


def main():
    ap = argparse.ArgumentParser(description="不用素材的演示：确认环境能出静帧")
    ap.add_argument("--out", default="alexcut-demo")
    ap.add_argument("--video", action="store_true", help="再渲染 4 秒视频")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "素材"), exist_ok=True)

    with open(os.path.join(SKILL, "examples", "timeline-示例.json"), encoding="utf-8") as f:
        tl = json.load(f)
    dur = max(s["t1"] for s in tl["segments"])
    print("生成假口播和假录屏…", flush=True)
    fake_aroll(os.path.join(out, "素材", "口播.mp4"), dur)
    fake_screen(os.path.join(out, "素材", "录屏.mp4"), 20)

    tl["_说明"] = "alexcut 演示：由 scripts/demo.py 生成。素材是假的，只用来确认环境能出静帧。"
    tl["sources"] = {"aroll": {"path": "素材/口播.mp4", "scale": 1.0},
                     "screen": {"path": "素材/录屏.mp4", "fit": "native", "scale": 1}}
    tl["audio"] = "素材/口播.mp4"
    tl.pop("masks", None)
    tl.pop("music", None)
    for s in tl["segments"]:
        for it in s.get("items", []):
            if it.get("kind") == "card":
                it["crop"] = [0.2, 0.05, 0.68, 0.75]
    p = os.path.join(out, "timeline.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tl, f, ensure_ascii=False, indent=1)

    py = sys.executable
    print("\n时间轴检查：", flush=True)
    subprocess.run([py, os.path.join(HERE, "timeline.py"), "check", p])
    print("\n出静帧…", flush=True)
    r = subprocess.run([py, os.path.join(HERE, "compose.py"), p, "--stills", "1.0 3.3 4.1 4.6 5.6 6.7 10.5 12.9",
                        "--guides", "--sheet"])
    if r.returncode:
        raise SystemExit("出静帧失败，把上面的报错发给 Agent。")
    if a.video:
        print("\n渲染 4 秒视频…", flush=True)
        subprocess.run([py, os.path.join(HERE, "compose.py"), p, "--range", "2.6", "6.6",
                        "--out", os.path.join(out, "output", "演示.mp4")], check=True)
    print(f"\n完成。看 {os.path.join(out, 'stills', 'sheet.jpg')}")


if __name__ == "__main__":
    main()
