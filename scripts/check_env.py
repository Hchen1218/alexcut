"""检查渲染环境：缺什么、怎么装。第一次用 alexcut 先跑这个。

  python3 check_env.py
"""
import importlib
import os
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

OK, BAD, OPT = "✅", "❌", "➖"
mac = platform.system() == "Darwin"


def has_mod(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def main():
    rows, must_fail = [], False

    def row(state, name, note=""):
        rows.append(f"{state} {name}" + (f"：{note}" if note else ""))

    v = sys.version_info
    if v >= (3, 9):
        row(OK, f"Python {v.major}.{v.minor}")
    else:
        must_fail = True
        row(BAD, f"Python {v.major}.{v.minor}", "需要 3.9 以上")

    for mod, pip in (("numpy", "numpy"), ("PIL", "pillow")):
        if has_mod(mod):
            row(OK, pip)
        else:
            must_fail = True
            row(BAD, pip, f"pip install {pip}")

    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) or any(os.path.exists(os.path.join(d, exe)) for d in ("/opt/homebrew/bin", "/usr/local/bin")):
            row(OK, exe)
        else:
            must_fail = True
            row(BAD, exe, "brew install ffmpeg" if mac else "用系统包管理器安装 ffmpeg")

    fonts = ["Anton-Regular.ttf", "Montserrat-VF.ttf", "SmileySans-Oblique.ttf"]
    miss = [f for f in fonts if not os.path.exists(os.path.join(core.FONT_DIR, f))]
    if miss:
        must_fail = True
        row(BAD, "字体", "缺 " + "、".join(miss) + f"，放进 {core.FONT_DIR}")
    else:
        row(OK, "字体（3 款，均为 OFL 授权）")

    try:
        core.mediapipe()
        mp_ok = True
    except Exception:
        mp_ok = False
    if mp_ok:
        row(OK, "mediapipe", "可以生成人像遮罩，字能躲到人身后")
    else:
        row(OPT, "mediapipe（可选）", "pip install mediapipe；不装也能出片，只是没有字躲到人身后的效果")

    if shutil.which("whisper-cli") or os.path.exists("/opt/homebrew/bin/whisper-cli"):
        m = os.environ.get("ALEXCUT_WHISPER_MODEL")
        if m and os.path.exists(m):
            row(OK, "whisper-cli + 模型")
        else:
            row(OK, "whisper-cli", "模型用 transcribe.py --model 传，或设环境变量 ALEXCUT_WHISPER_MODEL；"
                "还没有模型就下载 ggml-large-v3-turbo-q5_0.bin")
    else:
        row(OPT, "whisper-cli（可选）", ("brew install whisper-cpp" if mac else "编译 whisper.cpp")
            + "；已有逐字时间轴就不需要")

    print("\n".join(rows))
    print("\n必需项都齐了，可以开始。" if not must_fail else "\n有必需项没装，装好再继续。")
    sys.exit(1 if must_fail else 0)


if __name__ == "__main__":
    main()
