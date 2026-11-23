"""可选：逐字转写，给贴脸字幕和底部字幕用。已经有逐字时间轴的可以跳过。

依赖 whisper.cpp（命令 whisper-cli）和一个模型文件，例如 ggml-large-v3-turbo-q5_0.bin：
  macOS：brew install whisper-cpp
  模型：https://huggingface.co/ggerganov/whisper.cpp 下载，路径传给 --model 或设环境变量 ALEXCUT_WHISPER_MODEL

用法：
  python3 transcribe.py 口播.mov --out words.json [--lang zh] [--range 60 90] [--terms "AB roll,对标,编排"]

--range 只转其中一段时，输出的时间从 0 开始（= 时间轴时间），JSON 里记 "offset": 60。
时间轴里对应的 source 写同样的 "offset": 60，这样 words / sentences 可以直接贴进时间轴。
要原片时间就加 --keep-time。
--terms：片子里的专有名词，用逗号隔开，喂给 whisper 当提示，能少听错几个。转完仍要人工校一遍。
输出 words.json：
  {"words": [{"t0": 0.11, "t1": 0.35, "text": "我"}, ...],
   "sentences": [{"t0": 0.11, "t1": 1.36, "text": "我现在所有的视频"}, ...]}
sentences 按标点和停顿切好，可以直接贴进时间轴的 subtitles.items。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

PUNCT = set("，。！？、；：,.!?;:")
STOP = set("，。！？；：,.!?;:")          # 句子在这些标点处断；顿号「、」是列举，不断
LEAD = set("都就也还而但所因然再又才你我他她它这那在把被让给对和跟")   # 常出现在小句开头的字，断在它前面比较自然


def run(src, model, lang="zh", t0=0.0, t1=None, terms="", shift=0.0):
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "a.wav")
        cmd = [core.tool("ffmpeg"), "-v", "error", "-y", "-ss", str(t0)] + (["-t", str(t1 - t0)] if t1 else []) + \
              ["-i", src, "-vn", "-ac", "1", "-ar", "16000", wav]
        subprocess.run(cmd, check=True)
        prefix = os.path.join(d, "out")
        prompt = ("以下是普通话的句子，带标点。" if lang == "zh" else "") + (f"会提到：{terms}。" if terms else "")
        cmd = [core.tool("whisper-cli"), "-m", model, "-f", wav, "-l", lang, "-ml", "1", "-oj", "-of", prefix,
               "-np"] + (["--prompt", prompt] if prompt else [])
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:                        # GPU 不可用（例如沙盒里）时退回 CPU
            print("GPU 转写失败，改用 CPU（会慢一些）", flush=True)
            subprocess.run(cmd + ["-ng"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(prefix + ".json", encoding="utf-8") as f:
            data = json.load(f)
    words = []
    for s in data.get("transcription", []):
        text = s["text"].strip()
        if not text:
            continue
        a, b = s["offsets"]["from"] / 1000 + t0 - shift, s["offsets"]["to"] / 1000 + t0 - shift
        if all(ch in PUNCT for ch in text):          # 标点不单独成词，记在前一个词上
            if words:
                words[-1]["punct"] = words[-1].get("punct", "") + text
            continue
        words.append({"t0": round(a, 3), "t1": round(b, 3), "text": text})
    return words


def sentences(words, gap=.45, max_chars=12, min_chars=3):
    """按句末标点和停顿切句；超长时回头在句中停顿最长、两边又不太碎的地方断；太短的碎句并回前一句。"""
    out, cur = [], []
    L = lambda ws: sum(len(x["text"]) for x in ws)

    def emit(ws):
        text = "".join(x["text"] + ("、" if "、" in x.get("punct", "") and x is not ws[-1] else "") for x in ws)
        out.append({"t0": ws[0]["t0"], "t1": ws[-1]["t1"], "text": text,
                    "_stop": any(ch in STOP for ch in ws[-1].get("punct", ""))})

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if any(ch in STOP for ch in w.get("punct", "")) or nxt is None or nxt["t0"] - w["t1"] > gap:
            emit(cur)
            cur = []
            continue
        while L(cur) > max_chars:                     # 太长：在 cur 里找最好的断点
            best, score = None, -1e9
            for j in range(len(cur) - 1):
                a = L(cur[:j + 1])
                if a < min_chars or a > max_chars:
                    continue
                pause = cur[j + 1]["t0"] - cur[j]["t1"]
                nt = cur[j + 1]["text"]
                sc = (pause * 10 + ("、" in cur[j].get("punct", "")) * 2      # 停顿、顿号
                      + (len(nt) >= 2) + (nt[0] in LEAD) * 1.5               # 下一个词是完整的词 / 小句开头
                      + (cur[j]["text"] in ("的", "了", "着", "过")) * .5
                      - abs(a - max_chars * .6) * .05)
                if sc > score:
                    best, score = j, sc
            if best is None:
                break
            emit(cur[:best + 1])
            cur = cur[best + 1:]
    merged = []
    for s in out:
        if merged and len(s["text"]) < min_chars and not merged[-1]["_stop"] and s["t0"] - merged[-1]["t1"] < .6 \
                and len(merged[-1]["text"]) + len(s["text"]) <= max_chars + 2:     # 上一句已经句号收尾的不并
            merged[-1] = {"t0": merged[-1]["t0"], "t1": s["t1"], "text": merged[-1]["text"] + s["text"], "_stop": s["_stop"]}
        else:
            merged.append(s)
    return [{k: v for k, v in x.items() if k != "_stop"} for x in merged]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="逐字转写")
    ap.add_argument("src")
    ap.add_argument("--out", default="words.json")
    ap.add_argument("--model", default=os.environ.get("ALEXCUT_WHISPER_MODEL"))
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--range", nargs=2, type=float, metavar=("T0", "T1"))
    ap.add_argument("--terms", default="", help="专有名词，逗号隔开")
    ap.add_argument("--keep-time", action="store_true", help="--range 时保留原片时间，不从 0 开始")
    a = ap.parse_args()
    if not a.model or not os.path.exists(a.model):
        raise SystemExit("需要 whisper 模型：--model 路径，或设环境变量 ALEXCUT_WHISPER_MODEL。")
    t0, t1 = a.range if a.range else (0.0, None)
    shift = 0.0 if a.keep_time else t0
    ws = run(a.src, a.model, a.lang, t0, t1, a.terms, shift)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"offset": shift, "words": ws, "sentences": sentences(ws)}, f, ensure_ascii=False, indent=1)
    if t1 and ws and not any(ch in STOP for ch in ws[-1].get("punct", "")):
        print(f"注意：到 {t1:g} 秒时最后一句「{sentences(ws)[-1]['text']}」还没说完，可以把 --range 的结尾往后放一点")
    print(f"{len(ws)} 个词 → {a.out}" + (f"（时间从 0 开始，对应原片 {shift:g} 秒；时间轴 source 写 \"offset\": {shift:g}）" if shift else ""))
