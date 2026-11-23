"""音效和配乐：放音效、找推起点、按时间轴对点混音、人声段自动压低、限幅、响度统一。只依赖 numpy + ffmpeg。

0) 放音效：按时间轴 sfx 把音效的峰值对准画面事件那一帧，输出纯音效轨和「人声+音效」。
   python3 bgm.py sfx timeline.json

1) 找推起点（swell）：用户放进来的歌，哪些时刻音乐明显「推起来」，用来对准关键句。
   python3 bgm.py swells 歌.mp3
2) 混音：按时间轴里的 music 段落对点，输出配乐轨和最终混音。
   python3 bgm.py mix timeline.json [--mux 成片.mp4]

时间轴 music 写法（时间都是视频时间，song_start 是歌曲里的起点）：
"music": {
  "pieces": [
    {"file": "bgm/开场.mp3", "song_start": 0.54, "t0": 0, "t1": 36.1, "fade_in": 0.02, "fade_out": 0.04, "gain": -3.5,
     "auto": [[15.4, 3.0], [15.56, 0]], "lowpass": [51.0, 52.6]}
  ],
  "bus_db": -14, "duck_db": -5,
  "voice": "口播.mov",              # 用来判断哪里有人声（默认用时间轴的 audio）
  "mix_with": "output/人声+音效.wav", # 和配乐叠在一起的人声轨（默认同 voice）
  "lufs": -14
}
时间轴 sfx 写法：
"sfx": {
  "pools": {"pop": ["sfx/pop1.wav", "sfx/pop2.wav"], "thump": ["sfx/thump.wav"]},
  "items": [{"t": 3.97, "pool": "thump", "db": -6}, {"t": 4.5, "file": "sfx/tick.wav", "db": -14, "align": "start"}]
}
align 默认 "peak"：音效最响的那一下落在 t；"start" 是从 t 开始播。同一个 pool 轮换着用，相邻两次不重样。
规则见 references/05-音效.md、references/06-配乐.md：思考型 / 情绪型交替，情绪要挣来，推起点对准关键句，关键句前可以留一小段空白。
"""
import argparse
import json
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

SR = 48000


def load(path, ss=None, t=None, af=None, sr=SR, ch=2):
    cmd = [core.tool("ffmpeg"), "-v", "error"] + (["-ss", str(ss), "-t", str(t)] if ss is not None else []) + \
          ["-i", path, "-vn", "-ac", str(ch), "-ar", str(sr)] + (["-af", af] if af else []) + ["-f", "f32le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).copy()
    return x.reshape(-1, ch) if ch > 1 else x


def write(path, x):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


# ================================================================ 音效
def _peak(x):
    """音效起头 0.5 秒里最响的位置（采样点）。"""
    a = np.abs(x[:int(SR * .5)]).max(1) if x.ndim > 1 else np.abs(x[:int(SR * .5)])
    return int(np.argmax(a)) if len(a) else 0


def sfx_track(tl, n):
    m = tl.get("sfx") or {}
    base = tl["_base"]
    fix = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    pools = {k: [fix(f) for f in v] for k, v in m.get("pools", {}).items()}
    used, last, out = {}, {}, np.zeros((n, 2), np.float32)
    for it in sorted(m.get("items", []), key=lambda i: i["t"]):
        if it.get("pool"):
            files = pools[it["pool"]]
            k = used.get(it["pool"], 0)
            f = files[k % len(files)]
            if len(files) > 1 and f == last.get(it["pool"]):
                k += 1
                f = files[k % len(files)]
            used[it["pool"]], last[it["pool"]] = k + 1, f
        else:
            f = fix(it["file"])
        if not os.path.exists(f):
            raise SystemExit(f"找不到音效：{f}")
        if f not in _songs:
            _songs[f] = load(f)
        x = _songs[f] * 10 ** (it.get("db", -12) / 20)
        i = int(round(it["t"] * SR)) - (_peak(x) if it.get("align", "peak") == "peak" else 0)
        a, b = max(i, 0), min(i + len(x), n)
        if b > a:
            out[a:b] += x[a - i:b - i]
    return out


def sfx(tl_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import timeline as TL
    tl = TL.load(tl_path)
    if not (tl.get("sfx") or {}).get("items"):
        raise SystemExit("时间轴里没有 sfx.items。")
    dur = TL.end_time(tl)
    n = int(round(dur * SR))
    fx = sfx_track(tl, n)
    out_dir = os.path.join(tl["_base"], "output")
    write(os.path.join(out_dir, "音效.wav"), fx)
    if tl.get("audio"):
        v = load(tl["audio"], tl.get("audio_offset", 0.0), dur)[:n]
        full = fx.copy()
        full[:len(v)] += v
        write(os.path.join(out_dir, "人声+音效.wav"), limiter(full))
        print("人声+音效：", os.path.join(out_dir, "人声+音效.wav"))
    print("纯音效：", os.path.join(out_dir, "音效.wav"), f"（{len(tl['sfx']['items'])} 个）")


# ================================================================ 找推起点
def swells(path, top=8, gap=4.0):
    """推起点：之后 2 秒比之前 2 秒明显变响的时刻（鼓组进、弦乐进、副歌起）。"""
    sr, hop = 8000, 800                              # 0.1 秒一格
    x = load(path, sr=sr, ch=1)
    nb = len(x) // hop
    rms = np.sqrt((x[:nb * hop].reshape(nb, hop) ** 2).mean(1) + 1e-10)
    db = 20 * np.log10(rms)
    win = 20
    after = np.convolve(db, np.ones(win) / win, "full")[win - 1:nb + win - 1]     # i 往后 2 秒的平均
    before = np.concatenate([np.full(win, db[:win].mean()), np.convolve(db, np.ones(win) / win, "valid")])[:nb]
    rise = after - before
    picks = []
    for i in np.argsort(-rise):
        if rise[i] < 2.5 or len(picks) >= top:
            break
        if all(abs(i - j) * .1 >= gap for j in picks):
            picks.append(i)
    loud = int(np.argmax(np.convolve(db, np.ones(80) / 80, "same")))
    return sorted((round(i * .1, 2), round(float(rise[i]), 1)) for i in picks), round(loud * .1, 1), round(nb * .1, 1)


# ================================================================ 混音
def ramp(n, a, b):
    return np.linspace(a, b, max(n, 1), dtype=np.float32)[:n]


_songs = {}


def render_piece(p):
    key = (p["file"], bool(p.get("lowpass")))
    if p["file"] not in _songs:
        _songs[p["file"]] = load(p["file"])
    x = _songs[p["file"]]
    n = int(round((p["t1"] - p["t0"]) * SR))
    i0 = int(round(p.get("song_start", 0) * SR))
    seg = x[i0:i0 + n].copy()
    if len(seg) < n:
        seg = np.pad(seg, ((0, n - len(seg)), (0, 0)))
    tv = p["t0"] + np.arange(n) / SR
    if p.get("lowpass"):                             # 收尾：逐渐混到 450 Hz 低通版本，像音乐退到墙后面
        if key not in _songs:
            _songs[key] = load(p["file"], af="lowpass=f=450")
        lp = _songs[key][i0:i0 + n]
        lp = np.pad(lp, ((0, n - len(lp)), (0, 0))) if len(lp) < n else lp
        a, b = p["lowpass"]
        k = np.clip((tv - a) / max(b - a, 1e-3), 0, 1)[:, None]
        seg = seg * (1 - k) + lp * k
    g = np.full(n, p.get("gain", 0.0), np.float32)
    if p.get("auto"):
        ts, ds = zip(*p["auto"])
        g += np.interp(tv, ts, ds).astype(np.float32)
    amp = 10 ** (g / 20)
    fi, fo = int(p.get("fade_in", .05) * SR), int(p.get("fade_out", .3) * SR)
    if fi:
        amp[:fi] *= ramp(fi, 0, 1) ** 2
    if fo:
        amp[n - fo:] *= ramp(fo, 1, 0) ** 2
    return int(round(p["t0"] * SR)), seg * amp[:, None]


def voice_duck(path, n, duck_db):
    """有人声的地方把音乐再压 duck_db；保持 250 ms，避免字间缝隙里音乐来回抽。"""
    v = load(path).mean(1)[:n]
    hop = 480
    nb = len(v) // hop
    rms = np.sqrt((v[:nb * hop].reshape(nb, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = np.percentile(db, 60) - 6
    act = (np.convolve((db > thr).astype(np.float32), np.ones(25), "same") > 0).astype(np.float32)
    sm = np.empty_like(act)
    cur = 0.0
    att, rel = np.exp(-1 / 6), np.exp(-1 / 40)        # 60 ms 压下去，400 ms 回来
    for i, a in enumerate(act):
        cur = a + (cur - a) * (att if a > cur else rel)
        sm[i] = cur
    env = np.interp(np.arange(n), np.arange(nb) * hop + hop / 2, sm).astype(np.float32)
    return 10 ** (duck_db * env / 20)


def limiter(x, ceil=.97):
    blk = 240
    n = len(x)
    nb = (n + blk - 1) // blk
    pad = np.zeros((nb * blk, 2), np.float32)
    pad[:n] = x
    pk = np.abs(pad).reshape(nb, blk, 2).max((1, 2))
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    rel = np.exp(-blk / (SR * .12))
    gs = np.empty(nb, np.float32)
    cur = 1.0
    for k in range(nb):
        cur = g[k] if g[k] < cur else cur * rel + (1 - rel) * g[k]
        gs[k] = cur
    gain = np.interp(np.arange(nb * blk), np.arange(nb) * blk + blk / 2, gs)[:n].astype(np.float32)
    return np.clip(x * gain[:, None], -1, 1)


def mix(tl_path, mux=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import timeline as TL
    tl = TL.load(tl_path)
    m = tl.get("music")
    if not m or not m.get("pieces"):
        raise SystemExit("时间轴里没有 music.pieces。先按配乐方案把歌放进来、写好对点。")
    base = tl["_base"]
    fix = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    pieces = [{**p, "file": fix(p["file"])} for p in m["pieces"]]
    for p in pieces:
        if not os.path.exists(p["file"]):
            raise SystemExit(f"找不到歌曲文件：{p['file']}")
    dur = TL.end_time(tl)
    n = int(round(dur * SR))
    mus = np.zeros((n, 2), np.float32)
    for p in pieces:
        i, seg = render_piece(p)
        k = min(len(seg), n - i)
        if k > 0:
            mus[i:i + k] += seg[:k]
    out_dir = os.path.join(base, "output")
    raw = os.path.join(out_dir, "_配乐_未处理.wav")
    write(raw, mus)
    # 母线 EQ：低切 70 Hz，2.8 kHz 挖 2.5 dB 给人声让路
    mus = load(raw, af="highpass=f=70,equalizer=f=2800:t=q:w=0.8:g=-2.5")[:n]
    voice = fix(m["voice"]) if m.get("voice") else tl.get("audio")
    if voice:
        mus *= voice_duck(voice, n, m.get("duck_db", -5.0))[:len(mus), None]
    mus *= 10 ** (m.get("bus_db", -14.0) / 20)
    music_path = os.path.join(out_dir, "配乐.wav")
    write(music_path, mus)
    os.remove(raw)
    vs_auto = os.path.join(out_dir, "人声+音效.wav")
    with_path = fix(m["mix_with"]) if m.get("mix_with") else (vs_auto if os.path.exists(vs_auto) else voice)
    full = mus.copy()
    if with_path:
        vs = load(with_path, 0, dur)[:n]
        full[:len(vs)] += vs
    mixed = os.path.join(out_dir, "混音.wav")
    write(mixed, limiter(full))
    lufs = m.get("lufs", -14)
    final = os.path.join(out_dir, "混音_响度统一.wav")
    subprocess.run([core.tool("ffmpeg"), "-v", "error", "-y", "-i", mixed, "-af",
                    f"loudnorm=I={lufs}:TP=-1.0:LRA=11", "-ar", str(SR), final], check=True)
    print("配乐轨：", music_path)
    print("最终混音：", final)
    if mux:
        out = os.path.splitext(mux)[0] + "_配乐版.mp4"
        subprocess.run([core.tool("ffmpeg"), "-v", "error", "-y", "-i", mux, "-i", final, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", out], check=True)
        print("配乐版视频：", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="音效 / 配乐：放音效、找推起点、对点混音")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("sfx", help="按时间轴 sfx 放音效，输出纯音效轨和人声+音效")
    a0.add_argument("timeline")
    a1 = sub.add_parser("swells", help="列出歌曲里音乐明显推起来的时刻")
    a1.add_argument("song")
    a1.add_argument("--top", type=int, default=8)
    a2 = sub.add_parser("mix", help="按时间轴 music 段落混音")
    a2.add_argument("timeline")
    a2.add_argument("--mux", help="把混音替换进这个视频")
    a = ap.parse_args()
    if a.cmd == "swells":
        picks, loud, total = swells(a.song, a.top)
        print(json.dumps({"推起点": [{"时间": t, "抬升dB": r} for t, r in picks], "最响段落中心": loud, "时长": total},
                         ensure_ascii=False, indent=1))
    elif a.cmd == "sfx":
        sfx(a.timeline)
    else:
        mix(a.timeline, a.mux)
