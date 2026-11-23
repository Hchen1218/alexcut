# alexcut：让 Agent 帮你包装口播视频

把一条粗剪好的口播交给 Agent，它会帮你把视频包装成节奏紧凑、贴脸字幕、液态玻璃界面、配乐对点的竖屏短视频（1080×1920）。

这是我自己剪视频时用的那套流程，送给你。

## 它能做什么

- **节奏编排**：把开头、讲框架、主体、结尾按片长比例分好，决定每句话用口播全屏、玻璃界面、讲课卡片还是章节转场。
- **贴脸逐词字幕和运镜**：说到哪个词，哪个词就贴着脸出现，镜头跟着关键词急推。
- **液态玻璃**：输入框、文件胶囊、素材卡片、章节转场的玻璃大字。
- **音效和配乐**：音效最响的那一下对准画面事件；配乐先出方案和搜索词，你把歌放进来后自动对点、人声段自动压低、统一响度。
- **拆对标**：给一条「我想要这个感觉」的视频，它会测出切点、镜头时长、字几帧出现、多大、什么颜色，写成拆解报告给你确认，再转成参数。不靠「高级」「有质感」这类形容词。

它**不做**粗剪（删口误、剪气口），也不做封面。

## 安装

需要：macOS 或 Linux、Python 3.9 以上、ffmpeg，以及一个支持 skill 的 Agent 工具。

**1. 把 skill 放进 Agent 的 skills 目录**

以 Codex 为例：

```bash
git clone https://github.com/Hchen1218/alexcut.git ~/.codex/skills/alexcut
```

用别的 Agent 工具，就按它的文档放进对应的 skills 目录，比如 `~/.agents/skills/alexcut`。

**2. 装依赖**

```bash
brew install ffmpeg
```

```bash
pip3 install -r ~/.codex/skills/alexcut/requirements.txt
```

**3. 检查环境**

```bash
python3 ~/.codex/skills/alexcut/scripts/check_env.py
```

看到「必需项都齐了，可以开始」就行。

**4. 跑一遍演示（不需要任何素材）**

```bash
python3 ~/.codex/skills/alexcut/scripts/demo.py --out ~/Desktop/alexcut-demo
```

几秒钟后打开 `~/Desktop/alexcut-demo/stills/sheet.jpg`，能看到字幕、贴脸字、玻璃卡片和章节大字，就说明装好了。演示里的「人」只是几个色块。

### 可选：字躲到人身后

```bash
pip3 install mediapipe
```

```bash
curl -L -o ~/Downloads/selfie_multiclass_256x256.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

不装也能出片，只是没有字被头发和身体挡住的效果，字的位置也只能按画面中间来估。

### 可选：自动转写逐字时间轴

贴脸字要精确到每个词说出口的时刻。剪映导出的句子级 SRT 不够用，所以附了一个转写脚本：

```bash
brew install whisper-cpp
```

```bash
curl -L -o ~/Downloads/ggml-large-v3-turbo-q5_0.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
```

模型约 550 MB。已经有逐字时间轴的可以跳过。

## 怎么用

在 Agent 里直接说，比如：

```text
这是我的口播 ~/Movies/口播.mov，录屏在 ~/Movies/录屏.mp4，帮我用 alexcut 包装一下。
我想要这条视频里那种字幕的感觉（~/Movies/对标.mp4，第 3 秒到第 6 秒），先帮我拆一下。
第 0:21 的卡片里录屏字太小了，改一下。
```

Agent 会按这个顺序走，**每一步都会停下来等你点头**：

1. 问清素材、对标、你不喜欢的效果、不能出现在画面上的词
2. 转写，请你校对专有名词
3. 拆对标（有对标才做），出拆解报告
4. 排分镜：每句话用什么版式、上屏什么、素材从哪来
5. 出静帧总览（几秒钟），带安全线，按手机尺寸检查
6. 出 30 秒样片（几分钟），确认动起来的手感
7. 出全片，配音效和配乐
8. 你按「编号 + 时间点 + 问题」提修改，只重渲改动的那几段

先看静帧、再看样片、最后才出全片，是为了不让你等二三十分钟才发现方向不对。

## 几个常见问题

**渲染很慢？** 每帧大约 0.2 秒，3 分钟的片子要 15–25 分钟。所以先看静帧，改动只重渲那几段。

**贴脸字能用中文吗？** 默认是中文口播配英文意译的贴脸字。要用中文，跟 Agent 说一声，它会换成得意黑。内置的英文字体没有中文字形。

**音效和配乐从哪来？** 你自己准备，并确认授权。skill 不附带音效和音乐，原因见 `ASSET-NOTICE.md`。

**风格不是我想要的？** 给 Agent 一条对标视频，告诉它哪几秒是你要的感觉。别只说「高级一点」，它会追问你是哪几秒、哪个元素。

## 目录

```
alexcut/
├── SKILL.md            # Agent 读的流程
├── references/         # 01 节奏 … 08 修改与渲染，Agent 按需读
├── scripts/            # 渲染、检查、拆对标、音效配乐、转写、演示
├── presets/            # 两套成熟的动效参数：贴脸字幕、液态玻璃
├── examples/           # 时间轴示例
├── evals/              # 风格跑偏的检查用例
└── assets/fonts/       # Anton、Montserrat、得意黑（OFL 授权）
```

## 授权

- 代码和文档：MIT，见 `LICENSE`。
- 字体：SIL Open Font License 1.1，见 `assets/fonts/OFL.txt`。
- 音效、配乐、模型和你自己的素材：见 `ASSET-NOTICE.md`。
