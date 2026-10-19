# Local Speaker Lab

一个面向 **Ubuntu + RTX 5080 16GB** 的本地双人语音实验项目，提供 Gradio 界面、命令行和可替换的模型后端。实现：双人录音 → 每人一条等长音轨 → 带人物和时间戳的文字 → Ollama 中文纪要。

单人普通话与英文混合听写的独立实验见 [个人术语词表实验](../experiments/glossary/README.md)：复用 Qwen3-ASR，对比无词表、个人词表与无关词表，包含 ASCEND 数据准备和配对计分入口。

本文介绍双人流程的 Ubuntu / RTX 5080 部署；该流程尚未在目标 GPU 上完成真实推理验收。单人词表实验已经另行在 Windows CPU 上执行，见实验目录。代码测试、依赖解析与目标机器上的实际 GPU 验证应分别看待；部署后请执行下方 `doctor.py` 和真实录音验收。

## 一键部署、升级与验收（0.1.4）

将发布的 `deploy_ubuntu.sh`、`local-speaker-lab.zip` 和 `local-speaker-lab.zip.sha256` 放在同一目录。先在 [community-1 模型页](https://huggingface.co/pyannote/speaker-diarization-community-1) 接受访问条件，准备一段两人各自独说并有数秒重叠的录音，然后在 **Ubuntu 24.04 x86_64 的普通用户终端**执行：

```bash
bash deploy_ubuntu.sh --zip ./local-speaker-lab.zip --audio /absolute/path/two-speakers.wav
```

脚本校验并安全解压 ZIP、补齐系统包、检查 RTX 5080 open 驱动、安装隔离环境，隐藏输入 HF_TOKEN，配置本地 Ollama 并做实际 GPU 生成测试，然后运行 CUDA 诊断、分离冒烟测试和真实录音验收。首次失败立即停止，日志和 JSON/Markdown 报告保存在 `~/speaker-lab/deployment_logs/<本次ID>/`。同一命令可重复运行，复用匹配的项目、环境和模型缓存，每次重新验收。

已有 0.1.3 时，使用升级入口：

```bash
bash deploy_ubuntu.sh --upgrade \
  --zip ./local-speaker-lab.zip \
  --audio /absolute/path/two-speakers.wav
```

脚本交互选择旧目录和新目录，默认分别为 `~/speaker-lab/local-speaker-lab`、`~/speaker-lab/local-speaker-lab-0.1.4`。旧版本保留，新目录重建隔离 Python 环境；旧项目的 Hugging Face 缓存通过链接复用，ClearVoice 的用户缓存和同一 Ollama 服务中的模型也会复用。新目录已有独立 HF 缓存时保留它，并在报告中说明；旧缓存不存在时按需下载。复用链接仍依赖旧 HF 缓存目录，请保留它。恢复或再次验收时，可使用相同目录重跑，或用 `--old-directory` 和 `--destination` 显式指定路径。

驱动不合格时可显式加 `--install-driver`，只安装 Ubuntu 推荐的 open 驱动；安装后退出码为 20，需要手动重启后重跑。结束时直接显示本次报告，并打印验收报告和失败日志的绝对路径；验收通过后还会输出 Gradio 启动命令。自动 `PASS` 代表执行及产物检查通过，仍需人工回听与核实纪要。[完整用法、报告说明和故障恢复](../DEPLOYMENT.md)

0.1.4 修复了虚拟环境首次创建中断后无法重跑、早期失败没有报告、worker 日志/异常凭证脱敏不完整等问题；驱动安装增加包状态验证与原子重启标记，并提供交互升级与缓存复用入口。[本次复查及验证范围](../DEPLOYMENT_REVIEW.md)。

## 默认流程

1. FFmpeg 将录音转为 16kHz 单声道，默认接受最长 30 分钟的录音。
2. pyannote `speaker-diarization-community-1` 检测两人的发言与重叠区间，按首次发言映射为 `SPEAKER_00`、`SPEAKER_01`。
3. 单人区间直接归轨；重叠区间使用 ClearVoice `MossFormer2_SS_16K` 分离。默认核心块长 12 秒、两侧上下文各 0.5 秒。
4. 从每个人的单独发言中提取参考语音，用 `pyannote/wespeaker-voxceleb-resnet34-LM` 对每个分离块进行双向声纹匹配，处理输出声道交换。匹配分数接近时标记待复核。
5. 拼回两条与原录音时间对齐的浮点 WAV，逐人用 Qwen3-ASR-1.7B 转写，并用 Qwen3-ForcedAligner-0.6B 提供字词时间戳。
6. 所有语音模型进程退出后调用本机 Ollama `qwen3:8b` 生成引用片段 ID 的纪要；纪要失败时保留音轨和转写。

每个 GPU 阶段在独立 Python 进程中运行，完成后退出以释放显存。界面每次执行一个任务。请不要另开 CLI 或其他程序同时占用同一张 GPU；进程间没有全局 GPU 调度器。

## 1. 准备 Ubuntu

推荐 Ubuntu 24.04 x86_64、Python 3.12，建议 32GB 或更多系统内存，并为几个虚拟环境和模型缓存预留至少数十 GB 磁盘。多个环境会共享 pip 下载缓存，但安装文件仍分别占用空间。

先通过 Ubuntu 驱动管理器或 NVIDIA 官方方法安装支持 RTX 5080 / Blackwell 的 **NVIDIA open 内核模块驱动**（Ubuntu 包名通常以 `-open` 结尾；不是 Nouveau）。Blackwell 必须使用 open 内核模块；建议使用 Ubuntu 当前提供的受维护版本，CUDA 12.8 Update 1 对应 Linux 驱动至少为 570.124.06。重启后确认：[内核模块要求](https://download.nvidia.com/XFree86/Linux-x86_64/570.86.16/README/kernel_open.html)、[CUDA 12.8 驱动版本](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)

```bash
nvidia-smi
```

安装系统依赖（以下包名针对 Ubuntu 24.04）：

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv ffmpeg libsndfile1 \
  libgl1 libglib2.0-0t64 sox libsox-fmt-all git curl zstd
```

本项目使用 **PyTorch 2.8.0 + CUDA 12.8、TorchAudio 2.8.0** 作为兼容基线。CUDA 12.8 路线支持 Blackwell，同时保留较老语音工具所需的 TorchAudio 接口；`torchcodec==0.7.0` 与 Torch 2.8 配对。TorchCodec 使用 PyPI 的 **CPU 音频解码 wheel**，模型仍在 GPU 推理；无需 CUDA 视频解码版，也无需额外安装 NVDEC/NPP。Ubuntu 24.04 的 apt FFmpeg 6 满足该 TorchCodec 的 FFmpeg 4–7 共享库要求。PyTorch wheel 自带 CUDA 运行库，默认不需要安装完整 CUDA Toolkit。驱动是否可用仍以实际 CUDA 运算检查为准。[PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、[官方 wheel 版本](https://pytorch.org/get-started/previous-versions/)、[TorchCodec 版本对应表](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)

## 2. 安装项目

将整个 `local-speaker-lab` 目录复制到 Ubuntu，进入该目录运行：

```bash
cd local-speaker-lab
bash scripts/install_ubuntu.sh
```

安装脚本不会安装或卸载驱动、运行 sudo、下载模型权重、启动 Ollama。可重复执行：复用已有环境，安装锁定依赖，并执行 `pip check`；不会删除已有文件。

| 环境 | 用途 | 关键版本 |
| --- | --- | --- |
| `.venvs/app` | Gradio、音频拼接、调度、LLM HTTP 客户端 | Gradio 6.30.0，无 PyTorch |
| `.venvs/diarize` | 人物分段、声纹匹配 | pyannote.audio 4.0.7，NumPy 2.3.3，TorchCodec 0.7.0 |
| `.venvs/separate` | 双人分离 | ClearVoice 0.1.2，NumPy 1.26.4，TorchVision 0.23.0 |
| `.venvs/asr` | 转写、强制对齐 | qwen-asr 0.0.6，Transformers 4.57.6 |

环境隔离是必要的：ClearVoice 需要 NumPy <2，而 pyannote.core 6 需要 NumPy ≥2；Qwen 固定的 Transformers 需要 Hugging Face Hub <1，与界面环境的新版 Gradio 依赖不同。ASR 环境中的 Gradio 6.0.2 是上游 Qwen 依赖，实际界面在 app 环境中运行。

顶层版本在 `requirements/*.txt`，Ubuntu x86_64 / Python 3.12 的完整依赖解析快照在 `requirements/locks/`；安装脚本同时读取顶层要求和快照。`diarize.txt` 对 TorchCodec 使用 `===0.7.0`，确保已有环境中的 `0.7.0+cu128` 也会被替换为 CPU 解码版。每个环境还记录 `.venvs/<环境>/installed-versions.txt`。这些快照已完成依赖解析，不等同于 GPU 推理验收。更新依赖的方法见 [requirements/README.md](../requirements/README.md)。

其他安装选项：

```bash
# 只安装界面/调度环境，适合先检查 UI；不能执行模型推理。
bash scripts/install_ubuntu.sh --app-only

# 同时安装可选的 SepFormer 分离环境。
bash scripts/install_ubuntu.sh --with-speechbrain

# 指定已有的 Python 3.12。
bash scripts/install_ubuntu.sh --python /path/to/python3.12
```

## 3. 获取模型权限与本地 LLM

先登录 Hugging Face，在 [community-1 模型页](https://huggingface.co/pyannote/speaker-diarization-community-1) 接受访问条件，再创建有该模型读取权限的 token。模型在本地推理，首次下载需要联网。[官方使用说明](https://github.com/pyannote/pyannote-audio)

```bash
# 隐藏终端输入；不要把 token 写进 TOML、源码或版本管理。
read -rsp 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
printf '\n'

# run.sh 默认也会设置这个位置；CLI 建议显式设置。
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export PYANNOTE_METRICS_ENABLED=0
export GRADIO_ANALYTICS_ENABLED=False
```

通过 [Ollama 官方 Linux 安装说明](https://docs.ollama.com/linux) 安装 Ollama。例如：

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
curl -fsS http://127.0.0.1:11434/api/tags
```

若 Ollama 服务没有运行，另开终端执行 `ollama serve`。默认读取 `http://127.0.0.1:11434`；可在 `[summary]` 更换模型与地址。启动语音任务前避免 Ollama 中已有模型继续占显存，必要时先运行 `ollama stop qwen3:8b`。项目的纪要请求使用 `keep_alive=0`，完成后要求释放该模型。[Ollama 内存驻留说明](https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)

## 4. 检查与启动

```bash
.venvs/app/bin/python scripts/doctor.py
bash scripts/run.sh
```

浏览器打开 <http://127.0.0.1:7860>，上传双人录音，选择是否生成纪要，点击“开始处理”。界面显示两条人物音轨、带人物的文字、纪要，并提供字幕、转写和完整结果 ZIP 下载。默认 `share=False`，只监听本机。

`doctor.py` 不下载权重：检查系统命令、各环境导入、实际 CUDA float32 卷积和 bfloat16 矩阵乘法，以及 Ollama 服务和模型是否存在。通过检查仍需跑一段真实录音，确认模型访问、权重加载和完整推理。暂时不装 LLM 时使用：

```bash
.venvs/app/bin/python scripts/doctor.py --skip-ollama
```

诊断还会实际检查 NumPy/PyTorch 数据转换、浮点 WAV 读写、TorchCodec/FFmpeg 音频解码，以及 CUDA BF16 SDPA 和 STFT/Mel 运算，捕获单纯导入不能发现的 ABI 或算子问题。若要进一步检查 ClearVoice 的实际模型代码与 NumPy 接口，可在分离环境运行：

```bash
.venvs/separate/bin/python scripts/smoke_models.py --device cuda:0
```

这个冒烟测试使用小型随机权重网络，不下载权重，也不评价分离质量。真实录音验收仍使用默认完整模型。

0.1.2 增加了完整录音验收入口。准备一段两人分别独说、并有至少几秒同时说话的 30–60 秒录音，配置 HF_TOKEN 和本地 Ollama 后执行：

```bash
.venvs/app/bin/python -u scripts/acceptance.py ./recording.wav
```

脚本会运行真实模型，记录 `acceptance_logs/<时间-ID>/gpu.csv`，并检查四个 worker 确实执行、均请求 CUDA、分离任务非空、音轨与原时轴对齐、两人的字词时间戳及纪要引用。成功后保存 `result.json` 和 `acceptance-report.json`；`PASS` 只代表执行和文件结构检查通过，还必须回听重叠区间、校对文字和纪要。没有重叠导致跳过分离，或纪要失败时，验收会明确失败，不能仅凭普通流程的完成状态判断通过。可用 `--reference-1`、`--reference-2` 提供清晰声纹参考，或用 `--run-dir runs/<某次任务>` 复核已有结果。

从另一台电脑访问服务器时，保持默认绑定，通过 SSH 转发：

```bash
ssh -L 7860:127.0.0.1:7860 user@ubuntu-host
```

命令行批处理：

```bash
.venvs/app/bin/python -m speaker_lab.cli run ./recording.wav \
  --config configs/default.toml

# 只做音轨和转写。
.venvs/app/bin/python -m speaker_lab.cli run ./recording.wav --no-summary

# 可选参考语音：每段只包含对应人物，建议清晰的 3–10 秒。
.venvs/app/bin/python -m speaker_lab.cli run ./recording.wav \
  --reference-1 ./first_speaker.wav --reference-2 ./second_speaker.wav
```

人物 1 / `SPEAKER_00` 对应原录音首次发言者。如果开头两人同时说话，需要回听确认编号。若存在重叠、又找不到某人的至少 1 秒清晰单独发言，流程会报错并要求参考语音；不会凭空猜测该人物身份。参考语音用于把分离声道匹配回人物，不会给人物自动起真实姓名。

## 5. 输出与缓存

每次任务生成新的 `runs/<UTC时间>-<随机ID>/`：

```text
SPEAKER_00.wav          # 16kHz 单声道浮点 WAV；静默位置保留
SPEAKER_01.wav          # 与第一条及原录音时间对齐
segments.json          # 片段 ID、人物、原始文字、字词时间与模型名称
transcript.md          # 人工检查用转写
transcript.srt         # 带人物标签的字幕；重叠发言允许字幕时间交叠
summary.md             # 可选；带片段引用的 LLM 纪要
diarization.json       # 原始分段、人物映射与活动区间
speaker_matching.json  # 发生分离时的匹配分数与待复核标记
manifest.json          # 配置路径、使用的模型、提示与完成状态
results.zip            # 顶层结果文件，不包含 work/ 中间文件
work/                  # 归一化录音、参考片段、分离块、请求/响应、worker.log
error.json             # 任务失败时的错误及运行目录
```

`segments.json` 保留 `text_raw`；LLM 纪要不会覆盖原始转写。纪要引用会检查片段 ID 是否存在，但语义是否正确仍需要回听。输出中可能包含私人声音和文字，请自行决定保留时间。

Hugging Face 模型默认缓存于 `.cache/huggingface/`；ClearVoice 缓存在 `~/.cache/speaker-lab/clearvoice/checkpoints/`。首次完整推理会下载几个模型。以后复用本地权重，避免反复下载。ClearVoice 适配器会检查 checkpoint 指针和权重文件；下载不完整时直接失败。

需要断网运行时，应先联网跑完整流程预热所有权重，再设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 并测试相同任务。严格离线部署也可按模型官方说明下载本地目录并配置路径。当前版本保留失败任务的中间文件，但没有自动断点续跑或跨任务推理结果缓存。

## 6. 切换与扩展模型

所有参数集中在 `configs/default.toml`。相对路径以该文件中的 `[project].root` 为基准。运行前复制一份配置，保留对照实验：

```bash
cp configs/default.toml configs/sepformer.toml
```

在副本中更改：

```toml
[workers]
# 还需保留其他三个 stage 的 python 配置。
separate_python = ".venvs/separate-sb/bin/python"

[separate]
backend = "sepformer"
model = "speechbrain/sepformer-wsj02mix"
device = "cuda:0"
```

安装可选环境后运行：

```bash
bash scripts/install_ubuntu.sh --with-speechbrain
.venvs/app/bin/python scripts/doctor.py --config configs/sepformer.toml
.venvs/app/bin/python -m speaker_lab.cli run ./recording.wav --config configs/sepformer.toml
```

该 SepFormer checkpoint 为 8kHz 模型，适配器会做 16k→8k→16k 重采样；输出标记为 16kHz 不会恢复高于约 4kHz 的声音信息。默认 ClearVoice 是 16kHz 模型。两条路线都应使用真实中文录音比较，不直接套用英文合成测试集结论。[SpeechBrain 模型卡](https://huggingface.co/speechbrain/sepformer-wsj02mix)

ASR 可直接将 `[transcribe].model` 改为 `Qwen/Qwen3-ASR-0.6B` 做速度对照，其他接口保持一致。`aligner = ""` 可关闭强制对齐：此时只返回输入语音片段的起止时间，不伪造字词时间戳。语言自动识别可设 `language = ""`。[Qwen3-ASR 官方接口](https://github.com/QwenLM/Qwen3-ASR)

更换架构时实现 `run(payload: dict, options: dict) -> dict`，并把 backend 设为可信 Python 模块的 `module:Class`。四类接口在 `src/speaker_lab/backends/__init__.py` 注册，内置适配器可作为实现示例；所有音频路径为绝对路径，跨环境仅交换 JSON 和文件。

| Stage | 输入 payload | 返回结果 |
| --- | --- | --- |
| `diarize` | `audio_path` | `turns: [{start, end, speaker}]`；时间单位秒，保留重叠 |
| `separate` | `jobs: [{id, input_path, output_dir, ...}]` | `jobs: [{id, sources: [path0, path1]}]`；两个 16kHz 单声道文件 |
| `match` | `references: {speaker: path}` 与分离 `jobs` | `jobs: [{id, speakers, margin, uncertain, ...}]`；speakers 顺序对应 sources |
| `transcribe` | `jobs: [{id, audio_path, speaker, start, end}]` | `segments: [{job_id, speaker, start, end, text, words}]`；返回原录音时间坐标 |

各接口还可返回 `model` 记录使用的模型。为新后端建立独立虚拟环境，在 `[workers]` 指定对应 Python 即可。不要把多个模型项目的依赖强行合并。TF-GridNet、目标人物提取和多麦波束形成可用这种方式扩展；本版只内置 ClearVoice 与 SepFormer 两种分离后端。

## 7. 真实录音验收与已知限制

先准备 30–60 秒的双人中文录音，包含每人几秒单独说话和一个短重叠片段。执行带纪要的完整任务，回听两条音轨并检查：人物是否交换、重叠处是否漏声、文字是否归对人物、时间戳是否落在正确位置、纪要引用是否支持对应结论。再逐步增加长录音、噪声、方言、相似声线和混响。

| 检查目标 | 方法 |
| --- | --- |
| GPU 与安装 | `doctor.py`，终端运行时观察 `nvidia-smi` |
| 分离效果 | 回听串音、漏声、音色和电平；有干净参考轨时计算 SI-SDRi |
| 转写与人物归属 | 人工标注一小批音频，统计 CER 与人物归属错误 |
| 模型对照 | 固定录音、ASR 和分段配置，每次只换一个后端，记录耗时和峰值显存 |
| 应用逻辑 | 在 app 环境安装 pytest，运行 `python -m pytest -q`；这些测试不代替模型推理 |

这是离线开发 MVP，支持恰好两位说话人，不提供实时流式处理、多麦空间信息利用、降噪前端或自动人物命名。它依赖人物分段检测出的重叠；漏检的重叠不会被分离。单人区间从混音直接归轨，背景声仍可能保留。

ClearVoice 上游会对分离来源做 RMS 电平归一化，因此两条轨的绝对响度、相加后的混合波形不保证与原始录音一致。浮点 WAV 保留模型电平；播放或导入剪辑软件后仍应检查峰值。分块拼接也可能存在边界音色变化。本版 16kHz 音轨适合语音研发，不承诺高保真录音棚音质。

人物匹配 `margin` 是两种声道排列的平均余弦分数之差，不是准确率或校准后的置信度。相似声线、短句和持续重叠可能仍然归错人物，`uncertain` 仅提供复核提示。

若显存不足，先关闭其他 GPU 程序，将 `[pipeline].separation_chunk_seconds` 降至 6–8 秒，ASR 改为 0.6B 或关闭 aligner。默认分离块长加两侧上下文必须不超过 20 秒，以避免 ClearVoice 内部分块导致未跟踪的人物交换。单卡顺序运行有助于控制显存，但不能保证所有录音都在固定显存内完成。

模型和代码各有上游许可与访问条件，相关链接见本文件与 `requirements/`；项目不会把模型权重打包进下载 ZIP。

## 兼容性复查：0.1.1

- TorchCodec 固定为 PyPI CPU 解码版，修复 CUDA 后缀 wheel 无法从默认索引安装的问题；重复安装也会修正已有 CUDA Codec 环境。
- ClearVoice 使用官方 NumPy 接口，绕过文件接口对输入电平的额外处理；加载时严格核对全部模型张量，拒绝部分随机权重。上游仍有逐来源 RMS 归一化，不能据此恢复每个人的原始绝对音量。
- SepFormer 的缓存目录按模型标识隔离，防止更换模型后沿用旧缓存。
- 五套 Ubuntu 24.04 / x86_64 / Python 3.12 锁定依赖图已重新解析，检查了 Linux wheel 可用性。
- Windows 上已实际安装 Torch 2.8 CPU 与 ClearVoice 0.1.2，运行真实模型代码的随机权重前向、NumPy API 与解码测试；完整架构与官方 checkpoint 元数据中的 1099 个张量形状一致。未下载完整 checkpoint，未在 Ubuntu / RTX 5080 上执行预训练模型推理。
