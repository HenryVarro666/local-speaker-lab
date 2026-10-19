# Ubuntu + RTX 5080 一键部署、升级和验收

适用版本：0.1.4；目标为原生 Ubuntu 24.04 x86_64、单张 RTX 5080、Python 3.12。使用默认四个 CUDA 后端。安装需要普通用户的 sudo 权限、网络和数十 GB 空间，首次安装及权重下载可能较久；多 GPU 机器请按本文手动部署流程明确选卡。

## 准备发布文件和录音

1. 在 Hugging Face 登录并接受 [community-1 的访问条件](https://huggingface.co/pyannote/speaker-diarization-community-1)，创建有该模型读取权限的 token。这一步需要账号本人完成一次。
2. 准备 30–60 秒双人录音：每人最好至少 5 秒单独发言，以及数秒同时说话。脚本接受 10–120 秒的音频，不自动截断；没有检测到重叠时验收失败。完全重叠且缺少独说参考的录音不适合此入口。
3. 将三个发布文件放在同一目录：`deploy_ubuntu.sh`、`local-speaker-lab.zip`、`local-speaker-lab.zip.sha256`。

## 首次部署

```bash
cd ~/Downloads
bash deploy_ubuntu.sh \
  --zip "$PWD/local-speaker-lab.zip" \
  --audio "$HOME/recordings/two-speakers.wav"
```

不要使用 `sudo bash`，也不要将输入重定向到非交互会话。需要时会提示输入 sudo 密码和隐藏的 HF_TOKEN；HF_TOKEN 留空可以复用当前终端的同名环境变量。token 只放在当前进程及确实需要 Hugging Face 权限的子进程环境里，不保存到配置或 HF 登录文件，也不传给系统包、pip 或 Ollama 安装命令。

默认解压至 `~/speaker-lab/local-speaker-lab`。可以指定其他目录：

```bash
bash deploy_ubuntu.sh --zip ./local-speaker-lab.zip \
  --audio /absolute/path/two-speakers.wav \
  --destination "$HOME/audio-projects/local-speaker-lab-0.1.4"
```

ZIP 的匹配 `.sha256` 文件必须在 ZIP 同目录。脚本校验摘要、成员路径、文件类型、CRC，再解压；拒绝链接、路径穿越和异常大源文件包。已有同路径源文件必须与 ZIP 内容一致，才会复用；用户修改过的文件会使脚本停止，请改用新的 `--destination`，或直接从你已修改的项目内运行。不自动覆盖源码或用户配置。

已经解压好项目时，可原地运行：

```bash
cd /absolute/path/local-speaker-lab
bash deploy_ubuntu.sh --audio /absolute/path/two-speakers.wav
```

## 从 0.1.3 升级并复用缓存

从新版发布文件所在目录执行：

```bash
bash deploy_ubuntu.sh --upgrade \
  --zip "$PWD/local-speaker-lab.zip" \
  --audio "$HOME/recordings/two-speakers.wav"
```

入口会交互选择两个目录，直接回车使用默认值：

- 旧版本目录：`~/speaker-lab/local-speaker-lab`。
- 新版本目录：`~/speaker-lab/local-speaker-lab-0.1.4`。

旧目录必须是可识别的 0.1.3 项目目录，新旧目录必须分开且不能互相包含。脚本保留旧版源码、配置、环境和历史结果，新版源码由本次 ZIP 校验解压。新目录可以已经存在：源码仍须与 ZIP 一致，才能继续复用；用户改过源码或安装过不同构建的 0.1.4 时停止，不覆盖更改，请选择空的新目录。旧版的自定义配置不会自动迁移，先用新版默认配置验收，再逐项迁移参数。

缓存处理：

| 内容 | 升级行为 |
| --- | --- |
| Hugging Face 模型 | 旧项目 `.cache/huggingface` 存在且新目录尚无缓存时，创建链接，复用 pyannote、声纹、Qwen ASR 和 ForcedAligner 权重；新目录已有独立缓存则保留并报告，不覆盖 |
| ClearVoice | 同一用户自动复用 `~/.cache/speaker-lab/clearvoice/` |
| Ollama | 同一系统服务自动复用已安装模型；脚本仍检查模型存在及实际 GPU 生成 |
| Python 虚拟环境 | 在新目录安装，不复制旧 `.venvs`；pip 用户下载缓存仍可复用 |

HF 缓存链接依赖旧缓存实际位置，升级后不能删除或移动它。旧项目默认缓存不存在时，报告说明无法复用，缺少的权重按需下载；不会自动猜测以前使用的自定义 HF 缓存目录。仅设置当前终端的 `HF_HOME` 不能替代此处理：一键入口使用新项目内的 HF 缓存位置。脚本仍隐藏读取 HF_TOKEN 并检查 gated 权限，已有缓存不代表账号权限已经验证。

驱动重启后恢复、失败修复后恢复或重复验收，可以再次运行同一升级命令。也可以预设目录作为选择的默认值，交互时直接回车确认，仍保留 HF_TOKEN 的隐藏输入：

```bash
bash deploy_ubuntu.sh --upgrade \
  --zip "$PWD/local-speaker-lab.zip" \
  --audio "$HOME/recordings/two-speakers.wav" \
  --old-directory "$HOME/speaker-lab/local-speaker-lab" \
  --destination "$HOME/speaker-lab/local-speaker-lab-0.1.4"
```

每次重新检查和验收，日志独立保存；不复用上次的通过状态。旧版本源码、配置、环境与历史报告不变；共用模型缓存可能补下载文件。

## 执行内容与重复运行

脚本按顺序执行，首次失败立即停止：

1. 检查系统与录音路径；只补装缺少的 Ubuntu 系统包。
2. 校验并解压/复用 ZIP；升级模式验证旧目录并建立或检查缓存链接。
3. 检查 GPU 0 为 RTX 5080、驱动不低于 570.124.06，且使用 NVIDIA open 内核模块；Nouveau 不满足条件。
4. 将录音完整转成 16 kHz 单声道 PCM，并检查时长。
5. 隐藏输入 HF_TOKEN，复用/安装四个隔离 Python 环境；执行 `pip check` 并保存版本快照。
6. 用该 token 下载 community-1 的配置文件，验证 gated 模型权限。
7. 缺少 Ollama 时通过 [官方 Linux 安装脚本](https://docs.ollama.com/linux) 安装，启用本地 systemd 服务。已有程序和已安装模型会复用；缺少配置模型时才 `ollama pull`，避免重复更新模型标签。
8. 对本地模型做一次真实生成，验证 `ollama ps` 为 `100% GPU`，且 `/api/ps` 中 `size_vram == size > 0`；随后卸载本次预热模型。默认模型为 `qwen3:8b`。
9. 执行 `doctor.py` 的实际 CUDA/解码/依赖检查，以及 ClearVoice CUDA 随机权重冒烟测试。
10. 用完整模型跑真实录音，记录 GPU CSV；强制检查四个 worker、重叠分离任务、双人音轨、带人物的文字、强制对齐时间戳、有效引用的本地纪要，并确认 Ollama 已释放驻留模型。

同一条命令可以再次运行：不删除虚拟环境、模型缓存、录音和旧报告；每次生成新日志并重新做检查及录音推理。脚本不以“目录存在”代替验收，也不从“最新 runs 目录”猜测本次结果。

首次创建环境若中断，仅留下 `pyvenv.cfg`、缺 Python 或缺 pip，重跑会补齐环境或执行 `ensurepip`。已有可运行的非 Python 3.12 环境则在修改前停止，需要自行将该环境移开；不覆盖已有不兼容解释器。无法安全隐藏输入时会在读取 HF_TOKEN 前停止。

Ollama 首装中断后，若程序存在但 systemd 服务缺失，重跑会修复安装；服务被明确 masked 时则停止。安装期间只为本次子进程使用非交互 sudo 包装，定时刷新已有终端授权；授权失效会立即失败并保存报告，不在后台等待密码。不会永久修改 sudo 配置。

默认仅使用本机 `http://127.0.0.1:11434`。已有其他 Ollama 模型驻留时会停止并提示，不会自动终止他人的模型。运行前关闭其他占用 GPU 的任务；本项目没有跨进程 GPU 调度器。可修改默认配置的本地 LLM 模型或语言，替换分离后端/设备的实验请按本文手动流程部署。

## 驱动与重启

若驱动已正常工作，直接复用。需要让脚本安装驱动时，显式添加：

```bash
bash deploy_ubuntu.sh --zip ./local-speaker-lab.zip \
  --audio /absolute/path/two-speakers.wav --install-driver
```

仅在驱动检查失败时，解析 `ubuntu-drivers devices`，选择唯一标为 recommended 的 NVIDIA open 驱动。若没有明确的 open 推荐项，停止并保留诊断，请通过 Ubuntu 驱动管理器安装受支持的 open 版本。脚本不自行猜测驱动分支。

驱动命令完成后还要确认推荐包的 `dpkg` 状态为 `installed`，才原子写入重启标记并返回 **20 / REBOOT_REQUIRED**，保存报告且不自动重启。包未完整安装则报告失败，不宣称需要重启即可恢复。手动执行 `sudo reboot`，重连后重跑同一命令。重启前重复运行会读取 boot ID 标记并继续提示重启，不重复安装。重启后再次检查驱动和实际 CUDA 运算，才清除标记并继续模型验收。

## 报告与失败恢复

普通部署默认日志在项目父目录的 `deployment_logs/deploy-<UTC时间>.<ID>/`；升级模式固定放在 `~/speaker-lab/deployment_logs/` 下，避免交互更改目标后丢失早期诊断。终端会打印绝对路径。入口结束时直接显示本次报告，并列出本次验收报告和失败日志路径；失败发生得太早、尚未产生的验收报告会明确标注。以本次终端打印的路径为准，不要从历史目录中猜测“最新结果”。

目标路径无效或其日志父目录不可写时，入口优先改用 `~/speaker-lab/deployment_logs/`，再尝试临时目录保存失败报告。Python 编排初始化失败会另建 `initialization-failure-*` 报告，保留已有同名报告；最终以终端打印的位置为准。

| 文件 | 内容 |
| --- | --- |
| `bootstrap.log`、`bootstrap-report.json/md` | 参数、系统包、ZIP 校验/解压及整个入口退出状态；早期失败也保留 |
| `deployment.log`、`01-*.log` 等 | Python 编排及每个步骤的输出，已知 token 脱敏 |
| `deployment-report.json/md` | 总体状态、阶段用时、错误原因、GPU/驱动、模型 digest、依赖快照路径及验收结果 |
| `source-sha256.txt` | 本次 ZIP 的已校验摘要（使用 `--zip` 时） |
| `upgrade-report.json` | 升级目录、缓存实际路径、复用策略、旧缓存保留要求及警告（升级模式） |
| `installed-versions/*.txt` | 本次四个环境的完整 pip freeze 快照 |
| `input.wav` | 本次验收使用的规范化录音 |
| `recording-acceptance/gpu.csv` | 推理期间 NVIDIA GPU 内存/利用率采样 |
| `recording-acceptance/result.json` | 本次确切的 `run_dir` 和产物路径 |
| `recording-acceptance/acceptance-report.json` | 完整录音自动验收证据 |

将终端打印的本次日志绝对路径填入以下变量，即可重新查看：

```bash
LAB_LOG="/absolute/path/deployment_logs/deploy-本次ID"

cat "$LAB_LOG/bootstrap-report.md"
if [[ -f "$LAB_LOG/deployment-report.md" ]]; then
  less "$LAB_LOG/deployment-report.md"
fi
if [[ -f "$LAB_LOG/recording-acceptance/acceptance-report.json" ]]; then
  jq . "$LAB_LOG/recording-acceptance/acceptance-report.json"
fi
```

查找失败阶段及它的日志文件：

```bash
tail -n 100 "$LAB_LOG/bootstrap.log"
if [[ -f "$LAB_LOG/deployment-report.json" ]]; then
  jq -r '.error // empty' "$LAB_LOG/deployment-report.json"
  jq -r '.steps[] | select(.status != "PASS") |
    [.name, .status, .log, (.error // "")] | @tsv' \
    "$LAB_LOG/deployment-report.json"
fi
```

录音验收通过后，`result.json` 保存确切的推理运行目录：

```bash
LAB_RUN="$(jq -r '.run_dir' "$LAB_LOG/recording-acceptance/result.json")"
ls -lh "$LAB_RUN"
less "$LAB_RUN/transcript.md"
less "$LAB_RUN/summary.md"
```

录音推理失败、尚未写出 `result.json` 时，在终端给出的失败阶段日志中查找 `Run artifacts preserved` 或 `运行目录及日志`，再查看该运行目录的 `error.json` 和 `work/<阶段>/worker.log`。所有故障恢复均使用原命令和同一个目标目录。

若失败发生在 Python 编排启动之前，只会有 bootstrap 报告。否则 Python 在每步开始/结束更新报告，并在失败、Ctrl+C 或 SIGTERM 时写最终状态；系统断电或 SIGKILL 无法执行清理，已写出的阶段日志和 `RUNNING` 报告仍会保留。退出码 0 表示自动验收通过，20 表示需重启，其他非零值表示失败/中断。

查看失败步骤日志，修复原因，重新运行原命令即可。例如，HF 401/403 需核对账号模型权限及 token；Ollama 模型驻留时先 `ollama ps`，自行停止相应模型；GPU 内存不足时关闭其他 GPU 任务。不会在失败后继续运行后续验收或启动界面。

脱敏覆盖 bootstrap、部署步骤及 `runs/<本次运行>/work/<阶段>/worker.log`、worker 响应 JSON、pipeline 错误 JSON；结构化数据先脱敏再编码，避免引号/反斜杠破坏 JSON。报告会记录清理警告；清理时再次按 Ctrl+C 不会跳过最终报告。

`PASS` 仅表示实际执行和产物检查通过，报告中的 `quality_status` 始终为 `PENDING_MANUAL_REVIEW`：还需要回听两条音轨、确认人物对应与串音，校对转写，并逐项核实纪要事实与引用。不声称自动证明分离质量或纪要准确率。

## 验收后启动 Gradio

报告会输出你机器上的绝对路径命令。默认目录可执行：

```bash
cd "$HOME/speaker-lab/local-speaker-lab"
read -rsp 'HF_TOKEN（如模型已完整缓存可留空）: ' HF_TOKEN
export HF_TOKEN
printf '\n'
bash scripts/run.sh
```

在 Ubuntu 本机打开 `http://127.0.0.1:7860`。远程服务器可通过 SSH 转发后访问：

```bash
ssh -L 7860:127.0.0.1:7860 user@ubuntu-host
```

入口不后台启动 UI，验收和界面进程的日志分别保留。系统依赖均已准备好时可加 `--skip-system-packages`；这只跳过 apt，仍检查所有运行条件。

升级模式的部署报告还记录 `evidence.upgrade`，可以确认本次使用的旧目录、新目录和实际缓存目标，以及新目录独立缓存是否被保留。

本发布包在 Windows 上做了离线单元测试、Bash 语法和安全解压测试；没有在 Ubuntu + RTX 5080 上执行真实模型推理。目标机器脚本生成的报告才是该机器实际部署与验收的证据。


## 手动部署与日常使用

以下命令从项目根目录执行；它们保留独立安装、诊断、UI 和 CLI 入口，适合开发与模型对照。自动升级、驱动恢复和本次报告的位置见前文。

### 1. 准备 Ubuntu

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

### 2. 安装项目

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

### 3. 获取模型权限与本地 LLM

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

### 4. 检查与启动

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
