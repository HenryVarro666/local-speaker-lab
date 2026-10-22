# PersonaPlex 预约 Demo

[English](README.md) · [验证与保留失败](native-injection/RESULTS.md)

英语预约助手，使用未微调的 PersonaPlex 权重、独立 CPU Vosk 识别器和版本化模拟工具。支持创建、插话更正、明确确认，以及改期同一编号。

**提案和回执使用 PersonaPlex 原生声音。** 同一 WebSocket 内恢复干净的音色/角色前缀，只预填充最新工具事实。默认 `native-controlled` 模式控制完整工具播报文字，全部音频由 PersonaPlex 生成；**B K number one** 对应卡片上的 **BK-0001**，工具上下文和起句共用同一别名。自由对话历史仍会重置；不等于原生函数调用。设备音色保留为明确标注的故障兜底。

最终两轮完整合成语音流程共八段关键波形核验通过；周二11点/周三9:30另有四段通过，并验证预填充期间取消。十次更新→网关首个有声 PCM 为 **1.474–1.734秒，中位1.552秒**，旧版中位3.338秒，减少 **53.52%**。主入口另以诊断文字+节拍零PCM验证原生音频。[精确结果与失败记录](native-injection/LATENCY_RESULTS.md)。计时不含识别/查询等待、浏览器/硬件或严格冷启动；真人麦克风、回声与主观听感仍待验。

后续可选预填充候选的十次回注→首声为 **0.828–0.976秒、中位0.898秒**，但额外日期波形核验失败（共 **11/12关键检查**），**未采用**，下述默认与主服务仍保持原版。[候选、失败记录与复现](native-injection/PREFILL_RESULTS.md)。

日期/星期节拍后续候选十次首声为 **0.778–0.942秒、中位0.868秒**，20段核心事实波形通过；但**实际网关入口61失败**，用户听到“28 11点，别的没听清”，预期日期20。因此候选仍不采用，主服务保持1.552秒版本；相同音频token的解码重放未建立修复收益。[后续实验与人工证据](date-clarity-2026-10-21/RESULTS.md)。

## 本地诊断模式

在仓库根目录运行：

```bash
uv venv --python 3.12 .venvs/scheduling
uv pip install --python .venvs/scheduling/bin/python -r requirements/scheduling.txt
uv pip install --python .venvs/scheduling/bin/python --no-deps -e .
.venvs/scheduling/bin/python -m speaker_lab.scheduling --diagnostic --today 2026-10-20 --port 8768
```

打开 `http://127.0.0.1:8768`。点击 Connect，展开 **Diagnostic text controls**，提交 `Book Thursday at three pm`，立即改为 `Actually Friday at four thirty`。确认当前提案，再提交 `Move it to Thursday at two pm` 并确认。编号保持 BK-0001。诊断输入不能当作真实模型语音或真人麦克风验收。

## 真实 GPU 配置

在独立 Linux GPU 环境安装官方 [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex)，源码 revision `3428dfd95309a7f3c84fd93259ded0f810d1ff91`。用自己的账号接受[模型访问条件](https://huggingface.co/nvidia/personaplex-7b-v1)，再交互登录独立 HF_HOME。token 不进入源码、提示词或命令参数。本次权重 revision：`fdaf4090a61cb315c138a1faee287ffd6c716309`。

下载 Apache 2.0 的 [vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models)。Vosk 只提供业务转写；原始24 kHz用户音频仍直接进入 PersonaPlex。requirements 中可选 Vosk wheel 面向 Linux，Mac 文字诊断不需要它。

从核验过的上游源码生成本地扩展：

```bash
python scripts/prepare_personaplex_native.py \
  upstream/personaplex/moshi/moshi/server.py \
  --output upstream/personaplex/moshi/moshi/server_native_cached.py
```

脚本校验源码/补丁哈希，保留 `server.py`，新建 `server_native_cached.py`，不覆盖已有不同内容的候选；扩展同时使监听 host 参数生效。固定权重已准备在独立缓存后运行：

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/hf-home" HF_HUB_CACHE="$PWD/models/hub" \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m moshi.server_native_cached --host 127.0.0.1 --port 8998 --static none

PYTHONPATH="$PWD/src" .venv/bin/python -m speaker_lab.scheduling \
  --model-url http://127.0.0.1:8998 --tool-voice native-controlled \
  --vosk-model models/vosk-model-small-en-us-0.15 \
  --today 2026-10-20 --port 8768
```

用 `ssh -N -L 8769:127.0.0.1:8768 USER@GPU_HOST` 转发，打开 `http://127.0.0.1:8769`。点击 **Connect**，准备好后点击 **Start microphone**，使用耳机。远程模型和网关均只监听 loopback。

音色路径：

- `native`：实验性引导起句/编号别名，日期时间由模型续说；额外波形核验有失败，未选作主展示。
- `native-controlled`（默认）：完整工具文字约束模型文字通道，全部声音仍由模型原生生成。
- `device`：明确使用设备音色；原生后端不支持或不可用时，也保留有来源标注的设备兜底。

页面业务日期固定，America/Chicago，工作日09:00–17:00、半小时槽位。未说明 am/pm 的1–5点按下午处理。一次说出星期和时间；**yes confirm** 授权写入，单独 **yes**、**okay** 或附和不授权；**never mind** 撤回待确认改动。预约在会话内存中，不连接实际日历或客户数据。

## 验证

```bash
.venvs/scheduling/bin/python -m pytest tests/test_scheduling_*.py -q

# 原生PCM及模型文字检查；仍需单独验证实际波形。
.venvs/scheduling/bin/python scripts/scheduling_smoke.py \
  --url http://127.0.0.1:8769 --audio-dir runs/scheduling/synthetic \
  --require-model --receipt-mode native --output runs/scheduling/native-01

# 更强的检查：独立QA环境，已缓存固定Qwen ASR模型；不是Demo运行依赖。
HF_HUB_OFFLINE=1 .venvs/asr-mac/bin/python scripts/scheduling_smoke.py \
  --url http://127.0.0.1:8769 --audio-dir runs/scheduling/synthetic \
  --require-model --receipt-mode native --native-check waveform \
  --output runs/scheduling/native-waveform-01
```

合成目录需要24 kHz单声道 `book/correct/confirm/move.wav`，实际按节奏送入，不覆盖已有输出目录。波形模式使用缓存的 Qwen3-ASR-0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`、CPU FP32、English，保存音频哈希和独立转写。文字通道漏字时音频可能正确；文字 token 正确时音频也可能错误，两者分别核对。

网关默认不录音；`--event-dir runs/events` 显式保存本地转写/工具事件，浏览器可导出同一事件轨迹。原生首个有声块的浏览器排定/结束回调只证明播放生命周期，不证明硬件可听或整句听感。运行目录、录音、token、权重不进入源码发布。完整行为、失败、计时边界和待验项见 [RESULTS.md](native-injection/RESULTS.md)。
