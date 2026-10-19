# 模型后端与配置

模型能力由 TOML 中四个阶段的 `backend`、模型参数及 `[workers]` 解释器路径决定。适配器导入发生在 worker 中，app 只处理音频与文件协议。

## 默认后端

| 阶段 | 别名 | 默认模型 | 解释器 |
| --- | --- | --- | --- |
| 人物分段 | `pyannote` | `pyannote/speaker-diarization-community-1` | `.venvs/diarize/bin/python` |
| 双人分离 | `clearvoice` | `MossFormer2_SS_16K` | `.venvs/separate/bin/python` |
| 人物匹配 | `pyannote_embedding` | `pyannote/wespeaker-voxceleb-resnet34-LM` | `.venvs/diarize/bin/python` |
| 转写和对齐 | `qwen` | `Qwen/Qwen3-ASR-1.7B`、`Qwen/Qwen3-ForcedAligner-0.6B` | `.venvs/asr/bin/python` |

四个默认阶段请求 `cuda:0`。分离另内置可选 `sepformer` 别名，需安装 `.venvs/separate-sb`。Ollama `qwen3:8b` 属于纪要 HTTP 客户端配置，不通过语音 worker 协议运行。

## 替换和接口

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

## 适配器的责任

- 只在本阶段进程内导入模型框架、加载权重；依赖缺失和设备不可用时明确报错，不回退为模拟结果。
- 保留主流程传入的 job ID，结果完整且唯一地覆盖请求；分离返回恰好两个不同的、已经存在的绝对文件路径，匹配的 `speakers` 顺序对应 `sources` 顺序。
- 默认处理协议使用 16 kHz 单声道音频；其他采样率模型负责内部重采样，输出重新满足协议。
- ASR 返回原录音坐标中的时间，`speaker` 与输入 job 一致；`words` 仅在模型确实提供对齐结果时填写。
- 匹配的 `uncertain` 若提供必须是布尔值，`margin` 若提供必须是有限非负数。
- 返回 `model` 以进入 manifest；配置引用的模型与真实加载实现应一致。

输入结构、分离/匹配组合和 ASR 时间戳校验位于 `pipeline.py`；后端注册和协议见 `backends/__init__.py`。先运行模型无关的契约测试，再在目标机器验收真实权重。依赖快照和更新说明见 [requirements](../requirements/README.md)，代码扩展工作流见 [贡献指南](../CONTRIBUTING.md)。

启动人物匹配前，流水线读取分离文件并核对 16 kHz、单声道、有限采样值与 job 时长。ASR 的 100 ms 边缘容差会裁剪到 job 和父片段边界；完全在范围外或裁剪后倒置的区间会失败。同一 worker 阶段重跑前清除旧响应，子进程必须新写标准 JSON 响应；成功响应为 `{"ok": true, "result": {...}}`，失败响应为 `{"ok": false, "error": "..."}`。非有限 JSON 数字和无效封套会失败，并给出本阶段日志路径。

`module:Class` 是受信配置的扩展入口，插件拥有本地 Python 代码权限。所有模型框架仍在阶段子进程内导入；本接口不提供不受信插件的沙箱。上游实现、模型卡及许可见[引用说明](../THIRD_PARTY_NOTICES.md)。

## LLM 纪要边界

`[summary]` 指定 Ollama 根地址、模型、超时、分块字符数和上下文长度。纪要输入是有片段 ID、人物、时间和不确定标记的转写；长输入按边界拆分后合并。请求禁用思考输出、使用固定温度，并设置 `keep_alive=0`。

纪要层验证完成状态、长度和引用片段 ID，原始 `text_raw` 不被改写。转写及模型笔记作为数据处理，不授权工具调用。引用有效不代表语义被证明，人工仍需核对事实、行动项及其证据。
