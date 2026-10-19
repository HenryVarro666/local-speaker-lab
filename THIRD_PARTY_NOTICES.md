# 上游项目、模型与数据引用

核对日期：2026-10-19。Local Speaker Lab 提供本地流水线编排、跨环境适配、界面、部署验收和实验工具；语音与语言模型能力来自以下上游。感谢原作者和维护者。本项目为独立集成项目，与这些团队没有隶属或背书关系。

本仓库原创代码采用 [MIT](LICENSE)。第三方软件、预训练模型和数据保留各自的版权与许可；项目的 MIT 不替代它们的许可。本源码包不捆绑第三方仓库源码、模型权重、语音数据或 FFmpeg/Ollama 二进制。运行时由包管理器和模型客户端从上游安装、下载。

## 直接使用的软件

| 上游与许可原文 | 用途 | 本仓库的集成位置 |
| --- | --- | --- |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) · [MIT](https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE) | 人物分段与说话人嵌入推理 | `src/speaker_lab/backends/pyannote.py` |
| [ClearerVoice-Studio / ClearVoice](https://github.com/modelscope/ClearerVoice-Studio) · [Apache-2.0](https://github.com/modelscope/ClearerVoice-Studio/blob/main/LICENSE) | MossFormer2 双人分离及 NumPy 推理接口 | `src/speaker_lab/backends/clearvoice.py` |
| [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) · [Apache-2.0](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE) | Qwen ASR、强制对齐、词表上下文实验 | `src/speaker_lab/backends/qwen.py`、`glossary_experiment.py` |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) · [Apache-2.0](https://github.com/speechbrain/speechbrain/blob/develop/LICENSE) | 可选 SepFormer 分离后端 | `src/speaker_lab/backends/sepformer.py` |
| [Ollama](https://github.com/ollama/ollama) · [MIT](https://github.com/ollama/ollama/blob/main/LICENSE) | 独立本地 LLM 服务，通过 HTTP 生成纪要 | `src/speaker_lab/summarize.py` |
| [Gradio](https://github.com/gradio-app/gradio) · [Apache-2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE) | 本地交互界面 | `src/speaker_lab/app.py` |
| [PyTorch](https://github.com/pytorch/pytorch) · [BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE) | 独立模型环境的张量与 CUDA 运行时 | `requirements/` 与各模型适配器 |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) · [上游许可说明](https://ffmpeg.org/legal.html) | 外部命令进行音频归一化；具体许可取决于所安装构建及其组件 | `src/speaker_lab/audio.py` |

这张表记录主要直接集成，不是完整的传递依赖清单。全部环境版本要求见 [requirements](requirements/README.md)；实际安装的依赖版本由部署报告中的 `installed-versions/` 保存，各包的许可文件随其发行物提供。没有把上述模型架构重新实现或标作本项目原创。

## 默认与可选预训练模型

以下许可来自对应官方模型卡，分别于核对日期查看。更换 checkpoint 时应同时更新此表和配置，不能将某个模型的许可推广到整个模型系列。

| 模型与官方来源 | 在本项目中的用途 | 模型卡许可 |
| --- | --- | --- |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | 默认人物分段和重叠检测 | CC-BY-4.0；下载需由账号本人接受访问条件 |
| [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | 分离来源与人物匹配；上游为 [WeSpeaker](https://github.com/wenet-e2e/wespeaker) 的模型封装 | CC-BY-4.0 |
| [alibabasglab/MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K) | 默认 ClearVoice 分离权重 | Apache-2.0 |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 默认转写与单人词表实验 | Apache-2.0 |
| [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | 可选字词时间对齐 | Apache-2.0 |
| [speechbrain/sepformer-wsj02mix](https://huggingface.co/speechbrain/sepformer-wsj02mix) | 可选 8 kHz 分离路线 | Apache-2.0 |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Ollama 默认 `qwen3:8b` 对应模型系列；实际制品 digest 由验收记录 | Apache-2.0 |

Ollama 模型标签和量化制品也有自己的元数据；具体部署以实际模型 manifest 和许可为准。项目不通过源码发布来转授模型访问权限。

## 数据与衍生实验材料

单人词表实验使用 [CAiRE/ASCEND 官方数据集](https://huggingface.co/datasets/CAiRE/ASCEND)及 [HLTCHKUST/ASCEND 仓库](https://github.com/HLTCHKUST/ASCEND)。数据集由 Holy Lovenia 等作者发布，官方标明 [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)。论文为 [ASCEND: A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation，LREC 2022](https://aclanthology.org/2022.lrec-1.788/)。

本仓库的实验报告含少量注明来源的参考文本示例；冻结协议保留样本 ID、配置和从官方 train 文本按固定规则计算的词表。这些数据引用及衍生数据材料保留 ASCEND 的 CC-BY-SA-4.0，适用范围与原创程序代码的 MIT 分开。处理过程为训练文本导出、标注跨度移除、频次排序和固定抽样，规则与 revision 见[实验说明](experiments/glossary/README.md)及[冻结协议](experiments/glossary/protocols/ascend-val200-train-vocab30-v2/README.md)。没有分发原始音频、完整转写语料或下载缓存。

## 学术使用与后续扩展

评价或发表模型结果时，请引用所用上游模型卡列出的论文，以及 ASCEND 数据论文。项目方法入口包括 [ClearerVoice-Studio 论文](https://arxiv.org/abs/2506.19398)、[Qwen3-ASR 技术报告](https://arxiv.org/abs/2601.21337)和 [SepFormer 论文](https://arxiv.org/abs/2010.13154)；人物分段与 WeSpeaker 的具体论文由上述官方模型卡列出。

引用本项目的软件元数据见 [CITATION.cff](CITATION.cff)。新增后端、复制或修改上游源码、引入新数据前，应保留其原版权声明，并在此记录来源、使用方式与适用许可；不能只给本项目的 MIT 标签。
