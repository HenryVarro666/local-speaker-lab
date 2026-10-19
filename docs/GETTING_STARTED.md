# 开发与本地运行

所有命令在仓库根目录 `local-speaker-lab/` 执行。先选择一个入口：纯代码开发无需模型；词表实验只需要 ASR；双人流程另有独立环境。[双人流程指南](TWO_SPEAKER_GUIDE.md) 和 [Ubuntu 部署](../DEPLOYMENT.md) 不属于下面的单人实验安装步骤。

## 轻量开发环境

需要 Python 3.11 或 3.12。创建环境后，后续统一使用其中的 `python`：

```text
python -m venv .venv-dev
```

Windows PowerShell 激活：

```powershell
.\.venv-dev\Scripts\Activate.ps1
```

Linux / macOS 激活：

```bash
source .venv-dev/bin/activate
```

```text
python -m pip install --upgrade --only-binary=:all: -r requirements/bootstrap.txt
python -m pip install --build-constraint requirements/bootstrap.txt -r requirements/dev.txt
python -m pip install --no-build-isolation -e ".[dev]"
python -m unittest discover -s tests -p "test_*.py"
python -m pytest -q
```

这些测试使用合成输入、固定样例和 mock，不下载模型或 ASCEND，也不代表真实识别质量或 GPU 验收通过。分发测试会把源码 ZIP 构建为 sdist/wheel，并在新的环境中离线安装检查；固定构建工具因此也是开发环境的一部分。部分可选组件检查会按安装情况跳过；完整 CI 与 Ubuntu 锁定应用依赖的检查见[开发与验证](development.md)。

## 单人词表实验：Windows / Linux CPU

这条路线与本次 Windows CPU 实验使用同一 Torch/Qwen 版本。它适合先复现完整流程；当前未实现 Mac MLX 推理后端，也没有 Mac 性能结果。

先创建 `.venv-asr`：

```text
python -m venv .venv-asr
```

按上面的对应平台方法激活它（把路径中的 `.venv-dev` 换成 `.venv-asr`），再安装：

```text
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . -r requirements/asr.txt
python -m pip check
```

安装 Torch CPU wheel 后再安装 ASR 依赖。不要把 `requirements/locks/*linux-py312.txt` 直接用于 Windows，也不要在 ASR 环境中安装双人流程的全部模型依赖。

可移植的固定配置是 [configs/glossary_cpu.toml](../configs/glossary_cpu.toml)，模型为 Qwen3-ASR-1.7B，revision 为 `7278e1e70fe206f11671096ffdd38061171dd6e5`，CPU FP32、8 线程、Chinese、最多 256 个生成 token。`glossary_cpu_pilot.toml` 则记录本机已有缓存位置，两份配置的实验参数相同。

第一次运行需要联网下载模型。若希望把缓存留在仓库内，启动前设置：

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
```

```bash
export HF_HOME="$PWD/.cache/huggingface"
```

缓存完整后才可加 `--offline`。模型目录、环境和运行结果均不进入 Git。

## 准备数据、运行和分析

数据准备使用另一个 `.venv-data`，避免改变 ASR 依赖。创建并激活后执行：

```text
python -m pip install -e . -r requirements/glossary-data.txt
```

完整的固定数据导出、词表重建、200 条抽样、三组推理与配对分析命令见 [实验运行手册](../experiments/glossary/README.md)。先用 `check` 验证清单和两份词表，再启动 `run`；每个复现使用新的输出目录，保留原始预测。

无需安装大模型即可查看入口：

```text
python -m speaker_lab.glossary_experiment --help
python scripts/select_glossary_pilot.py --help
python scripts/build_ascend_train_glossary.py --help
python scripts/analyze_glossary_comparison.py --help
```

运行目录应包含 `run.json`、`predictions.jsonl` 和完成后生成的 `scores.json`。`run.json` 状态完成、200 个 ID 的三组预测全部齐全后，再生成 `comparison.json`。运行失败时保留已有记录，不把部分样本的分数作为完整结果；当前 runner 没有自动续跑。

## 平台边界

| 路线 | 当前状态 | 入口 |
| --- | --- | --- |
| 单人词表实验 / Windows CPU | 已完成 24 条及固定 200 条三组真实推理，后者共 600 次 | [200 条结果](../experiments/glossary/ASCEND_200_RESULTS.md) |
| 单人词表实验 / Linux NVIDIA | 已有 Transformers 配置，尚未提供本实验 GPU 实测结果 | [glossary.toml](../configs/glossary.toml) |
| 双人分离、转写、纪要 / Ubuntu NVIDIA | 已有实现和部署检查；目标 GPU 完整真实验收待执行 | [双人指南](TWO_SPEAKER_GUIDE.md) |
| 单人词表实验 / Apple Silicon | 后端待开发，沿用相同数据和计分接口 | [路线图](ROADMAP.md) |

轻量开发命令可用于 Mac；这不等于 Qwen 的 Mac 推理路径已经实现或验证。跨平台质量对比应固定模型、精度和解码设置，先各自完成三组配对实验。
