# ASCEND 剩余 validation 725：冻结协议 v1

本协议在新一轮模型推理前冻结。先对旧 200 条的 [6 个事件进行证据复核](../../ASCEND_SIX_EVENT_REVIEW.md)，然后以原来的词表、主评分和模型配置，对所有剩余合格片段作三条件成对复验。**没有完成人工听音；3 个词语／单复数事件的声学真值未裁定。** 这些未知不改变本轮任何输入和判断规则。

## 冻结对象

| 文件 | 用途 |
| --- | --- |
| `protocol.json` | 数据、选样、模型、条件、输入与源码哈希、禁止的改动 |
| `analysis_plan.json` | 预定主结果、辅助结果、区间、完整性门槛和失败策略 |
| `selected_ids.json` | 新 725 条可移植 ID；与旧 200 条无交集 |
| `selection_audit.json` | 1130→925→725 选择轨迹；925 份合格音频的字节哈希 |
| `correct.json` / `unrelated.json` | 与旧协议逐字节相同的两份 30 项词表 |
| `config.toml` | 原 CPU pilot 配置的原样副本；模型缓存路径相对项目根目录 |
| `environment.json` | 原推理环境包版本、Python、缓存模型文件哈希及环境检查 |
| `FREEZE.json` | 上述文件及本文的最终哈希清单；冻结后不修改 |

官方 validation 共 1130 条：沿用音频头 **1≤时长≤12 秒**，合格 925 条；排除旧冻结 200 条后全部保留 725 条（mixed 107／普通话 453／英文 165），共 2409.865375 秒。选样只使用 ID、语言组、音频头时长；参考文本仅在本地运行清单中原样复制，不影响筛选或排序。没有用词表覆盖、模型错误或 test。

固定 Qwen3-ASR-1.7B revision `7278e1e70fe206f11671096ffdd38061171dd6e5`，原环境 CPU FP32、8 线程、Chinese、greedy、256 token、seed 20261018。每段重跑无词表／train 词表／无关词表，共 **2175 次**计时推理，另有一次不评分预热。`personal` 只是既有 CLI 条件名，此处仍是 train 高频词表。

## 复现

先依照 [词表实验说明](../../README.md)准备原固定 ASCEND 数据和模型环境，然后在项目根目录运行。所有输出目录都必须是新目录；原始录音、完整参考及预测保留本地，不进入源码 ZIP 或 Git。

```bash
python scripts/freeze_ascend_remaining.py \
  --source-manifest experiments/glossary/data/ascend-validation-737e9800/manifest.jsonl \
  --prior-protocol experiments/glossary/protocols/ascend-val200-train-vocab30-v2 \
  --output-dir experiments/glossary/data/ascend-validation-remaining725-train-v2 \
  --seed 20261018
```

新机器重新生成的 `selected_ids.json` 应与此处完全相同，音频哈希也应相同。本地运行 manifest 含绝对音频路径，其字节哈希只描述本次记录，跨机器须重新生成，不能直接使用其他人的路径。将两份冻结词表与下列配置交给原 runner；模型路径按 `config.toml` 缓存布局准备，不在本轮修改推理参数。

```bash
python -m speaker_lab.glossary_experiment run \
  --config experiments/glossary/protocols/ascend-val-remaining725-train-vocab30-v1/config.toml \
  --manifest experiments/glossary/data/ascend-validation-remaining725-train-v2/manifest.jsonl \
  --personal experiments/glossary/protocols/ascend-val-remaining725-train-vocab30-v1/correct.json \
  --unrelated experiments/glossary/protocols/ascend-val-remaining725-train-vocab30-v1/unrelated.json \
  --output-dir runs/glossary-ascend-remaining725-train-vocab30-cpu-001 \
  --seed 20261018 --offline
```

仅当 run 为 complete、2175 条完整唯一配对、输入／音频／源码哈希与保存分数一致，才运行：

```bash
python scripts/analyze_glossary_comparison.py \
  --run-dir runs/glossary-ascend-remaining725-train-vocab30-cpu-001 \
  --manifest experiments/glossary/data/ascend-validation-remaining725-train-v2/manifest.jsonl \
  --correct experiments/glossary/protocols/ascend-val-remaining725-train-vocab30-v1/correct.json \
  --unrelated experiments/glossary/protocols/ascend-val-remaining725-train-vocab30-v1/unrelated.json \
  --bootstrap-replicates 1000 --bootstrap-seed 20261018
```

失败时停止，保留部分预测、stdout、stderr 和退出状态；不对不完整输出给主结论，不覆盖或选择性重跑。重试必须另建 run，记录原因并保持同一冻结协议。详细操作记录放在 [execution_log.jsonl](../../audits/ascend-remaining-v1/execution_log.jsonl)，冻结文件自身不随运行更新。

## 解释边界

主结果仍为术语修复、破坏、错误候选的增减；MER、语言和覆盖切片、计时为辅助。区间沿用 1000 次、seed 20261018 的语言分层成对 clip bootstrap。没有说话人／会话 ID，不能保证与开发集说话人独立，也不能做可靠的说话人聚类区间。

旧 200 与新 725 分开报告；旧 24 已包含在旧 200 中。两批语言配比不同，绝对 MER 的跨批差异不解释为词表作用。若报告 925 合并数字，只作混入开发片段的补充描述。本次不更换词表、不放宽拼写／别名／翻译／单复数匹配、不修订参考、不使用 LLM、不据结果加样。

数据来自 [CAiRE/ASCEND 固定 revision](https://huggingface.co/datasets/CAiRE/ASCEND/tree/737e9800ae31be9932ba8464c80366559bd28424)，原转写 CC BY-SA 4.0；模型与项目原创代码的许可分别见 [第三方声明](../../../../THIRD_PARTY_NOTICES.md)。本实验评分不是 ASCEND 官方 recipe，不直接与论文分数横比；本机 CPU 结果也不构成 Ubuntu／RTX 5080 实测。
