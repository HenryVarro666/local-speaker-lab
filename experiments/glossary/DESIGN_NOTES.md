> 早期研究方案与数据集调研记录。当前执行方案见 [README](README.md)；当前阶段不新增个人录音。

# 普通话和英文混合听写的个人词表实验

> **在固定 Qwen3-ASR 和相同音频的条件下，预先冻结的个人术语词表，能否减少术语漏识，同时不增加未说术语的误插入？**

实验先使用现有 Ubuntu / NVIDIA 环境中的 Qwen3-ASR-1.7B。数据清单、原始预测和计分器保持平台独立，后续可接 Mac 的 MLX 后端。个人录音回答“我的术语是否识别得更准”，公开语料回答“通用中英混说是否退化”。两套结果分别报告。

词表可能帮助识别少见英文名称，也可能使模型输出音频中不存在的名称。Qwen 支持 ASR 输入端的文本 `context`，这是软性偏置，不是必须命中词表的解码约束。这里测纯词表对原始识别的影响，使用简短词表文本；模型报告指出它不遵循通用自然语言指令，因此不依赖“请保留英文”等提示。[官方源码](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)、[技术报告 §2.2](https://arxiv.org/html/2601.21337v1)

## 数据集选择

资料核实日期：2026-10-18。下表依据官方数据页和论文；不代表已下载音频或复现其成绩。

| 数据集 | 语料与标准划分 | 获取条件 | 本实验用途 |
| --- | --- | --- | --- |
| [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND) / [LREC 2022](https://aclanthology.org/2022.lrec-1.788/) | 普通话和英文自然对话，10.62 小时、23 人；官方 train / validation / test。香港收集，包含内地、台湾和香港说话者 | CC BY-SA 4.0，公开下载。当前 validation 和 test 各约 107 / 106 MB | 第一阶段首选。保留官方划分；先开发集，冻结后跑完整测试集，并单报 `mixed` 子集 |
| [CS-Dialogue](https://huggingface.co/datasets/BAAI/CS-Dialogue) / [论文](https://arxiv.org/html/2502.18913v2) | 104.02 小时、200 人，来自大陆 30 个省级地区；说话者独立的 train / dev / test | CC BY-NC-SA 4.0，限非商业；短音频分片合计约 9.6 GB | 第二阶段外部测试，口音分布更贴近大陆普通话用户 |
| [SEAME](https://catalog.ldc.upenn.edu/LDC2015S04) | 经典中英混说基准，新加坡及马来西亚口音；LDC 原始录音约 192 小时，常用处理后训练集约 100 小时 | LDC 授权获取；公开划分文件不包含免费音频 | 已有机构授权时补充，不阻塞最小实验 |

ASRU 2019 也是正式混说竞赛语料，但本次未核实当前直接开放下载及再利用许可，暂不设为依赖。没有在这些资料中确认可直接评价“你的个人技术词表”的标准测试集；它们主要评价通用混说识别，需要补充个人录音。

ASCEND 并非每句都中英混说：全库约 27% 为 mixed。官方 HF 当前 validation / test 为 1130 / 1315 条，原论文 validation 表格为 1129；以锁定 revision 的实际数据为准。Importer 会记录 revision 和导出条数。不要把前 N 条子集结果标为完整 benchmark，也不要将 ASCEND 的香港采集地点理解为粤语语料。[官方仓库](https://github.com/HLTCHKUST/ASCEND)

## 最小实验数据

1. 从已有项目、笔记和常用应用名称整理 **20–30 个术语**。在查看测试转写和模型输出前冻结词表、拼写与顺序。目录中的两个 `*.example.json` 仅为 10 项格式示例，应替换为实际词表。
2. 建立等项数、长度大致相近、英文词和缩写比例相近的无关词表；两份列表不重叠。所有音频始终使用同一完整词表，不能根据每条参考答案选择“相关词”。
3. 录制 **20 条开发集 + 100 条测试集**，每条约 5–12 秒。测试集暂设 60 条含术语、40 条不含词表术语；覆盖不同语序、短英文片段、缩写及普通中文表达。尽量让每个目标词在不同句式中出现至少 3 次；不足时减少词表或增加录音。
4. 根据旧笔记或任务提纲自然口述；为覆盖词表而设计的句子属于受控测试，不能代表日常错误发生率。参考转写由人回听确认，不能拿模型输出直接当标准答案。一个人的结果只支持个人可行性判断。
5. 用开发集选择 `language = "Chinese"` 或 `language = ""`（自动），并确定词表格式。选定后固定模型、revision、精度、语言、处理方式和最大输出长度，再处理封存测试集。

公开语料的主回归实验也使用同一固定个人词表，检测它是否妨碍普通对话。若另做公开语料的“领域词表收益”实验，只能从 train 或事前独立词典构建词表，不能从 validation/test 参考答案反向抽取。只有 validation+test 下载也足够运行个人词表回归检查。

## 三组对照

| 条件 | 传给模型的 context | 回答的问题 |
| --- | --- | --- |
| `none` | 空字符串 | 当前模型基线 |
| `personal` | 固定个人词表，逐行列出 | 实际部署词表是否有效 |
| `unrelated` | 等项数的固定无关词表，使用相同格式 | 收益是否依赖正确术语，以及模型是否被错误上下文诱导 |

每条音频跑全部三组，随机化音频顺序和每条音频内的条件顺序。模型只加载一次，先预热再计时。仅执行单人音频 ASR，不加载分离、人物识别、ForcedAligner 或总结模型，也不做 LLM 改写、翻译、词表替换。这样差异才能归因于识别时的 context。

## 计分与判断

主要比较个人词表组相对无词表组的 **术语漏识率 = 1 − term_recall**。同步检查：

- **个人术语 precision / recall**：基于参考与预测的最小编辑对齐，按完整术语提及计数；错误术语替换会产生一次漏识和一次误命中。
- **混合错误率 MER = (S + D + I) / N**：汉字逐字、英文逐词。汇总使用总编辑数除以总参考 token 数，不平均每句百分比。
- **未说词误插入**：对不含候选术语的录音，报告误插入数、每 100 条录音的误插入数和至少一次误插入的录音比例。所有组使用相同候选集合；`all_candidates` 额外评价个人词与无关词的并集，捕捉无关词表诱导。
- **热运行延迟 p50 / p95 和 RTF**：包括 ASR 预处理和解码，不含文件读取、首次下载、模型加载和预热。加载与预热单列；这不是完整 Mac 输入流程延迟。

当前计分器规则名 `nfkc_han_latin_digits_v1`：NFKC、忽略大小写、标点作为边界、数字连续段为一个 token；不把 `API` 与 `A P I` 合并，不把中文翻译当英文术语命中。保留原文供复核。此规则便于跨平台配对，但不声称逐项复现 ASCEND 官方分词/归一化；与论文分数直接比较时应另跑其[官方计分代码](https://github.com/HLTCHKUST/ASCEND/blob/main/utils.py)。

`all_candidates` 是固定候选并集下的诊断指标。术语按最长完整序列匹配，存在重叠短语时，其提及数可能与仅评价个人词表不同；只在相同评分范围内比较三组。

试验前可暂定扩大实验的门槛：术语漏识率相对下降至少 20%，整体 MER 增量不超过 0.5 个百分点，且在同一负例集合没有观察到更多误插入。这些是可随实际用途调整的实用阈值，不是文献证明的标准。若基线术语错误极少，先增加难例和样本量，不能用接近零的分母宣称大幅改善。

逐句配对检查改正和引入的错误。扩大样本后再补配对 bootstrap 置信区间和多说话者验证；当前 100 条个人测试不支持普遍性结论，40 条负例也不能证明罕见误插入风险很低。人工纠错时间作为后续产品评测，当前自动计分器不生成该指标。

## 本地运行

以下命令在项目根目录执行，复用原 `.venvs/asr`。Runner 延续项目固定的 `qwen-asr==0.0.6` 和 Transformers 4.57.6 环境；不需要云端 API。

先在单独的数据准备环境导出 ASCEND，避免给 ASR 环境增加依赖：

```bash
python3.12 -m venv .venvs/glossary-data
.venvs/glossary-data/bin/python -m pip install -e . -r requirements/glossary-data.txt
.venvs/glossary-data/bin/python scripts/prepare_ascend_glossary.py \
  --split validation --output-dir experiments/glossary/data/ascend-validation
.venvs/glossary-data/bin/python scripts/prepare_ascend_glossary.py \
  --split test --output-dir experiments/glossary/data/ascend-test
```

第一次解析官方数据 revision 后，把 `dataset.json` 中的 SHA 用 `--revision SHA` 固定到下一份导出。`--limit 20` 仅用于管线检查；正式测试使用完整官方 test。输出目录必须是新目录，防止覆盖。

个人录音使用与导出数据相同的 JSONL 格式：

```json
{"id":"me-test-001","audio":"audio/me-test-001.wav","reference":"这个 API 的 latency 有点高。","split":"test","group":"term_mixed"}
```

`audio` 相对清单所在目录，必须是非空、16kHz、单声道音频。将开发集和测试集放在不同的 manifest 文件。示例清单没有对应音频，不是可运行测试数据。

```bash
# 先将两个 example.json 复制成 personal.json / unrelated.json 并冻结实际词表。
.venvs/asr/bin/python -m speaker_lab.glossary_experiment check \
  --manifest experiments/glossary/data/ascend-test/manifest.jsonl \
  --personal experiments/glossary/personal.json \
  --unrelated experiments/glossary/unrelated.json

.venvs/asr/bin/python -m speaker_lab.glossary_experiment run \
  --config configs/glossary.toml \
  --manifest experiments/glossary/data/ascend-test/manifest.jsonl \
  --personal experiments/glossary/personal.json \
  --unrelated experiments/glossary/unrelated.json \
  --output-dir runs/glossary-ascend-test-001
```

模型首次使用需要下载权重；之后可加 `--offline` 只使用已有缓存。`configs/glossary.toml` 可以填本地模型目录，或将 `revision` 固定为首次解析到的模型 commit。模型下载时间不包含在识别延迟中。运行会保存：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 模型位置、软件版本、配置/词表/清单/音频 SHA256、完整 context、解码设置、加载和预热耗时 |
| `predictions.jsonl` | 每条音频各条件的原始预测与耗时，逐条写入 |
| `scores.json` | 按全体、group、split 汇总的 MER、个人术语及候选并集指标、热延迟和 RTF |

中途失败会保留记录，不标成完整运行。计分器拒绝缺失某组的配对结果，不把漏掉的难例排除后出分。暂不实现自动续跑。`score` 子命令也可给未来 Mac 后端的同格式预测计分。

## Mac 开发路径

保持相同 manifest、词表文本和 predictions JSONL 接口，再增加独立 Mac 推理适配器。Apple Silicon 可考察社区 [MLX-Audio Qwen3-ASR](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/stt/models/qwen3_asr)，将同一 context 文本传为 `system_prompt`；它不是 Qwen 官方 Transformers 后端，应使用独立虚拟环境，不与原 ASR 环境混装。

先在每个平台内分别做三组对照。不要把 CUDA 1.7B 无词表与 Mac 0.6B 量化词表组直接比较，否则模型大小、精度和后端都会成为混杂变量。当前尚未确定用户 Mac 芯片/内存，也未实现或测量 Mac 后端；不承诺本机实时性能。0.6B 可作为资源不足时的另一套独立实验。

## 实现与验收范围

本目录提供实验方案、词表示例和数据格式。新增的本地入口为 `speaker_lab.glossary_experiment`，计分模块为 `speaker_lab.glossary_metrics`，公开数据导出入口为 `scripts/prepare_ascend_glossary.py`。

工程验收包括纯计分、清单/对照完整性、公开数据导出逻辑的离线单元测试。这些检查不代替真实 Qwen 推理；只有在目标机器跑完配对音频后，才能填写词表收益、模型耗时及显存结果。
