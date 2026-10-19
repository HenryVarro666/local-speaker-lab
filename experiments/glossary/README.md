# 普通话＋英文词表实验

在固定 Qwen3-ASR 和相同音频下，检验训练词表能否修正术语漏识，以及是否引入新的误识别。当前阶段只用 ASCEND，不新增个人录音。

当前方案：**固定 30 项训练词表、固定 200 条 validation、每条运行三组真实推理**。先完成诊断，再决定是否扩大；完整官方 test 尚未作为本轮测试。历史研究方案与数据集选择见 [设计记录](DESIGN_NOTES.md)，本页是当前运行手册。

## 实验与结果索引

| 记录 | 状态与用途 |
| --- | --- |
| [最初 24 条技术词表示例](ASCEND_PILOT_RESULTS.md) | 已完成；没有参考词表覆盖，只用于管线验证 |
| [24 条训练词表 v2](ASCEND_TRAIN_GLOSSARY_RESULTS.md) | 已完成；仅 3 次目标词提及，基线全部正确，尚不能检验纠错收益 |
| [200 条冻结协议](protocols/ascend-val200-train-vocab30-v2/README.md) | 固定词表、样本 ID、模型参数与分析计划；本轮扩展实验 |
| [200 条训练词表结果](ASCEND_200_RESULTS.md) | 600 次真实推理已完成；术语修复 0／5，新增 1 次词表错误替换；整体 MER 小幅下降但区间跨 0 |
| [6 个关键事件复核](ASCEND_SIX_EVENT_REVIEW.md) | 文本／计分／音频字节已核对；没有完成人工听音，声学判断保留未知 |
| [剩余 725 条冻结协议](protocols/ascend-val-remaining725-train-vocab30-v1/) | 用户授权后冻结全部剩余合格 validation；词表、主评分、模型及配置保持不变 |
| [本轮逐步记录](ASCEND_REMAINING_EXECUTION.md) | 改进理由、验证、来源与失败边界；机器时间线附 UTC 时间 |

本轮完整运行记录为 `runs/glossary-ascend-val200-train-vocab30-cpu-001/`；已核对 600 条完整配对、输入／音频哈希、保存分数和耗时。报告保留严格计分结果，同时说明写法变体；没有据结果调整词表、选样或主评分。

## 冻结设计

| 项目 | 规则 |
| --- | --- |
| 数据 | [CAiRE/ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND/tree/737e9800ae31be9932ba8464c80366559bd28424)，revision `737e9800ae31be9932ba8464c80366559bd28424`，CC BY-SA 4.0 |
| 词表来源 | 仅 train；mixed 文本中英文候选，去固定停用词和所有方括号标注跨度，文档频次至少 3；按文档频次、总频次、字典序取 20 单词＋10 双词短语 |
| 无关对照 | 独立固定科学／地理候选库，排除 train 词汇碰撞，匹配项数、token 数和近似字母长度 |
| 评价抽样 | validation 中 1–12 秒；按语言分别以 `(sha256(seed:id), id)` 排序，取 mixed 100、mandarin 50、english 50；seed `20261018` |
| 模型 | Qwen3-ASR-1.7B，revision `7278e1e70fe206f11671096ffdd38061171dd6e5`；CPU FP32，8 线程，Chinese，最大 256 token，greedy |
| 配对执行 | 每条全部三组；随机音频顺序及组内顺序；一次加载、一次不计分预热，无后处理 |
| 固定边界 | 不按参考答案、词表覆盖或识别错误挑样；不改词表／提示／模型；原 24 条包含在 200 条中 |

两份词表各 30 项、40 个规范化 token。相关词表 SHA256 为 `a8d5d34aca20cee94a3e265dbdec38a3fb725d2eb4f51eae93f23e248b9b8590`，无关词表为 `42b77131c7ed6c5e0cf7793438068eabb4183dc5d40eafbc800653b8c0d6daa5`。文件在 [冻结协议目录](protocols/ascend-val200-train-vocab30-v2/README.md)，与本次运行原始输入逐字节一致。

抽样完成后再统计覆盖：200 条共 725.2109375 秒，24 条含目标词，共 33 次提及，按主评分的最长非重叠匹配涉及 **13／30** 个词表项；其余 176 条为无目标词样本。若把 `smart phone` 内的 `smart`、`phone` 也分别算覆盖，则为 15 项，这是先前进度说明所用的描述性口径。正式结果统一采用 13 项；33 次主匹配、选样、词表和分数均未改变。这些覆盖数字没有用于选样或补词。

## 三组与评价

| 内部条件名 | ASR context |
| --- | --- |
| `none` | 空字符串 |
| `personal` | 全部固定 train 词表，逐行列出；沿用 CLI 命名，不代表个人录音词表 |
| `unrelated` | 全部固定无关词表，格式相同 |

重点报告相对 `none` 的配对变化：基线术语 **miss→hit**、原本正确术语 **hit→miss**，以及按参考位置对齐的错误词表提及新增／消失。错误提及分为替换、插入、目标短语残片等，不能都叫“凭空插入”。另报全体、各语言、含词／不含词切片的 MER、召回率、精确率及耗时。

评分规则为 `nfkc_han_latin_digits_v1`：NFKC、忽略大小写、汉字逐字、英文逐词、数字连续段、标点边界；术语按最长完整序列匹配，不合并缩写别名或翻译。参考中的既有 `[UNK]` 标注仍按固定规则计分。MER 汇总为总编辑数除以总参考 token 数，不平均每句百分比。该规则不声称复现 ASCEND 官方计分，不能直接与论文分数比较。

分析预先固定 1000 次按语言分层的配对 clip bootstrap，seed `20261018`；它没有建模同一说话者／会话相关性。一次 CPU 延迟包含 ASR 预处理和解码，不含文件读取、模型加载和预热；不代表 Mac 或 GPU 的产品延迟。

## 从新工作区复现

环境安装见 [GETTING_STARTED](../../docs/GETTING_STARTED.md)。以下命令均在仓库根目录执行，使用单行写法兼容 PowerShell 和 Bash。输出目录必须尚不存在；如已运行过，换一个新名称，保持下面各步骤路径对应。

### 1. 数据环境：导出官方 validation

激活数据准备环境后：

```text
python scripts/prepare_ascend_glossary.py --split validation --revision 737e9800ae31be9932ba8464c80366559bd28424 --output-dir experiments/glossary/data/repro-validation
```

这一步导出完整官方 validation（1130 条），再做固定抽样；不能用 `--limit 200` 代替分层哈希抽样。

### 2. 用已冻结词表固定选出 200 条

```text
python scripts/select_glossary_pilot.py --manifest experiments/glossary/data/repro-validation/manifest.jsonl --personal experiments/glossary/protocols/ascend-val200-train-vocab30-v2/correct.json --unrelated experiments/glossary/protocols/ascend-val200-train-vocab30-v2/unrelated.json --mixed-count 100 --mandarin-count 50 --english-count 50 --seed 20261018 --vocabulary-scope "Frozen ASCEND train-only glossary v2" --output-dir experiments/glossary/data/repro-val200-v2
```

生成 `selection.json`、`manifest.jsonl` 和词表副本。其 `ids` 应与版本化的 `selected_ids.json` 相同；选择函数只使用 ID、group 和音频时长。

### 3. ASR 环境：检查并运行

激活 ASR 环境后：

```text
python -m speaker_lab.glossary_experiment check --manifest experiments/glossary/data/repro-val200-v2/manifest.jsonl --personal experiments/glossary/data/repro-val200-v2/personal.json --unrelated experiments/glossary/data/repro-val200-v2/unrelated.json
python -m speaker_lab.glossary_experiment run --config configs/glossary_cpu.toml --manifest experiments/glossary/data/repro-val200-v2/manifest.jsonl --personal experiments/glossary/data/repro-val200-v2/personal.json --unrelated experiments/glossary/data/repro-val200-v2/unrelated.json --output-dir runs/repro-ascend-val200-001
```

固定模型第一次运行会下载权重；缓存完整后可加 `--offline`。真实推理共 600 次。不要同时启动其他重型运算：单次 CPU 延迟会受资源竞争影响。

### 4. 配对分析

只有 600 条记录完整后再执行；轻量开发／数据环境即可分析：

```text
python scripts/analyze_glossary_comparison.py --run-dir runs/repro-ascend-val200-001 --manifest experiments/glossary/data/repro-val200-v2/manifest.jsonl --correct experiments/glossary/data/repro-val200-v2/personal.json --unrelated experiments/glossary/data/repro-val200-v2/unrelated.json --bootstrap-replicates 1000 --bootstrap-seed 20261018
```

分析器拒绝不完整配对、冻结输入／音频哈希不符或已有输出；生成 `comparison.json`，包括逐句变化、术语纠正／退化、错误提及、切片、置信区间和源码哈希。若指定 `--old-run-dir`，旧 24 条仅用于基线一致性核查，不作为独立重复实验。

### 可选：从 train 重建词表以审计来源

数据环境执行，仍然不需要 train 音频：

```text
python scripts/export_ascend_train_text.py --output-dir experiments/glossary/data/repro-train-text
python scripts/build_ascend_train_glossary.py --train-jsonl experiments/glossary/data/repro-train-text/train.jsonl --output-dir experiments/glossary/data/repro-train-vocab30-v2
```

导出器固定同一数据 revision 的全部三个 train parquet 分片，只用 HTTP byte range 读取文本列。检查重建 `correct.json`、`unrelated.json` 的哈希与冻结副本相同；`extraction.json` 保存规则、候选频次、train 来源 ID 和分配审计。本轮不使用重建结果替换已经冻结的运行输入。

## 文件与后续开发边界

| 位置 | 内容 | 版本管理 |
| --- | --- | --- |
| `src/speaker_lab/glossary_*.py` | 实验 runner 与纯计分 | 代码 |
| `scripts/*glossary*.py`、`export_ascend_train_text.py` | 导出、选样、建词表和分析 | 代码 |
| `protocols/`、`*.example.*` | 固定规则、词表、样本 ID、格式样例 | 小型审阅材料 |
| `data/` | 下载音频、完整转写、派生清单 | 本地，Git 忽略 |
| 根 `runs/` | 原始预测、环境来源、分数和逐句分析 | 本地，Git 忽略 |
| `ASCEND_*_RESULTS.md` | 已完成实验的结论和限制 | 文本报告 |

大产物不进入源码仓库；文本报告必须标明 run ID、词表／模型／数据版本和样本范围。示例 manifest 没有配套音频，只演示格式。

后续 Mac 后端沿用相同 manifest、context 和 predictions 接口，平台内先完成三组对照；它目前尚未实现。是否扩大到完整 validation／封存 test，应先根据本轮术语错误分母、修正／新增错误和不确定性决定，不按验证答案补词或选难例。
