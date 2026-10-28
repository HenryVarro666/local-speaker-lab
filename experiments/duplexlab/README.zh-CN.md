# Speech Data Curation Lab · 语音数据整理实验

[English](README.md) · 简体中文

Speech Data Curation Lab（原展示名 DuplexLab）在原 `speaker_lab` 包内新增独立实验入口，当前已实现离线数据闭环：冻结与校验 → 音频审计和精确去重 → 独立模型进程 → 原始转写 → 分组计分 → 本地回听与人工纠错记录。原双人和 glossary 入口保持可用。

实现入口仍为现有 [Local Speaker Lab 仓库](https://github.com/HenryVarro666/local-speaker-lab) 中的 `speaker_lab.duplex` 与 `experiments/duplexlab/`。公开备份仅含已审查源码和汇总证据，原音、权重与运行状态留在本地。2026-10-27 本次只复核既有证据，无新训练或模型调用。

## 最新：归一化与词元加权

十二次基座新训完成：随机原CE / 筛选原CE / 同筛选全1归一化CE / 同筛选2倍词元加权CE，各500条、63步、seeds42/43/44。新test362条、10个当前拆分隔离speaker IDs，平均MER **9.7315% / 9.8353% / 9.8353% / 10.1329%**；三次总删除 **221 / 237 / 237 / 271**。归一化对照三seed对应权重逐字节相同、batch1输入1242对全同，最终转写也相同；保存tokenizer的左右padding侧效应另记，不外推到ragged批处理。加权−全1归一化 **+0.2976pp**，配对seed/整说话人区间 **[-0.0556,+0.6673]pp**；筛选−随机 **+0.1038pp**、区间 **[-0.3246,+0.5328]pp**。区间包含零，未证实总体加权收益。 train1015/dev52与上一轮数据及选数不变，全部12组dev先完成才读新test；10个IDs有历史项目曝光，不称全项目未见说话人，源参考不改、人工听音pending。 [Results](reports/2026-10-21-normalization/RESULTS.md) · [Diagnosis](reports/2026-10-21-normalization/REGRESSION_ANALYSIS.md) · [Reproduction](reports/2026-10-21-normalization/REPRODUCE.md).

`prepare_duplexlab_normalization.py --root . --shards /path/to/pinned/ASCEND/main` → input/loss preflight → `run_duplexlab_normalization.py --root . --model /path/to/pinned/snapshot` → `verify_duplexlab_normalization.py` → `summarize_duplexlab_normalization.py` → `report_duplexlab_normalization.py`.

## 上轮：删除保留与训练目标消融

删除保留三臂消融已完成九次基座新训（随机原loss / 改进筛选原loss / 同筛选加权loss，各500条、63步、42/43/44）。新test280条/7个当前拆分隔离speaker IDs，平均MER **10.7691% / 10.3624% / 10.9632%**。联合候选−随机 **+0.1941pp**，配对seed/整说话人区间 **[-0.7538,+1.1337]pp**；三次总删除 **284 / 270 / 295**。删除和整体编辑必须一起判断，不能凭更少D直接称内容保留修复。train1015、dev提前统一40+12=52，所有九组dev重载后才读新测试；新录音排除此前ID/文件/PCM，7位有项目历史曝光，不能称全项目未见说话人。源参考和严格主分不改，人工听音pending。区间包含零，未建立可靠总体优势。 同数据加权目标三个seed总编辑均更多（平均MER+0.6009pp），不支持采用本轮加权目标；筛选原CE的填充词删除25→35，仍有退化。 [结果](reports/2026-10-20-deletion/RESULTS.md) · [删除核对](reports/2026-10-20-deletion/REGRESSION_ANALYSIS.md) · [复现](reports/2026-10-20-deletion/REPRODUCE.md).

`analyze_duplexlab_deletions.py` → `prepare_duplexlab_deletion.py --root . --shards /path/to/pinned/ASCEND/main` → input/loss preflight → `run_duplexlab_deletion.py --root . --model /path/to/pinned/snapshot` → `verify_duplexlab_deletion.py` → `summarize_duplexlab_deletion.py` → `report_duplexlab_deletion.py`.

## 上轮：填充词频次、重复与切换跨度

频次/重复/切换跨度约束实验已完成六组基座新训（各500条/63步、42/43/44）。新test320条/4位当前拆分隔离说话人上，随机/新策略平均MER **12.8807% / 12.6199%**，差 **-0.2608pp**，配对seed/说话人区间 **[-0.9295,+0.2436]pp**；区间仍包含零，未建立可靠优势。填充词64条均值19.6048%→18.8248%。三次总删除296→318，填充词token删除26→25；未证实口语保留已修复。两臂既有覆盖和新增频次/重复总数/切换跨度边际一致，音频差0.9643%。原dev40保留；另冻12条补足重复等缺口，数据在新test前冻结、诊断在主流程后运行，不改训练eval_file。测试speaker13/14/18/24有历史确认/测试曝光，不称整个项目新说话人。 [结果](reports/2026-10-20-disfluency/RESULTS.md) · [旧退化核对](reports/2026-10-20-disfluency/REGRESSION_ANALYSIS.md) · [复现](reports/2026-10-20-disfluency/REPRODUCE.md)。

`analyze_duplexlab_disfluency.py` → `prepare_duplexlab_disfluency.py --root . --shards /path/to/pinned/ASCEND/main` → `run_duplexlab_disfluency.py --root . --model /path/to/pinned/snapshot` → `run_duplexlab_auxdev.py --root . --model /path/to/pinned/snapshot` → `verify_duplexlab_disfluency.py` / `verify_duplexlab_auxdev.py` → `summarize_duplexlab_disfluency.py` → `report_duplexlab_disfluency.py --root .`.

## 上轮：覆盖约束与三seed基座新训

本轮覆盖筛选已完成六次基座新训（42/43/44、各500条/63步），在新300条/3位当前拆分隔离说话人上，随机/覆盖平均MER **8.8984% / 8.6209%**，差 **−0.2774pp**，配对seed/说话人区间 **[−0.5559,0.0000]pp** 仍包含零。三seed整体均改善，填充词91条平均却从10.4099%升至10.5731%。说话人/难度/填充词/语言时长边际严格相同，音频预算差0.954%；dev40保留并先于test冻结。测试speaker10/16/26曾在旧训练池，仅声明本轮train/dev/test隔离；未复用旧SFT，不能声称整个项目从未见过这些人。 [结果](reports/2026-10-20-coverage/RESULTS.md) · [约束设计](reports/2026-10-20-coverage/DESIGN.md) · [协议](COVERAGE_PROTOCOL.md)。

## 上轮：纯参考一致性筛选

42/43/44三个配对seed已评测同一冻结新300条、3位此前未用官方speaker ID。随机/筛选平均MER为 **10.41% / 10.90%**；筛选−随机 **+0.4952个百分点**，配对seed/说话人区间 **[−0.1378,+1.1198]** 跨零，这套固定参考一致性筛选没有可靠优势。旧100条/2人单列次要分析（合计400条/5人，不是400条全新盲测）；出分前冻结209条audit pass敏感性结果亦为筛选均值较差（+0.4522pp）、区间跨零。[完整结果](reports/2026-10-20-multiseed/RESULTS.md) · [12案例分析](reports/2026-10-20-multiseed/REGRESSION_ANALYSIS.md) · [主协议](MULTISEED_PROTOCOL.md)。

复用原seed42 checkpoint，补43/44四组真实SFT，各63步并实际重载。两臂仍各500条固定集合，尚未衡量选数seed方差。旧中文/混合12退化仅2条在3seed均更差；原参考与分数保留，人工听音pending。

## 早期pilot

2026-10-20 已在 Apple M4 Pro / 24 GiB / CPU 4 线程完成真实 Qwen3-ASR-0.6B 开发实验。20 段唯一真人录音，中英混合 10、中文 5、英文 5；共 74.4204 秒。自动语言识别总 MER 为 13.99%，英文 5 段的 WER 为 48.48%。指定 English 的事后诊断为 18.18%，差异全部来自其中一段。两个条件各复跑一次，原文预测逐字一致。

这些是官方 train 分片中的开发样本，不是独立测试或训练效果。该首轮未训练；继续实验见下节。原生流式、TTS 首声或真人麦克风仍未测量。[完整结果与范围](reports/2026-10-20-mac-cpu/RESULTS.md) · [验证记录](reports/2026-10-20-mac-cpu/VALIDATION.md) · [统计 JSON](reports/2026-10-20-mac-cpu/SUMMARY.json) · [技术设计](../../docs/superpowers/specs/2026-10-20-duplexlab-design.md) · [执行计划](../../docs/superpowers/plans/2026-10-20-duplexlab-implementation.md)

## 继续实验

[预写协议](CONTINUATION_PROTOCOL.md)与[新说话人结果](reports/2026-10-20-continuation/RESULTS.md)：新 40 条英文 auto WER 20.92%，已知 English 20.50%，仅少 1 个编辑错误，未重现旧小样本的大变化。训练池 1,465 条、dev40/test100 按官方 speaker ID 隔离；已在 H100 GPU1 完成各 500 条 SFT、各 63 步、1epoch 与真实 checkpoint 重载。同一固定 test100 的基座/随机/筛选 MER 为 **15.31% / 14.26% / 14.16%**；筛选只比随机少 1 个错误，配对区间跨零，未获得筛选优于随机的明确证据。旧导入遗漏 speaker/session 字段的更正见结果报告。

## 当前命令

从仓库根目录使用轻量开发环境：

```bash
.venv-dev/bin/python -m speaker_lab.duplex --help
.venv-dev/bin/python -m speaker_lab.duplex score --run runs/duplexlab/qwen06-cpu-dev20-002
.venv-dev/bin/python -m speaker_lab.duplex report --run runs/duplexlab/qwen06-cpu-dev20-002 --output runs/duplexlab/qwen06-cpu-dev20-002/replay/index.html
.venv-dev/bin/python -m speaker_lab.duplex review --run runs/duplexlab/qwen06-cpu-dev20-002 --output runs/duplexlab/qwen06-cpu-dev20-002/replay
```

`freeze` 校验来源、元数据、哈希、路径、split 和脚本家族后创建不可覆盖的新协议。`audit` 独立运行原音检查，不加载模型。`run` 只使用冻结配置和固定本地模型快照；可显式设语言，默认基线用空字符串自动识别。

`--resume` 不重跑已成功项目；失败重试保留旧日志和源码快照，协议、模型或 ASR 源码改变时拒绝续跑。遗留 running 保守拒绝，不推断旧进程已退出；需核实后另建 run，尚未提供自动回收功能。

人工回听后可在 review.html 填写审阅者、实际转写与决定，下载 reviews.json，再运行：

```bash
.venv-dev/bin/python -m speaker_lab.duplex import-reviews --run runs/duplexlab/qwen06-cpu-dev20-002 --file /absolute/path/to/reviews.json
```

页面初始全部未审阅；界面/播放器测试不等于人工听音。导入验证协议、音频哈希与旧版本，不覆盖原参考；相同文件重复导入幂等。`report --public` 只输出统计，不带原音、原文、预测或绝对路径。

## 环境与数据

基础 CPU 逻辑沿用 Python 3.12 和原音频依赖。模型在 `.venvs/asr-mac/` 独立环境，当前记录为 qwen-asr 0.0.6、Transformers 4.57.6、Torch 2.8.0、FP32；该 Mac 组未使用 CUDA 或 MLX；继续实验另建 Linux/H100 BF16 环境，seed42/43/44、batch1/grad_acc8、lr2e-5、1epoch。环境均为运行快照，尚未证明任意干净主机可一次安装。

`data/duplexlab/`、`runs/duplexlab/` 与模型缓存均按既有 gitignore 留在本机。源码只携带允许公开的计数、来源 ID 和哈希，不包含第三方音频或完整转写。ASCEND 原转写保留 CC BY-SA 4.0，模型许可独立于原创代码许可。

## 训练与报告入口

`scripts/prepare_duplexlab_pilot.py` 准备固定分片，已有目录拒绝覆盖。`scripts/run_duplexlab_continuation.py` 提供 pool/train/evaluate 三阶段，先做8train/4dev smoke并真实重载，再从同基座训练随机/筛选各500条。独立 `scripts/duplexlab_train.py` 默认只验证，需 `--execute` 才启动。数据与JSONL导出哈希绑定，dev独立、test不进入训练导出。

`scripts/report_duplexlab_continuation.py` 从本地归档重建汇总图、训练曲线及三模型私有回放；筛选器支持按“编辑数减少/增加/相同”查看全部案例，不伪造人工审阅。[正式结果](reports/2026-10-20-continuation/RESULTS.md) · [统计与模型来源](reports/2026-10-20-continuation/SUMMARY.json)。

## 下一步

1. 人工回听分歧与退化，纠正另存版本，保留当前原分数。
2. 当前三seed和覆盖约束已完成；继续需新增可核验说话人/会话来源的冻结测试，记录已观察集合，不复用当前test调参。
3. 频次/跨度和删除保留九次新训已完成。先回听稳定重复/句尾反例，再在train/dev分开核对归一化与加权目标；当前280已观察，后续另冻测试。
4. 原生流式、合成表达、麦克风与Avatar分别验收，不以本次离线SFT代替。

新扩样入口：`prepare_duplexlab_multiseed.py`冻结speaker3/4/17；`run_duplexlab_multiseed.py --phase train|evaluate`通过哈希绑定的seed适配器补43/44。`verify_duplexlab_multiseed.py`无模型重推理复算，`report_duplexlab_multiseed.py`生成聚合图与私有回放。[复现和审计命令](reports/2026-10-20-multiseed/REPRODUCE.md)。


覆盖实验入口：`prepare_duplexlab_coverage.py`、`run_duplexlab_coverage.py --root . --model /path/to/pinned/snapshot`、`verify_duplexlab_coverage.py`、`report_duplexlab_coverage.py --root .`。[前置依赖与复现命令](reports/2026-10-20-coverage/REPRODUCE.md)。私有test/dev回放位于 `runs/duplexlab/coverage-v1/replay/`；人工听音仍pending。
