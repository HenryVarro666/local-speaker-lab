# DuplexLab 语音数据与流式识别设计

**归一化/权重最新进度（2026-10-22，已完成）：** 十二次基座新训完成：随机原CE / 筛选原CE / 同筛选全1归一化CE / 同筛选2倍词元加权CE，各500条、63步、seeds42/43/44。新test362条、10个当前拆分隔离speaker IDs，平均MER **9.7315% / 9.8353% / 9.8353% / 10.1329%**；三次总删除 **221 / 237 / 237 / 271**。归一化对照三seed对应权重逐字节相同、batch1输入1242对全同，最终转写也相同；保存tokenizer的左右padding侧效应另记，不外推到ragged批处理。加权−全1归一化 **+0.2976pp**，配对seed/整说话人区间 **[-0.0556,+0.6673]pp**；筛选−随机 **+0.1038pp**、区间 **[-0.3246,+0.5328]pp**。区间包含零，未证实总体加权收益。 train1015/dev52与上一轮数据及选数不变，全部12组dev先完成才读新test；10个IDs有历史项目曝光，不称全项目未见说话人，源参考不改、人工听音pending。 [结果](../../../experiments/duplexlab/reports/2026-10-21-normalization/RESULTS.md)。本轮12新训/362新测试已完成，不重复执行；后续先固定保存padding契约、核对参考/词语遗漏，再另冻数据，不能沿用已观察测试调参。

**删除保留最新进度（2026-10-20，已完成）：** 删除保留三臂消融已完成九次基座新训（随机原loss / 改进筛选原loss / 同筛选加权loss，各500条、63步、42/43/44）。新test280条/7个当前拆分隔离speaker IDs，平均MER **10.7691% / 10.3624% / 10.9632%**。联合候选−随机 **+0.1941pp**，配对seed/整说话人区间 **[-0.7538,+1.1337]pp**；三次总删除 **284 / 270 / 295**。删除和整体编辑必须一起判断，不能凭更少D直接称内容保留修复。train1015、dev提前统一40+12=52，所有九组dev重载后才读新测试；新录音排除此前ID/文件/PCM，7位有项目历史曝光，不能称全项目未见说话人。源参考和严格主分不改，人工听音pending。区间包含零，未建立可靠总体优势。 同数据加权目标三个seed总编辑均更多（平均MER+0.6009pp），不支持采用本轮加权目标；筛选原CE的填充词删除25→35，仍有退化。 [结果](../../../experiments/duplexlab/reports/2026-10-20-deletion/RESULTS.md)。本轮9次新训与280新测试不重复执行。

**频次/重复/跨度实验完成（2026-10-20）：** 频次/重复/切换跨度约束实验已完成六组基座新训（各500条/63步、42/43/44）。新test320条/4位当前拆分隔离说话人上，随机/新策略平均MER **12.8807% / 12.6199%**，差 **-0.2608pp**，配对seed/说话人区间 **[-0.9295,+0.2436]pp**；区间仍包含零，未建立可靠优势。填充词64条均值19.6048%→18.8248%。三次总删除296→318，填充词token删除26→25；未证实口语保留已修复。两臂既有覆盖和新增频次/重复总数/切换跨度边际一致，音频差0.9643%。原dev40保留；另冻12条补足重复等缺口，数据在新test前冻结、诊断在主流程后运行，不改训练eval_file。测试speaker13/14/18/24有历史确认/测试曝光，不称整个项目新说话人。 [最新结果](../../../experiments/duplexlab/reports/2026-10-20-disfluency/RESULTS.md)。已完成阶段不重复执行。

日期：2026-10-20。状态：离线闭环、训练导出、两组各500条SFT及100条独立speaker ID测试已完成pilot；人工听音与原生流式尚未验收。代码宿主为现有 `local-speaker-lab`，Python 包名保持 `speaker_lab`。[执行状态](../plans/2026-10-20-duplexlab-implementation.md#执行状态与自查)、[首轮开发结果](../../../experiments/duplexlab/reports/2026-10-20-mac-cpu/RESULTS.md)与[筛选/SFT结果](../../../experiments/duplexlab/reports/2026-10-20-continuation/RESULTS.md)记录已实现范围，以下其余部分仍为设计。

## 1 目标与首版范围

构建可复现的语音实验闭环：音频来源 → 冻结样本 → 质量检查和去重 → ASR 原始结果 → 回听纠正 → 版本化训练数据 → 随机/筛选对照 → 真人测试与流式回归。

首版的体验是点开一段语音，听到停顿或改口，看到识别结果与修订记录，将问题加入复核并导出训练样本。首版回放先支持离线结果，流式功能必须有真正分块推理的事件证据。产品展示名称中的 DuplexLab 不表示已经训练原生全双工模型。

一期不建设通用标注 SaaS 或分布式调度系统。数据处理为单机、单任务协调；模型在独立进程/环境运行，界面不加载 GPU 模型。

## 2 当前源码与受保护的实验

核对快照：2026-10-20，`HenryVarro666/local-speaker-lab` main `6092ddeb9ba68852e1b3144e3e684825d165ae76`，项目 0.1.4，Python ≥3.11。执行时以新的实际 SHA 为准，记录漂移再修改。

| 已有代码 | 复用方式 | 保持的边界 |
| --- | --- | --- |
| `audio.py` | 参考 FFmpeg 解码和 16k mono 检查 | 它会拒绝静音并限制时长；新质量审计要先留拒绝原因，不直接用它替代原音检查 |
| `backends/qwen.py` | 离线转写入口 `Transcriber.run(payload, options)` | 完整片段 Transformers 推理，不冒充流式 |
| `runner.py`、`worker.py` | 复用 `WorkerRunner.run("transcribe", payload)` 和独立 Python 环境 | 原协议只有最终 JSON；流式事件另建适配，不把模型每块重载 |
| `config.py` | 延续 TOML 和显式 root 的约定 | 原 loader 强制要求四个双人阶段；新配置单独读取，不修改旧校验 |
| `glossary_metrics.py` | 调用 `score_pair(reference, hypothesis, terms=[])` 与计数聚合 | 保持旧主评分不变；新指标独立命名 |
| `make_glossary_review_packet.py` | 参考本地回听、答案暴露和哈希验证 | 它绑定历史三条件任务；新页面不修改旧脚本或历史导出 |

`experiments/glossary/protocols/ascend-val200-train-vocab30-v2/` 与 `ascend-val-remaining725-train-vocab30-v1/` 均保持原样。725 条任务的当前运行状态必须在操作其机器前核实；远端执行记录不能证明现在已完成。旧导入遗漏了官方original_speaker_id/session_id；后续已更正并按官方speaker ID隔离。旧冻结保持原样；仅无字节重复不能升级为说话人独立。

## 3 目录与环境

```text
src/speaker_lab/duplex/
  __init__.py   __main__.py   cli.py
  protocol.py       # manifest、冻结、配置与 schema
  audio.py          # 原音检查、解码、精确去重
  store.py          # SQLite 任务、结果与审阅状态
  runner.py         # 离线 ASR 编排，调用原 WorkerRunner
  metrics.py        # 新实验评分及流式事件计分
  review.py         # 本地回听与审阅 JSON 导入
  report.py         # 私人音频回放与可公开的汇总导出
  datasets.py       # 选择对照样本与导出训练 JSONL
  streaming.py      # vLLM 常驻进程与分块事件，第二阶段
  synthesis.py      # 可选的受预算约束合成，第二阶段
scripts/duplexlab_train.py  # 受控调用固定上游 SFT，不重写 trainer
configs/duplexlab.example.toml
experiments/duplexlab/{README.md,protocols/,reports/}
tests/test_duplexlab_*.py
```

只随任务创建真正使用的文件，不预先生成全部空模块。基础层使用现有 Python 3.12、NumPy/SciPy/SoundFile 和标准库；无需新增 FastAPI、Gradio 或数据库服务。先输出可本地打开的 HTML。SQLite 使用标准库。

开发环境、离线 Transformers ASR、流式 vLLM、训练各自独立。默认不装模型依赖；任何模块 import 与 CLI `--help` 不下载模型、不启动 CUDA。Mac 上 CPU 功能可做完；CUDA、vLLM 和训练只在有资源的 Linux 环境验证。没有资源窗口时不尝试同卡挤占现有服务。

## 4 数据与文件约定

规范格式为 UTF-8 JSONL，所有 schema 都带整数 `schema_version=1`。本地数据不自动进入源码发布包。

| 记录 | 必填字段与含义 |
| --- | --- |
| ClipRecord | `clip_id`、`audio_relpath`、`audio_sha256`、`source_id`、`source_group_id`、`split`、`language_group`、`reference`、`origin`、`license_id`、`label_status` |
| 音频元数据 | `sample_rate`、`channels`、`frames`、`duration_seconds`；审计实读并验证，不能只信输入字段 |
| 合成附加字段 | `script_id`、`voice_id`、`style_requested`、`generator_model`、`generator_revision`；表达未回听确认则 `style_verified=false` |
| AuditRecord | `clip_id`、原音哈希、标准化 PCM 哈希、`decision=pass/needs_review/reject`、`reasons[]`、实际测量字段与规则哈希 |
| PredictionRecord | `run_id`、`clip_id`、`condition`、`attempt`、`hypothesis`、`mode`、模型/配置哈希、加载与推理时间；没有 final 不写成功结果 |
| ReviewRecord | 数据与音频哈希、`clip_id`、`reviewer_id`、`listened`、`decision=approve/reject/defer`、`corrected_text`、`answer_exposed`、时间与旧记录 ID |

`origin` 仅取 `real/synthetic/fixture`；`split` 仅取 `train/dev/test`；`language_group` 为 `zh/en/mixed`；`label_status` 为 `source_reference/needs_review/human_approved`。fixture 只用于工程测试，不进入模型质量成绩。真人语料的公开原参考标为 source_reference，不伪称本项目人工复核过。

音频路径相对于显式 `data_root` 解析；拒绝 `..`、解析后越界的符号链接、ID 重复、未知 split、非有限数字、丢失许可来源和哈希不符。原始文字与音频不原地修改；纠正写独立 ReviewRecord，导出时选择有效最新版本。

冻结目录保存 manifest、config、分析计划和 `FREEZE.json`。冻结哈希只覆盖内容，不包含当前时间；同一内容在另一目录仍可验证。运行快照另存实际时间、机器与路径。每次运行新建 run 目录。

## 5 音频检查与去重

先检查原音，再转成 16kHz、mono、float32 规范 PCM；保留原音哈希、解码器版本及幅度信息。源文件不会被覆盖；不做自动响度归一化，避免抹掉削波和静音证据。

初始工程规则：解码失败、零帧、NaN/Inf、文件缺失、完全静音为 reject；小样本协议限定 1–12 秒，超范围单独记录，不自动截断。原音削波样本比例 >1%、RMS <-50 dBFS 或 20ms 窗口中 RMS <-50 dBFS 的比例 ≥80% 标为 needs_review。这些是待开发集审阅的起始规则，不是已验证的最优阈值；正式训练前冻结规则版本。

精确重复先检查文件字节，再检查规范化 float32 little-endian PCM 字节。不同编码的同一解码结果可以识别，但不宣称完成语义近重复检测。近重复 embedding 和跨说话人重复是后续能力。

已知同源切片、同一脚本的所有合成变体，按 `source_group_id` 共同切分；精确重复跨 train/test 直接拒绝冻结。缺少原始录音或说话人身份时标记 `split_unit=clip_only`，不宣称 speaker-disjoint。

## 6 运行状态与恢复

一个 run 保存 `run.json`、`state.sqlite`、`predictions.jsonl`、每个任务的事件文件、日志与摘要。协调器单进程；不引入通用分布式队列。

任务唯一键为 `(run_id, clip_id, condition)`；每次尝试有递增 `attempt`。状态为 `pending → running → succeeded/failed`，未处理项目可显式 skipped 并记录原因。运行进程异常结束时保留 running，不自动当成功。

`--resume` 只处理 failed/pending 和已确认旧进程退出的遗留 running；succeeded 不重做。恢复前核对协议、模型和配置哈希；变化则另建 run。子进程退出但没有有效最终结果视为失败，不读取上次的旧 response。模型超时、用户取消后等子进程退出再释放任务状态。

预测写入 SQLite 后提交成功状态，JSONL 从已提交成功结果生成；不靠 JSONL 行数猜任务完成。初版完整率要求所有预定 clip×condition 成功；有缺失就只出工程诊断，正式比较不得选择性忽略失败。

## 7 离线与流式 ASR

离线首选通过原 `WorkerRunner` 调用 qwen 后端，构造仅包含 transcribe 的适配配置，绕过要求四阶段的旧 `load_config`。一次批请求包含多个 job，模型加载一次，job 顺序和 ID 可验证；不每条音频启动一次模型。先使用 Qwen3-ASR-0.6B；需要与旧 1.7B 对照时另设 condition，不能混并。

离线选项显式设置 `language=""`（旧后端会转为 None，自动识别语言）、`aligner=""`（禁用默认强制对齐模型）、`max_new_tokens=256`，避免继承原后端默认的 Chinese 与额外 aligner。流式初始 max_new_tokens=32；两种模式的参数分别保存，不混称同一解码条件。新基线不能直接与旧 Chinese/1.7B CPU 实验的数值作归因比较。

原生流式依照官方 `Qwen3ASRModel.LLM`、`init_streaming_state`、`streaming_transcribe`、`finish_streaming_transcribe`。独立进程每次加载一个模型，逐 clip 新建 state；不得跨 clip 残留上下文。起始配置 `step_ms=500`、`chunk_size_sec=2.0`、`unfixed_chunk_num=2`、`unfixed_token_num=5`，并明确 500ms 输入步长不等于 500ms 首次出字。[官方示例](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py)

官方示例将块连续快速送入，**新实验要增加单调时钟 pacing**：第 n 块只有在对应音频已经“播放到”时才能送入。否则测得的是离线分块速度，不是实时输入延迟。vLLM 显存预留必须依据专用资源窗口配置，不能照搬示例的 0.8 抢占共享卡。

事件 schema 包含 `run_id/clip_id/attempt/seq/type/audio_cursor_samples/t_rel_ns/text`。类型为 `audio.started/audio.chunk/asr.partial/asr.final/run.error`，时间全部来自同一个 worker 的 monotonic 时钟；UTC 只用于日志排序。partial 是累计全文快照；final 由 finish 后写入，每 clip 最多一次。流式不使用未来音频，也不伪造在线 forced-alignment 时间戳。

## 8 评分与时钟

| 结果 | 固定定义 |
| --- | --- |
| mixed MER | 复用旧规范化和编辑计数，按总编辑数/总参考 token 聚合，不平均每句错误率 |
| 中文 CER 与英文 WER | 单语分组按对应 token 口径计算；混合 MER 不称为官方 ASCEND recipe |
| 改口和数字 | 开发时冻结脚本的词项/规范化规则；ASR 原文与可选任务状态解析分开评分 |
| 离线 RTF | 模型推理 wall time/音频时长；模型加载另列，不当作首声 |
| 首次 partial 延迟 | 输入第一个有声采样进入 pacing 时轴到首次非空 partial；不代表稳定或正确 |
| final 等待 | 输入最后有声采样到客户端可读取 final 的时刻；同 worker 测量，不减不同机器时钟 |
| 修订 | 相邻 partial 的最长共同前缀之后被删除/替换的 token 数与更新次数 |
| 加载/冷启动 | 分别记录进程/模型加载。只有明确卸载、缓存条件和重载步骤才称严格冷启动 |

有声边界来源为冻结的本地分段方法及参数，或人工确认的样本级时间；保留算法和边界文件。输入文件长度不直接当作有声结束时刻。

单条或 20 条报告逐次值、中位数和范围，p95 如给出必须注明样本数且只作描述。正式延迟集先冻结至少 100 条后才给探索性 p95；仍不称为稳定 SLA。质量比较保存逐条配对差异；可确认来源分组时做 group bootstrap，否则报告语言分层 clip bootstrap并说明说话人相关未知。默认 1000 次、seed 20261020。

后续接 voice_ai 的 TTS 首块、完整说完到首声、插话到停播各有独立起止，不能从 ASR 事件推导。跨浏览器和服务器计时须另做时钟对齐或同端测量；首版不报告这些指标。

## 9 人工审阅与回放

首版生成本地 HTML，音频使用明确授权的本地路径，默认不复制或上传。浏览器限制 file URL 时，提供音频路径和 JSON 导出；后续可在新仓库内提供局域本机只读服务，不改 voice_ai 服务。

审阅页面先隐藏参考和预测，播放后可填写听到的文本，再展开对照；一旦打开答案面板记录 `answer_exposed=true`。从结果页进入者已见预测，不能声称盲审。只有 listened=true、reviewer_id 非空且决策明确才算完成；自动生成页面不算人工完成。

导入审阅时验证 schema、音频/协议哈希、clip ID 和旧版本；重复文件幂等，冲突需要显式选择新版本。历史参考保持原样。训练导出可使用未被标记的公开原参考；所有 needs_review 项必须 human_approved 或排除。

私人回放可包含原文与绝对路径；公开汇总默认去除这些内容，只发布许可允许的自写示例、统计、配置和源码哈希。HTML 中任何转写或备注都按文本转义，不能注入脚本。

## 10 数据选择与训练对照

开发集：20 个短样本用于跑通和审阅起始规则。正式真人 dev 目标 40 条、test 目标 200 条，来源和是否曾用于调参先审计；数量不足时创建小型 pilot，不将开发集冒充新测试集。test 在规则、模型和训练参数冻结后才运行比较。

首轮训练只用公开真人 train 数据。A0 为未微调基座；A1 为随机选择；A2 为筛选选择。A1/A2 各 500 条，同一模型、batch、梯度累积、epochs、学习率、seed 和解码条件。两组语言×时长桶数量相同，实际总音频时长差≤1%；未满足时先重做选择，不宣称等预算。允许同一训练条目在 A1/A2 同时被选中，报告重叠；都不得与 dev/test 同源或精确重复。

A1 从通过基本有效性、去重且无未裁定 needs_review 的池，按 seed=20261020 抽样。A2 在同池采用明确的 ASR 参考一致性筛选：先用同一未微调基座转写候选 train，按参考与预测的 MER 递增、silence_fraction 递增、clip_id 字典序排序，在相同配额内取样；不能查看 test 误差决定选择。筛选偏向当前基座易识别样本，这不等于标签真值更准，也不是已证明的通用最优算法。第二版再研究失败类型/多样性选择，不在首轮临时混入多个算法。

候选池上限 2000 条，源自官方 train，先冻结选择规则；两臂语言配额为 mixed 250、zh 125、en 125，时长桶为 [1,3)、[3,6)、[6,12] 秒。桶配额在开发池统计后、正式挑选前冻结，达不到两组时长差≤1%时标记不可行，另立 pilot，不悄悄改规则。

使用固定上游 Qwen3-ASR SFT 脚本与可重载 checkpoint 回调；默认起始参数为 0.6B、batch=1、grad_acc=8、lr=2e-5、epochs=1、seed=42（当前 Trainer 默认种子，包装入口明确记录）。先用 8 条训练/4 条 dev 的独立 smoke 验证能训练、保存并重载。正式 eval_file 只能指向 dev，不能传 test。[官方 SFT 说明](https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md)

训练 JSONL 的 text 前缀固定为：zh 使用 `language Chinese<asr_text>`，en 使用 `language English<asr_text>`，mixed 使用 `language None<asr_text>` 后接实际转写；不虚构名为 Mixed 的语言标签。mixed 不以语言识别准确率为训练或评测目标。audio 为训练机器本地解析出的绝对路径，换机器重新导出，同时保留可移植来源与音频哈希。

当前官方 CLI 没有 max_steps、LoRA、gradient_checkpointing 参数，不把这些开关写成可直接执行。若需要支持，须单独版本化适配并对两臂一致应用。SFT 脚本默认不在本项目源码目录下载；放进 `.cache/upstream/Qwen3-ASR` 并固定 commit、保存文件哈希。

一个种子是 pilot 结果，后续可补 3 个训练种子；统计区间与跨种子方差分开。离线成绩改善后也需对新 checkpoint 做流式回归，不能推定两条路径相同。只有完整结果后才得结论；训练无提升也完成实验，不据此更换 test。

## 11 合成、原生全双工与 Avatar

合成不是第一轮训练前提。可选 Drama 3 离线生成 20 个自写脚本×3 种表达，先 1 条试听再决定是否扩展；同脚本所有变体同 split。标签表示 requested style，人工确认表达和实际说出的文字后才用作受控测试。默认 API 禁用、费用预算为 0；已有明确预算后才能启用，记录请求数、字符、耗时、失败与缓存哈希，不记录凭证。

网络超时且计费结果未知时不盲重试；若提供方没有已核验的幂等机制，记录 uncertain，避免重复扣费。合成故障不阻塞真人路线。

后续 PersonaPlex 只作为另一 Agent 对照，另写语言与模型资源协议；Avatar 先作独立回放层。两者不能使“合成混音测试”自动变成真人麦克风或 AEC 验证。原生模型、真人耳机与外放各需单独验收。

## 12 验收与证据

M1/M2：20 条来源可查、审计与哈希正确；真实 ASR 输出；审阅、纠错、导出完整；所有失败留痕。工程测试可以用 fixture，质量表禁止含 fixture。

M3：至少 6 条真实流式 clip，全部调用 finish，事件顺序和 pacing 可核验，无状态串入；记录真实首次 partial 与 final，不承诺特定延迟。条件不满足就维持“离线回放”标签。

M4/M5：两组训练样本与预算匹配，独立 dev/test，checkpoint 重载，三臂完整质量比较、至少小型流式回归、原始结果与参数可追。若只完成 smoke，状态保持 smoke，不能写训练改善。

所有检查分为 CPU 合约、真实模型、真人质量和在线集成四层；一层通过不代表后续层通过。原仓库测试、词表入口和冻结文件必须保留可用。默认不启动子代理、不提交或推送、不重启其他服务；执行时遵守宿主仓库新增指令。
