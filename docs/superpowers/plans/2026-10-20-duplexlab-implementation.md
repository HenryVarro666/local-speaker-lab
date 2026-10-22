# DuplexLab Implementation Plan

**归一化/权重最新进度（2026-10-22，已完成）：** 十二次基座新训完成：随机原CE / 筛选原CE / 同筛选全1归一化CE / 同筛选2倍词元加权CE，各500条、63步、seeds42/43/44。新test362条、10个当前拆分隔离speaker IDs，平均MER **9.7315% / 9.8353% / 9.8353% / 10.1329%**；三次总删除 **221 / 237 / 237 / 271**。归一化对照三seed对应权重逐字节相同、batch1输入1242对全同，最终转写也相同；保存tokenizer的左右padding侧效应另记，不外推到ragged批处理。加权−全1归一化 **+0.2976pp**，配对seed/整说话人区间 **[-0.0556,+0.6673]pp**；筛选−随机 **+0.1038pp**、区间 **[-0.3246,+0.5328]pp**。区间包含零，未证实总体加权收益。 train1015/dev52与上一轮数据及选数不变，全部12组dev先完成才读新test；10个IDs有历史项目曝光，不称全项目未见说话人，源参考不改、人工听音pending。 [结果](../../../experiments/duplexlab/reports/2026-10-21-normalization/RESULTS.md)。本轮12新训/362新测试已完成，不重复执行；后续先固定保存padding契约、核对参考/词语遗漏，再另冻数据，不能沿用已观察测试调参。

**删除保留最新进度（2026-10-20，已完成）：** 删除保留三臂消融已完成九次基座新训（随机原loss / 改进筛选原loss / 同筛选加权loss，各500条、63步、42/43/44）。新test280条/7个当前拆分隔离speaker IDs，平均MER **10.7691% / 10.3624% / 10.9632%**。联合候选−随机 **+0.1941pp**，配对seed/整说话人区间 **[-0.7538,+1.1337]pp**；三次总删除 **284 / 270 / 295**。删除和整体编辑必须一起判断，不能凭更少D直接称内容保留修复。train1015、dev提前统一40+12=52，所有九组dev重载后才读新测试；新录音排除此前ID/文件/PCM，7位有项目历史曝光，不能称全项目未见说话人。源参考和严格主分不改，人工听音pending。区间包含零，未建立可靠总体优势。 同数据加权目标三个seed总编辑均更多（平均MER+0.6009pp），不支持采用本轮加权目标；筛选原CE的填充词删除25→35，仍有退化。 [结果](../../../experiments/duplexlab/reports/2026-10-20-deletion/RESULTS.md)。本轮9次新训与280新测试不重复执行。

**频次/重复/跨度实验完成（2026-10-20）：** 频次/重复/切换跨度约束实验已完成六组基座新训（各500条/63步、42/43/44）。新test320条/4位当前拆分隔离说话人上，随机/新策略平均MER **12.8807% / 12.6199%**，差 **-0.2608pp**，配对seed/说话人区间 **[-0.9295,+0.2436]pp**；区间仍包含零，未建立可靠优势。填充词64条均值19.6048%→18.8248%。三次总删除296→318，填充词token删除26→25；未证实口语保留已修复。两臂既有覆盖和新增频次/重复总数/切换跨度边际一致，音频差0.9643%。原dev40保留；另冻12条补足重复等缺口，数据在新test前冻结、诊断在主流程后运行，不改训练eval_file。测试speaker13/14/18/24有历史确认/测试曝光，不称整个项目新说话人。 [最新结果](../../../experiments/duplexlab/reports/2026-10-20-disfluency/RESULTS.md)。已完成阶段不重复执行。

> **For agentic workers:** 使用 superpowers:executing-plans 按任务执行。本项目采用当前助手直接执行；用户未明确要求时不启动子代理。执行状态更新于 2026-10-20，未勾选项包含尚未逐项满足的验收要求。

**当前进度：** `feat/duplexlab` 已完成离线闭环、新40条英文语言确认、1465条训练池、匹配预算各500条SFT与同test100三臂评测；正式模型各63step/epoch1并真实重载。MER基座/随机/筛选为15.31%/14.26%/14.16%，区间跨零，未获得筛选优于随机的明确证据。人工听音、原生流式仍待做。任务7–9按[预写pilot协议](../../../experiments/duplexlab/CONTINUATION_PROTOCOL.md)执行；[首轮](../../../experiments/duplexlab/reports/2026-10-20-mac-cpu/RESULTS.md)与[继续实验](../../../experiments/duplexlab/reports/2026-10-20-continuation/RESULTS.md)分别保存。

**Goal:** 在既有 local-speaker-lab 中交付可回听、可审阅、可导出训练数据的语音实验，再完成流式识别和有对照的 ASR 微调。

**Architecture:** 保留 speaker_lab 包和原双人/词表工作流，新增 speaker_lab.duplex。CPU 数据与审阅层独立于模型进程，离线复用 WorkerRunner，流式另接 vLLM 常驻进程。模型和原始数据保留本地，公开结果只带有权分发的示例和汇总。

**Tech Stack:** Python 3.12、现有 NumPy/SciPy/SoundFile、SQLite、JSONL/TOML、静态 HTML；模型层为固定 Qwen3-ASR，合成为可选提供方适配。

**Spec:** [DuplexLab 技术设计](../specs/2026-10-20-duplexlab-design.md)。先读全部设计，再执行本计划。

## Global Constraints

- 主仓库为 HenryVarro666/local-speaker-lab；包名 speaker_lab 不变，新增目录 src/speaker_lab/duplex/。
- 当前本地路径 /path/to/projects/local-speaker-lab，已克隆；分支 feat/duplexlab，起点 6092ddeb9ba68852e1b3144e3e684825d165ae76。没有提交或推送。
- 本机开发使用 /opt/homebrew/bin/python3.12；不使用当前默认 Python 3.15.0rc3 建环境。
- 两份历史 ASCEND 冻结协议与运行不修改；725 条任务状态未核实，不自动另跑。
- CPU 测试、CLI 帮助和模块 import 不下载权重、不启 CUDA；GPU 环境独立且只用可用资源窗口。
- 不自动提交或推送；不影响 apex 的其他服务；API 默认关闭且费用预算为 0。
- 每个模型实验绑定输入、配置、源码、模型版本和运行状态；只完成 smoke 时如实标为 smoke。
- freeze/audit/run/score/review/import-reviews/report 已实现；任务 5–9 的其他 CLI 仍是未来接口，完成对应任务与测试之前不能当作已可用命令。

## Review Focus

1. 音频路径遍历、符号链接越界、哈希变化或重复 ID：在模型启动前拒绝，任务 1 覆盖。
2. 同源数据跨 split、合成标签冒充人工真值、非有限或完全静音音频：冻结/审计失败有明确原因，任务 1、2 覆盖。
3. 中断后读到旧 response、重复运行计费或重复成功结果：显式恢复、哈希绑定和不盲重试，任务 3、6 覆盖。
4. 超快喂完音频、partial 当 final、跨 clip 上下文残留：pacing、唯一 final 和新建 state，任务 5 覆盖。
5. 未回听即完成审阅、纠正覆盖历史、未完成运行被计分或 test 进入训练：状态与来源隔离，任务 4、7、8 覆盖。

## 任务 0 接手、迁移规划与原测试基线

**Files:** 建立已有仓库本地克隆；复制本设计与计划到同名 docs/superpowers 位置。新增 `experiments/duplexlab/README.md`；本地证据位于 `runs/duplexlab/bootstrap-<时间>/`。

**Interfaces:** 输入为已有远端；输出为确认的 checkout、源码 SHA、轻量开发环境与原测试报告。没有生产行为变更。

- [ ] 先确认目标路径不存在或核对已存在仓库的 remote、分支和未提交内容。需要隔离时按 using-git-worktrees 处理；不移动其他任务文件。
- [ ] 仅在路径确实不存在时执行：

```bash
git clone https://github.com/HenryVarro666/local-speaker-lab.git /path/to/projects/local-speaker-lab
cd /path/to/projects/local-speaker-lab
git status --short
git log -1 --format='%H %s'
git switch -c feat/duplexlab
```

- [ ] 检查 clone 中是否新增 AGENTS/CLAUDE 指令并遵守；确认 725 条任务的负责人和运行机器，本轮不启动/重启它。保存两份历史协议的 tracked 文件哈希及当前 diff。
- [ ] 迁入两份技术文档，之后只在新 checkout 更新本计划复选框；voice_ai 总路线指向该执行副本。
- [ ] 在仓库根目录建立轻量环境。当前以下工具版本取自已读仓库文件，执行时先核对是否已变：

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-dev
.venv-dev/bin/python -m pip install --upgrade --only-binary=:all: -r requirements/bootstrap.txt
.venv-dev/bin/python -m pip install --build-constraint requirements/bootstrap.txt -r requirements/dev.txt
.venv-dev/bin/python -m pip install --no-build-isolation --editable '.[dev]'
.venv-dev/bin/python -m pip check
.venv-dev/bin/python -m pytest -q -rs
.venv-dev/bin/python -m speaker_lab --help
.venv-dev/bin/python -m speaker_lab.worker --help
```

- [ ] 保存完整输出与退出码。通过数不预填；Mac 不支持的测试保留具体跳过/失败原因，先区分原失败再开发。基础层不安装 Gradio、Torch 或 vLLM。

**Exit:** 原仓库入口可用，基线结果已保存，已知任务不冲突。环境安装失败时先解决此门槛，不直接转 GPU 安装。没有 commit/push 步骤。

## 任务 1 manifest 与冻结协议

**Files:** Create `src/speaker_lab/duplex/{__init__.py,__main__.py,cli.py,protocol.py}`、`configs/duplexlab.example.toml`、`tests/test_duplexlab_protocol.py`；样例为自写 JSONL，不包含第三方音频。

**Interfaces:** `load_manifest(path: Path, data_root: Path) -> list[dict]`、`freeze_protocol(manifest: Path, data_root: Path, config: Path, output: Path) -> dict`、`verify_protocol(protocol_dir: Path, data_root: Path) -> list[dict]`；CLI `freeze --manifest --data-root --config --output`。

- [x] 先写拒绝重复 ID、错误哈希、路径越界、符号链接越界、同源跨 split 的测试；另测目录位置变化不影响内容哈希，fixture 与 synthetic 身份不丢失。
- [x] 运行 `.venv-dev/bin/python -m pytest tests/test_duplexlab_protocol.py -q`，确认缺少新模块时失败。
- [x] 实现设计 §4 数据字段与配置；冻结到不存在的新目录，不覆盖旧协议。完整验证后一次发布 FREEZE.json；不将部分输出当冻结成功。
- [x] 相同测试通过；`python -m speaker_lab.duplex --help` 与 `freeze --help` 不导入 Torch。使用合成生成的临时 WAV 验证正常冻结和篡改后的拒绝，不用真实模型。

**Exit:** 20 条开发 manifest 可冻结并复验；只有 schema 和哈希通过，不声称语音内容正确。

## 任务 2 音频审计与精确去重

**Files:** Create `duplex/audio.py`、`tests/test_duplexlab_audio.py`；Modify `duplex/cli.py`；输出到新 `runs/duplexlab/<run>/audit.jsonl`，规范 PCM 在本地 data/duplexlab/derived/。

**Interfaces:** `audit_clip(record: dict, data_root: Path, derived_root: Path, rules: dict) -> dict`、`find_exact_duplicates(records: list[dict]) -> list[dict]`；CLI `audit --protocol --data-root --output`，规则来自冻结 config。

- [ ] 测试损坏/零帧、NaN、完全静音、1 秒和 12 秒边界，以及有停顿但包含有效语音；原音哈希在操作前后相同。
- [ ] 测试两种无损封装解码到相同 PCM 可归为重复、幅度不同的音频不能被自动同一化；重复跨 split 必须报错而非静默丢掉 test。
- [ ] 测试先检查原音削波/静音特征，再生成规范 PCM；按设计初始阈值区分 pass、needs_review 与 reject。复用已有音频能力但不调用会提前拒绝静音的路径掩盖原因。
- [ ] 运行 `.venv-dev/bin/python -m pytest tests/test_duplexlab_audio.py -q` 完成失败→实现→通过；再用本人/有权使用的 20 条开发录音作真实审计，保留逐条原因。

**Exit:** 每条都有去向和原因；只实现文件与 PCM 精确去重，不写成近重复/说话人去重已完成。

## 任务 3 可恢复的离线 ASR 基线

**Files:** Create `duplex/store.py`、`duplex/runner.py`、`tests/test_duplexlab_store.py`、`tests/test_duplexlab_runner.py`；Modify `duplex/cli.py`；不改原 runner/worker 的公共协议。

**Interfaces:** `RunStore(db_path: Path)`，方法 `start_attempt(clip_id: str, condition: str) -> int`、`finish_attempt(clip_id: str, condition: str, attempt: int, result: dict) -> None`、`fail_attempt(..., error: str) -> None`、`iter_results() -> list[dict]`、`put_review(review: dict) -> None`；`run_offline(protocol_dir: Path, config: dict, output_dir: Path, resume: bool=False) -> dict`。省略号方法的完整参数与 finish_attempt 相同但以 error 替换 result。

- [ ] 测试成功任务不重复、失败显式恢复、配置改变拒绝恢复、旧进程仍在运行拒绝接手、遗留 running 不当成功、旧 response 不复用。SQL 唯一键与写结果/成功状态同事务。
- [ ] 用已有 fake backend 完成两条任务与一个故意失败任务，验证不完整 run 不能导出完整比较。测试中不启动模型。
- [ ] 新 loader 构造原 WorkerRunner 所需的 `_root`、`workers.transcribe_python`、timeout 和 `transcribe` 配置，调用其 transcribe 阶段。job 包含 id/audio_path/start/speaker；批量请求一次加载模型，完整检查返回 ID。
- [ ] 显式离线配置 `language=""`、`aligner=""`、`max_new_tokens=256`，使用 0.6B 模型的固定 revision；测试没有继承 Chinese 默认或隐式加载强制对齐模型。
- [ ] 新增 CLI `run --protocol --data-root --config --output [--resume]`；不经过原四阶段 config loader。
- [ ] 运行 `.venv-dev/bin/python -m pytest tests/test_duplexlab_store.py tests/test_duplexlab_runner.py -q`。随后在可用模型环境用 3 条非测试集录音真实 smoke，再完成 20 条开发基线。

**Exit:** 每条原始输出、模型版本、加载/推理时间和失败可查；在此阶段 mode 固定 offline，不报告实时延迟。

## 任务 4 计分、回听、纠正与第一版报告

**Files:** Create `duplex/metrics.py`、`duplex/review.py`、`duplex/report.py`、`tests/test_duplexlab_metrics.py`、`tests/test_duplexlab_review.py`、`tests/test_duplexlab_report.py`；Modify cli。

**Interfaces:** `score_run(run_dir: Path, protocol_dir: Path, data_root: Path) -> dict`、`build_review_packet(run_dir: Path, output_dir: Path) -> Path`、`import_reviews(run_dir: Path, review_file: Path) -> dict`、`render_report(run_dir: Path, output: Path, public: bool=False) -> Path`。

- [ ] 测试总体分数按编辑数和参考 token 聚合；中文、英文、mixed 分组独立；空参考与缺失预测单独报告；fixture 不进入质量表；不完整 run 拒绝正式计分。
- [ ] 新评分调用原 score_pair/aggregate_scores，另存 normalization_version，不改变历史 scorer。先用 3 对人工可算的例子验证，再计真实 20 条。
- [ ] 回放参考旧听音页，音频默认本地引用；标签和转写 HTML 转义。记录 listened/reviewer_id/answer_exposed，批准时校验当前音频/协议哈希。重复导入幂等，冲突保留两版本，不覆盖源 reference。
- [ ] 测试未听音不能标为完成人工、哈希改变拒绝导入、含 `<script>` 的预测只能作为文字显示、公开模式不泄漏绝对路径/第三方原文。
- [ ] CLI 提供 `score --run --protocol --data-root --output`、`review --run --output`、`import-reviews --run --file`、`report --run --output [--public]`。公开汇总仅统计与明确允许公开的示例。
- [ ] 运行三个新测试文件，并实际在浏览器回听至少 3 条；不能只看 HTML 源码。人工核对播放器、文字换行、修订记录和导出；生成页面不代替人工听音。

**Exit:** 可交付第一版离线 Demo：真实识别→错误回放→审阅→版本记录。若尚无用户/人工听音，页面状态保持 pending，不伪造审阅结果。

## 任务 5 原生流式适配与计时

**Files:** Create `duplex/streaming.py`、`tests/test_duplexlab_streaming.py`；Modify metrics/cli，配置加入 streaming 独立 Python 与模式。

**Interfaces:** `replay_stream(asr: object, samples: np.ndarray, settings: dict, clock: object, emit: Callable[[dict], None]) -> dict`；`run_streaming(protocol_dir: Path, config: dict, output_dir: Path) -> dict`；`score_stream_events(events: list[dict], speech_bounds: dict) -> dict`。clock 提供 `monotonic_ns()` 与 `sleep(seconds)`，真实与测试实现同接口。

- [ ] 测试每块等音频时轴到达才送入、partial 为空不记首次文本、finish 只执行一次、每 clip 新 state、取消后无旧 final、部分异常保留 events 但不记成功。
- [ ] 用假时钟和固定 partial 序列精确验证首次 partial、最后有声到 final、前缀修订数。分段方法与参数保存 speech_bounds.json；不能把文件尾静音也计为用户讲话。
- [ ] 对接已核对官方 API；初始 step=500ms、内部 chunk=2.0s、unfixed_chunk_num=2、unfixed_token_num=5。原示例未 pacing 的循环只作 API 参考。
- [ ] 模型进程常驻一批任务，stdout 日志与事件文件分离，事件绑定 attempt 与递增 seq。native streaming 只在独立 vLLM 环境验证；没有可用显存时不加载。
- [ ] 运行 `.venv-dev/bin/python -m pytest tests/test_duplexlab_streaming.py -q`，再在 Linux 以至少 6 条真人/授权开发录音测真实事件；核验新模型一次加载、clip 间状态清空、所有 final 完整。

**Exit:** 真实流式 smoke + 原始事件可复算。失败时保留离线 Demo，报告 stream_not_verified，不用离线结果充数，也不为抢工期改现有服务。

## 任务 6 可选表达数据与两日演示

**Files:** Create `duplex/synthesis.py`、`tests/test_duplexlab_synthesis.py`、`experiments/duplexlab/scripts.example.jsonl`；Modify cli/report。

**Interfaces:** `synthesize_scripts(scripts: Path, config: dict, output_dir: Path, dry_run: bool=True) -> dict`；CLI `synthesize --scripts --config --output [--execute]`。config 中 enabled=false 且 max_cost_usd=0 为默认。

- [ ] 先用本人录音或已存在且有使用权的音频完成同文本不同表达的 3 对样本，不等待 API。
- [ ] 可选适配已核对的 Drama 3 API；具体请求 schema、费用及密钥由实施当天官方文档与账户验证，不能照搬未核验字段。密钥只读环境变量，不输出请求 headers。
- [ ] 测试默认不发网络、缓存命中不再请求、预算不足前置拒绝、语音和脚本哈希记录、超时未知计费不自动重试；API 错误不伪造音频。
- [ ] 授权和预算可用时先合成 1 条，经人工确认实际说出的文字再扩到最多 20 脚本×3 表达；人工未回听的 style_verified=false。所有变体跟随 script_id 分组。
- [ ] 录制 90 秒展示：一个成功基线、一个停顿/改口失败、一个审阅与导出过程。用字幕明确真人/合成、离线/流式、历史/当前模型；不加入未经实现的训练前后改善。

**Exit:** 有声音与可追溯处理流程的展示。任务 6 是可选增强；无 API 不阻塞任务 7、8。

## 任务 7 冻结训练与独立测试数据

**Files:** Create `duplex/datasets.py`、`tests/test_duplexlab_datasets.py`、`experiments/duplexlab/protocols/curation-pilot-v1/`；Modify cli。

**Interfaces:** `select_training_arms(pool: list[dict], predictions: list[dict], reviews: list[dict], policy: dict) -> dict[str, list[dict]]`、`export_training(arms: dict, data_root: Path, output_dir: Path) -> dict`；CLI `export-train --protocol --selection --data-root --output`。

- [ ] 审计 train/dev/test 的来源和历史使用。目标 dev=40、test=200 条真人，旧 200/725 validation 不作新盲测；没有可验证分组时披露 speaker independence 未知。
- [ ] 从官方 train 冻结最多 2000 条候选的来源与选择规则，对候选做同一基座离线转写；不使用 test 误差构造选择规则。A1 随机与 A2 筛选各 500 条；不足则另立较小 pilot，保持两臂同量。
- [ ] A2 明确采用 ASR 参考一致性排序：先排除 reject 和未裁定 needs_review，按训练参考与基座预测的 MER 递增，再按 silence_fraction、clip_id 排序；不是通用最优 curation，也不把模型一致当真值。A1 从同一有效池固定 seed=20261020 抽样。
- [ ] 两臂语言×时长桶配额相同，数量相同、总时长差≤1%；language 配额默认 mixed 250、zh 125、en 125。时长桶为 [1,3)、[3,6)、[6,12] 秒，桶配额在开发池统计后、正式挑选前冻结；无法匹配时报告不可行，不隐式改配额。
- [ ] 测试跨 split 的源家族/PCM 重复拒绝、未裁定不导出、fixture/synthetic 不进真人首轮、审阅纠正优先但源 reference 保留、预算不匹配拒绝、固定输入与种子导出一致。
- [ ] 输出各 arm JSONL（audio/text）、selection.json、来源与内容哈希、counts.json；dev 单独导出，test 不写到训练目录。text 前缀固定 zh→`language Chinese<asr_text>`、en→`language English<asr_text>`、mixed→`language None<asr_text>`；测试这三种映射，不构造 Mixed 语言标签。audio 在训练机器解析为绝对路径，另保留可移植来源哈希。
- [ ] 运行 `.venv-dev/bin/python -m pytest tests/test_duplexlab_datasets.py -q`，独立复算样本数、时长和交集，人工抽查若干已纠正条目。

**Exit:** 形成可训练、可追溯、预算匹配的三臂协议。ASR 一致性筛选偏向当前基座易识别样本的局限预先记录，不等结果后才解释。

## 任务 8 SFT smoke、训练与 checkpoint 验证

**Files:** Create `scripts/duplexlab_train.py`、`tests/test_duplexlab_training.py`、固定训练 TOML；上游代码放 `.cache/upstream/Qwen3-ASR/`，模型在 checkpoints/duplexlab/，不纳入 Git。

**Interfaces:** `build_training_command(config: dict, export_dir: Path) -> list[str]`；包装脚本接受 `--config --export-dir --output --dry-run`，执行前核对导出哈希、dev 路径和输出新目录。

- [x] 固定上游 commit 与脚本 SHA，依照官方格式准备离线依赖环境。复用其数据 collator、prefix loss 与可推理 checkpoint 回调；不在原离线或流式环境原地升级 Transformers。
- [x] 测试误把 test 作为 eval_file 时拒绝、没有 dev 时拒绝、同一路径覆盖时拒绝、导出哈希变化时拒绝、dry-run 不启动训练。只传实际存在的 CLI 参数。
- [x] 先使用 8 train/4 dev 独立 smoke。初始 batch=1、grad_acc=8、lr=2e-5、epochs=1、num_workers=0、persistent_workers=0、save_steps=1、save_total_limit=2；这些小数据不用于结果表。
- [x] 检查 loss 有限、至少一个 optimizer step、checkpoint 和处理器文件齐全，再实际从 checkpoint 加载并转写一条 smoke 音频。保存 GPU 峰值、耗时、版本；只有训练退出 0 不算重载成功。
- [x] 两臂正式参数固定为 0.6B、batch=1、grad_acc=8、lr=2e-5、epochs=1、num_workers=0、persistent_workers=0、save_steps=50、save_total_limit=2，显式传同一 dev。seed=42 在 Trainer 配置和记录中核对；当前上游没有 --seed CLI，不杜撰开关。
- [ ] OOM 或处理器导出失败就保存失败；若改为 LoRA、梯度检查点或更短音频，另立配置与适配任务，两臂一致，不只让某一臂采用有利配置。
- [x] 两臂各自训练、保存和重载成功后，冻结模型 SHA 与最终选用 checkpoint 规则。首轮只用一个 seed，报告 pilot。

**Exit:** 两个真实、可重载的训练 checkpoint；未完成时保持 training_incomplete，不写提升或生产就绪。

## 任务 9 冻结测试、流式回归和结果包

**Files:** `experiments/duplexlab/reports/curation-pilot-v1/RESULTS.md`、`summary.json`、允许公开的统计图；更新 experiments/duplexlab/README、docs/ROADMAP、必要的顶层 README 与第三方说明。

**Interfaces:** 复用任务 3/4/5，对 A0/A1/A2 同一 test、同一解码参数运行；新增 `compare_runs(run_dirs: list[Path], protocol_dir: Path) -> dict` 在 metrics.py。

- [ ] 测试三臂 ID/参考/模型映射完整、缺项拒绝、总体 MER 用总计数计算；质量差值采用真实配对，group 信息缺失时不冒充说话人聚类。
- [ ] 三臂完整运行冻结真人 test。保存原始预测、编辑数、语言/表达切片及 1000 次 seed=20261020 区间；不重复改变 test 或按效果提前停跑。
- [ ] 以 6–20 条固定开发流式样本重载并比较新 checkpoint；资源足够时再完成 ≥100 条冻结延迟集。离线变好而流式退化需并列报告，不隐藏。
- [ ] 给出工程完成状态与研究结论两个独立字段。差值区间跨零写“未获得明确改善证据”；合成测试结果与真人结果分开。初次 partial 不称首声，clip bootstrap 不称 speaker-disjoint。
- [ ] 全量 CPU 测试、原 CLI 及冻结文件检查通过；不为凑全绿删除平台失败。人工查看公开报告和至少 3 个音频案例，核对图表确实来自本次 JSON。
- [ ] 保存每个产物的路径与哈希，写第二版 90 秒演示和一条复现入口；最后列出本地修改/模型运行/部署/提交/推送各自状态。提交和推送等待用户明确指令，不自动执行。

**Exit:** 数据→训练→评测有完整证据，无需正提升也能完成实验。之后再选一种扩展：合成数据训练、原生全双工、Avatar 或多种子复验。

## 执行状态与自查

2026-10-20 实际执行（环境核验同日）：

| 任务 | 当前状态与证据 | 仍未满足的要求 |
| --- | --- | --- |
| 0 | 克隆、分支、文档迁入、隔离环境和原测试基线完成；原 worker 的虚拟环境 Python 解引用问题已复现并修正 | 725 条历史任务只确认在其他机器；负责人和实时完成状态未知，未访问或重启 |
| 1 | 已验收：schema、路径、哈希、来源/split、合成标识、不可覆盖冻结、篡改拒绝与轻量帮助测试通过；真实 20 条冻结并复验 | 无新增功能待办；不等于内容标注已核对 |
| 2 | 原音审计、16k 单声道派生、文件/PCM 精确去重及跨 split 拒绝已实现；真实 20 条全部 pass、零精确重复 | 损坏/零帧、恰好 1/12 秒、有停顿边界和幅度不同的专项测试尚未逐项覆盖；不宣称近重复或说话人去重 |
| 3 | 实际 worker、事务成功、完成后不重复、失败重试保留旧日志、协议/模型/源码校验已实现；真实 20 条及 5 条诊断各复跑，预测分别 20/20、5/5 一致 | 未另跑计划中的 3 条独立 smoke；running 一律保守拒绝，无自动回收；只记录整批子进程耗时，逐句加载/推理时间未提供 |
| 4 | 真实计分、私有回放、待审阅页、哈希绑定的幂等审阅导入和脱敏汇总已实现；浏览器验证播放器 metadata、导出约束、移动版与无 JS 错误 | **没有实际人工听音**，全部待审阅；人工纠正演示尚未验收，不能将测试构造的审阅当真人记录 |
| 5–6 | 原生流式、表达配对未实施 | 流式真实事件、合成表达和90秒演示仍待验收 |
| 7 | 已完成pilot数据：官方speaker ID隔离、新语言40/dev40/test100/训练池1465；随机/筛选各500，语言×时长配额一致、时长差0.4547%，导出哈希和角色验证通过 | 目标test200在本轮预写协议明确为100条pilot；训练执行入口为阶段脚本，原设计export-train子命令未接；实际人工纠正尚未使用 |
| 8 | 已完成：上游7c6daf77…固定脚本、独立H100环境、8/4 smoke1step并真实重载，两组500/40各63step/epoch1与真实100条转写；实际参数seed42/BF16/batch1/grad_acc8/lr2e-5核对通过 | 仅一个训练seed；显存为2秒采样峰值9623MiB，非完整allocator峰值 |
| 9 | 已完成离线pilot结果包：同test100三臂、语言切片、配对1000次clip bootstrap、三模型回放、公开统计/图和Mac独立复算一致 | 原生流式回归、更多说话人/seed、实际人工听音与90秒录像未验收；不将当前区间称说话人聚类或筛选有效证明 |

完整测试、依赖和页面检查见[首轮验证](../../../experiments/duplexlab/reports/2026-10-20-mac-cpu/VALIDATION.md)及[继续实验验证](../../../experiments/duplexlab/reports/2026-10-20-continuation/VALIDATION.md)。用户后续明确授权本轮筛选与ASR训练，已执行上述pilot；与原生全双工Agent、Avatar并行，使用物理GPU1保留GPU0既有任务，没有访问apex、提交或推送。原native streaming与原生语音后训练不由此次ASR SFT自动视作完成。

设计覆盖：路径与冻结→任务 1；音频与去重→2；恢复与模型隔离→3；审阅/计分/导出隐私→4；原生流式与同钟计时→5；表达素材与预算→6；数据切分和对照→7；训练/重载→8；结果和原功能回归→9。沿用用户默认的直接执行方式，无子代理或自动 commit 步骤。

## 2026-10-20 频次/重复/跨度继续实验（执行中）

按用户继续实验：先完成旧1800条对齐/8退化核对，再冻结[主协议](../../../experiments/duplexlab/DISFLUENCY_PROTOCOL.md)、单候选选数和新test320/4人。两臂500条×3seed从基座新训，各63步；不得根据dev/test挑最好seed或改策略。补充[dev覆盖协议](../../../experiments/duplexlab/DISFLUENCY_DEV_ADDENDUM.md)：原dev40缺少重复2+，另冻12条纯文本挑战，新test尚未推理时绑定哈希，主流水线后诊断，不改原训练eval_file。验收为实际六训/21ASR批、严格源/权重/计分复算、频次/重复/跨度/删除分组与所有负结果、独立性限制、双语文档与共享状态同步；人工听音不冒称完成。
