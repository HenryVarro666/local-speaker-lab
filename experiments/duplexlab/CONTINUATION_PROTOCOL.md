# 新说话人语言对照与数据筛选 SFT pilot

预写日期：2026-10-20。此规则在新样本 ASR 输出和训练前确定，延续既有任务 7–9，暂不执行流式/合成分支。固定首个 ASCEND train Parquet 分片及原模型 revision，不使用旧 200/725 validation。

发现官方 Parquet 实际提供 original_speaker_id、session_id；旧导入遗漏它们，旧报告“来源无该字段”的说法须更正。旧冻结文件保持原样，新增映射补记。以下 speaker 指官方 original_speaker_id，跨 session 作为同一人；不据 session 数字声称完整对话独立。旧 20 条来自 speaker 1/2。

1. **语言确认**：speaker 13/14，各取 20 条英文，实际解码时长 1–12 秒。seed=20261020 按 SHA-256(seed, clip_id) 顺序取，不能按模型错误选。文件及 PCM 与旧 20 条无重合。对同 40 条运行自动语言/已知 English，固定模型、dtype、线程、aligner 与解码上限。总体按总编辑数/总词数；输出配对改善/退化/不变及 1000 次 clip bootstrap（仅片段重采样，不作说话人总体推断）。English 是已知真实语言条件，不能据此称自动路由改善。
2. **正式 pilot 划分**：另取 speaker 5/6 为 dev，18/24 为 test；其他说话人为 train，排除 1/2/13/14。三种角色按 speaker、源 ID、文件和 PCM 哈希验证隔离。dev 目标 40（mixed20/zh10/en10），test 目标 100（mixed50/zh25/en25）；若第一分片无法满足则报不可行，不看输出后补选。已有模型预训练是否包含公开数据未知。
3. **训练池**：剩余合格 train 最多 2000，先审计/精确去重；reject 和未裁定 needs_review 不训练，source_reference 可用但不冒称人工纠正。候选与桶配额在基座转写前冻结；按语言/时长桶的可用数，最大余数法分配每臂 mixed250/zh125/en125，桶为 [1,3)、[3,6)、[6,12]。不足时另记较小 pilot，禁止默默缩样。
4. **A1/A2**：A1 每桶 seed 固定随机抽；A2 每桶按参考一致性 MER、silence_fraction、clip_id 递增挑选。若两臂总音频时长差超过 1%，A2 允许预定义的同桶交换：每次取最缩小与 A1 时长差的交换，同差取最少排名代价，再按 ID；满足 1% 即停止，无法匹配则不训练。记录交换与交集。该筛选偏好基座容易识别的源参考，并非通用最优质量指标。
5. **训练**：先另立 8 train/4 dev smoke（从 train/dev 角色取，正式两臂恢复同一基座，不从 smoke 续训）。官方 qwen3_asr_sft.py 固定 commit 7c6daf77a2421100f5fb066495372c00129d39ff，源码 SHA 随运行记录。0.6B、batch1、grad_acc8、lr2e-5、1epoch、worker0、persistent0；H100 BF16、单 GPU，seed 使用并验证 Trainer 默认42。smoke save_steps1；正式 save_steps50/save_total_limit2。实际 step 数、有限 loss、模型/处理器文件及 checkpoint 真实重载均需通过。
6. **正式评测**：A0 基座/A1随机/A2筛选在同一 H100 环境、同一冻结 test100、相同自动语言解码比较。另报语言组与逐条计数；差值/区间允许负结果。40 条语言确认不作为正式 SFT test。此轮不报告流式、麦克风或首声。

资源：仅使用用户指定 H100 主机上独立 /mnt/data2/exx/duplexlab-20261020 目录及空闲 GPU；不触碰 apex、不终止其他进程。连接信息和凭据不进入源码/文档。每个条件新输出目录；训练过程不调 test 或按初步收益提前停止。检查源文件、模型与数据哈希，所有原文/音频/模型只保留私有 data/runs/checkpoints。
