# 覆盖约束筛选：train/dev设计与新拆分验证

预写日期2026-10-20；在本轮选数、训练、新测试ASR输出之前确定。上轮纯参考一致性筛选的负结果保留，本轮不在旧400条上挑参数。

## 本轮隔离与新测试的范围

- 新train从原1465个audit pass候选去掉**所有speaker10/16/26**，余1015条，speaker8/9/11/15/19/21。dev沿用speaker5/6的40条；旧1/2、13/14、18/24、3/4/17及官方validation全部不进入本轮训练。
- 新test仅来自其余两个官方train分片中的speaker10/16/26；排除任何已有支线manifest/转写clipID、文件或PCM重复。每人目标mixed50/zh35/en15，SHA256(`20261021:clip_id`)顺序，实际时长1–12秒、自动audit pass后取足。若任一人的英文pass少于15但至少10，三人统一改为mixed50/zh40/en10；仍不足则显式不可行，不按模型错误换人或换样。数据审计与最终manifest在训练前冻结。
- 这是**新录音、本轮train/dev/test说话人隔离**，不是整个项目从未见过这些人的声明：三位曾出现在旧训练池，旧项目还曾读取全部train文本构建词表。因此必须六组从原始基座重训，不复用任何旧SFT checkpoint，不使用旧词表上下文；旧pool ASR只复用本轮train的子集。公开语料的基座预训练曝光未知。
- 旧200/725 validation协议不改、不重跑。新test不参与策略设计、特征阈值、挑checkpoint或早停。即使dev结果负，也按原定六组完整报告新test，不选最好seed。

## 单一候选策略与对照

1. 原基座、归一化不变，训练池difficulty是基座与原参考的MER代理：easy=0，medium=(0,0.20]，hard>0.20。不是人工确认的声学难度。只用本轮train和dev的基座输出审查分布，阈值固定，不读取新test输出。
2. 填充词覆盖代理：参考出现任一`嗯呃啊额唔诶欸`，或独立英文`uh/um/hmm/erm/er/ah/eh`；英文按ASCII字母边界、忽略大小写。`like`等实词不直接当填充词；不宣称这套词表穷尽口语现象。
3. Random500：仅按本轮train语言/时长分层，固定selection seed20261021随机取；mixed250/zh125/en125。桶仍[1,3)/[3,6)/[6,12]，按候选数最大余数法分配，不按模型错误抽样。
4. Coverage500：0/1整数选择，在与Random相同的**语言×时长、语言×speaker、语言×difficulty、语言×填充词有无**四组边际计数约束下，最小化基座MER（加稳定clipID微小tie-break），且总音频时长与Random差≤1%。匹配边际不等于所有高阶联合分布匹配，不冒称speaker完全均匀。
5. 固定SciPy/HiGHS MILP，time_limit120s、mip_rel_gap0；只接受状态optimal或time-limit但已找到、逐项验证的可行解，目标不得差于Random。否则回退Random并标记选择退化为对照，不宣称优化成功；不根据dev/test结果改变约束。记录原始求解状态、objective/gap、边际、交换交集和时长。随机集合本身满足约束，确保可行性已知。
6. train1015/dev40的特征与基座错误分布保存后，冻结这一个策略，不进行多候选dev挑优。dev用于完整性/覆盖诊断和六模型效果对照，不是新测试。source_reference全部原样，不捏造人工审核。

## 等预算训练与顺序

两臂各500条、同dev40，**42/43/44全部六组从同固定Qwen3-ASR-0.6B基座新训**。均使用同seed适配器（本轮42也显式set_seed），batch1/grad_acc8/lr2e-5/epoch1/BF16/worker0，63step，保存上游固定脚本及适配器哈希，核对实际training_args.bin与原始trainer_state；loss/grad_norm/eval_loss有限且实际重载才完成。

先训练六组并完成六组dev40重载评测，冻结完成记录，再一次性评测新test基座及六checkpoint。各条件自动语言、BF16/SDPA、max256、aligner关闭，原评分不变。不会按dev最优seed/step替换63步模型。

## 统计、交付与资源

Random和Coverage对应原比较器的random/filtered键，filtered在本轮专指覆盖约束策略。逐seed/语言/speaker/filler组给S/D/I/N，差=Coverage−Random；报告三seed均值、样本SD、范围、speaker等权macro，以及固定seed20261020的5000次配对交叉seed/完整speaker bootstrap。每个seed算micro差再平均；不把三次同test当独立样本。仅3训练seed/3测试speaker，区间探索性；固定选数集合尚未测selection seed方差。训练/推理时长只报完整批次含加载，不当作首字、TTS或端到端延迟。

独立目录coverage-v1，物理H100 GPU1且每阶段先核对空闲，发现外部占用等待，不杀外部进程，不访问apex。失败记录保留、输出create-only；公共统计/源码与私有原音/全文/权重分开。完成后同步双语README、PROJECT_STATUS/EVOLUTION和samples，无子代理、自动提交或推送。
