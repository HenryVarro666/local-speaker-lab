# 剩余 validation 复验：逐步改进与执行记录

本轮受用户明确授权：复核旧 6 个事件，冻结新协议，跑完剩余合格 validation，保持词表及主评分不变，提交完整对比报告并更新 GitHub。以下记录工程和证据质量的改进，不把诊断当作模型增益。

机器可读时间线为 [execution_log.jsonl](audits/ascend-remaining-v1/execution_log.jsonl)；旧状态保护清单为 [preservation_before.json](audits/ascend-remaining-v1/preservation_before.json)。UTC 时间用于排序，当地时间为 America/Chicago。

## 1. 保留已存在的工作

记录旧 200 条的原始预测、分数、比较、配置、词表及源码 SHA-256。已有 200 条结果文档的未提交修改被保留，随本轮提交；没有回滚、覆盖或重新运行旧结果。原 24 条是 200 条的子集，始终不作为额外独立样本。

## 2. 先审查 6 个事件，再冻结下一轮

[逐事件报告](ASCEND_SIX_EVENT_REVIEW.md)和 [18 个条件分数的复算证据](audits/ascend-remaining-v1/six_event_audit.json)区分严格文字差异与声学未知。独立审计者复算结果相同；6 个音频哈希与旧 run 一致。当前代理不支持音频输入，未虚构人工回听；声学字段明确未裁定。

新增 `make_glossary_review_packet.py` 生成本地音频页面、默认折叠参考／预测、人工置信度和答案暴露记录及 JSON 导出。页面不复制音频、不改评分、不能自动升级为人工标注；已有答案暴露的审阅者不能声称盲审。真实旧数据 smoke 和合成测试均通过。

## 3. 从配额抽样改为取全部剩余合格片段

原 200 条采用 100 mixed／50 普通话／50 英文配额；新一轮直接保留全部剩余合格片段，减少另选子集的自由度。新增 `freeze_ascend_remaining.py`：只凭 ID、语言组和 WAV 头筛选，不访问词表或预测，不用参考作选择；验证 split、时长、路径、音频格式、重复 ID、旧 ID、音频重复及输出覆盖保护。测试还验证修改参考不改变选中 ID。

独立元数据审计与新脚本均得到 1130→925→725；新 725 为 107 mixed／453 普通话／165 英文，2409.865375 秒。925 份合格录音字节哈希全部唯一，旧新无 ID 和音频字节交集；没有 speaker/session 元数据，因此没有宣称说话人独立。

## 4. 锁定环境、来源与分析规则

[冻结协议](protocols/ascend-val-remaining725-train-vocab30-v1/)保存词表原样副本、配置、725 个 ID、全合格音频哈希、分析计划、源码和模型快照文件哈希。原 runner、主 scorer、比较分析脚本均不改动。保持 CPU FP32／8 线程／Chinese／greedy／256 token／原种子；没有 LLM 后处理。

复用原推理环境，没有升级依赖。该环境最初建立时未安装 pip，直接 `python -m pip check` 因缺少 pip 失败；改用已有开发环境的 `pip --python <原环境 Python> check`，确认无依赖冲突，没有向推理环境安装新包。原环境 editable 分发元数据仍显示 0.1.1；已核对实际 import 指向当前 checkout，并记录真正使用的源码哈希，避免把旧 metadata 当作执行代码版本。

本轮固定主结果依然是修复、破坏、错误候选增减，辅助报告 MER／切片／计时和原 1000 次语言分层 clip bootstrap。全部 2175 条配对完成之前不作治疗效果分析；失败保留输出并停止，不以中途效果决定停跑或增样。

## 5. 推理与验收

状态和后续逐步核验见机器时间线。本文件后续补充实际运行、独立核验、结果、测试及 GitHub 提交，冻结协议本身保持不变。
