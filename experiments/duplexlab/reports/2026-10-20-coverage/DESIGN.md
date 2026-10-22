# 覆盖约束选择设计与训练前冻结

2026-10-20。以下约束与选择集合在新训练、新测试输出前固定；只设计一个候选，没有dev多策略挑优。主[协议](../../COVERAGE_PROTOCOL.md)与哈希绑定manifest/SELECTION保留。

## 当前拆分与独立性范围

train1015：speaker8/9/11/15/19/21；dev40：5/6；新test300：10/16/26，其他分片中无旧片段重合，各mixed50/zh35/en15，全部audit pass。本轮角色说话人、ID、文件/PCM隔离。三位测试speaker曾在旧训练池，历史词表提取读过所有官方train文本，所以六模型均从同预训练基座新训，不用旧SFT/词表；不称整个项目第一次见这些人，不称已验证完整会话独立。新test预训练曝光未知。

## 约束与实现

固定selection seed20261021，Random500按语言/时长分层随机抽。Coverage500最小化sum(per-clip base-reference MER)，加极小ID排序tie-break，同时严格匹配Random的语言×时长、语言×speaker、语言×难度、语言×填充词四组边际，总音频时长差≤1%。匹配边际不保证所有高阶联合分布或桶内难度分布相同。

difficulty：easy=0，medium=(0,.2]，hard>.2，使用原基座与source_reference差异，非人工声学难度。filler：中文嗯呃啊额唔诶欸、ASCII边界英文uh/um/hmm/erm/er/ah/eh，忽略大小写；只计一段是否含词，不把like直接视为填充词，也不穷尽频次、重复、韵律或话轮现象。

| 语言 | 两臂各条数 | easy | medium | hard | 有填充词 | 无填充词 |
|---|---:|---:|---:|---:|---:|---:|
| zh | 125 | 71 | 46 | 8 | 41 | 84 |
| mixed | 250 | 53 | 137 | 60 | 102 | 148 |
| en | 125 | 44 | 41 | 40 | 32 | 93 |

| speaker | 两臂各条数 |
|---|---:|
| 8 | 44 |
| 9 | 170 |
| 11 | 43 |
| 15 | 55 |
| 19 | 76 |
| 21 | 112 |

## 实际选择结果

MILP报告optimal，gap约7.8e−9，耗时0.069秒，无fallback。两臂500条，交集317，并集683；Random1965.409秒、Coverage1984.1536875秒，差0.9537%。参考归一化token数7591/7863，非严格token/FLOP匹配。每臂1epoch、batch1/grad_acc8、63optimizer steps，三seed全部同配置；不因单seed dev好坏换规则。

sum clip MER71.1150→53.1299（平均14.2230%→10.6260%）是选择目标，不是下游或人工质量提升；UNK参考25→22均保留原文。中文非零难度54条和填充词41条、mixed难度hard60条与填充词102条均与Random相同；这解决上轮纯一致性筛选中文125/125全easy的结构问题，但准确率仍需新对照验证，且两轮拆分不同，不直接横比旧测试MER。

六模型从基座重新训练，42/43/44均使用同seed适配器，先六组dev40重载，冻结DEV_COMPLETE，再新test基座+六checkpoint。只有训练数据约束与诊断，没有test驱动调参。全部源参考未人工纠正。最终[结果](RESULTS.md)、[覆盖审计JSON](COVERAGE_AUDIT.json)和私有回放另存。
