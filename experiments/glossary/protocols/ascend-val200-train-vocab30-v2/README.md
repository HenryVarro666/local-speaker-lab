# ASCEND 200 条固定协议

这里保存本轮推理前已冻结输入的可审阅副本；正在运行的任务仍使用原始数据目录，没有替换它的输入。

| 文件 | 用途 |
| --- | --- |
| [protocol.json](protocol.json) | 数据 / 模型 revision、抽样规则、参数、来源及哈希 |
| [selected_ids.json](selected_ids.json) | 按固定规则选出的 200 个有序样本 ID，不含答案 |
| [correct.json](correct.json) | 只由 ASCEND train 构建的 30 项相关词表，字节不变 |
| [unrelated.json](unrelated.json) | 30 项无关对照词表，字节不变 |
| [analysis_plan.json](analysis_plan.json) | 推理完成前冻结的指标、切片和不确定性分析规则 |

验证集先按 1–12 秒筛选，再在 mixed / mandarin / english 各组内按 `(sha256("20261018:" + id), id)` 排序，分别取前 100 / 50 / 50 条。抽样函数只使用 ID、分组和时长。词表覆盖仅在选样冻结之后统计，不参与抽样。

`personal` 是 CLI 延续的条件名，本实验实际使用 train 词表，并非来自个人录音或个人笔记。重建方法见 [运行手册](../../README.md)。音频、参考答案和预测不随这个目录分发；从固定 revision 的官方数据重建。

词表文件按二进制写入，哈希可跨平台核对。`protocol.json` 中的 manifest 哈希描述本次 Windows 生成的清单；跨平台还应核对固定 ID、内容和音频，不能只依赖文本换行格式的字节哈希。
