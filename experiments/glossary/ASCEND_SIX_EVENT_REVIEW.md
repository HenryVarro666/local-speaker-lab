# 旧 200 条的 6 个关键事件：冻结前证据复核

本复核在剩余 validation 推理前进行。问题是：旧报告中的 5 次术语漏识与 1 次新增错误候选，哪些能由文字规则解释，哪些仍需要回听？本复核不修改词表、参考、预测、主评分、原来的 200 条或下一批的选样规则。

## 复核能力与边界

逐条比对原始参考、三个条件的原始预测，重新执行 `nfkc_han_latin_digits_v1` 主评分，并与旧 `comparison.json` 的逐提及诊断交叉核对。音频文件的存在、16 kHz 单声道 PCM16 格式、时长与 SHA-256 可检查；**当前代理的工具明确不支持音频输入，因此没有完成人工回听，也没有声学裁定**。参考文本与多个模型输出一致，都不能替代听音证据。

没有声称盲审：本复核者已知旧预测。提供本地回听页面和待填写模板，让能够听音的审阅者保留听到的文字、置信度以及是否先看过参考。机器生成的模板不能算人工标注。

## 逐事件判断

| ID | 冻结参考片段 | 三条件的相关预测 | 已确定的计分解释 | 尚未确定 |
| --- | --- | --- | --- | --- |
| `00621` | `bed time` 中的 `time` | 均为 `bedtime` | 合写之后不含独立 `time` token；这是可由文本确认的边界差异，不能直接解释为概念丢失 | 实际语音及是否还有其他错误，未回听 |
| `00062` | `smart phone` | 均为 `smartphone` | 最长非重叠匹配以双词为一个参考提及；合写不命中它，同时形成另一个字面候选 `smartphone` 的替换 | 实际语音及最佳书写规范，未回听 |
| `00957` | `Hong Kong` | 相关地名均为 `香港` | 汉字不命中英文字面词项；可确认书写体系差异；同句其他词的差异另计 | 是否为忠实转写、翻译或其他现象，未回听 |
| `01010` | `play time` 中的 `play` | 均为 `free time` | 相对冻结参考的词替换；train 词表未修复 | `play` / `free` / 其他，声学未裁定 |
| `00928` | `life in university` 中的 `life` | 均含 `London University` | 相对冻结参考存在替换与缺失；train 词表未修复 | 真实词语及参考可靠性，声学未裁定 |
| `00904` | `cultural universities` | 无词表／无关为 `Culture Universities`；train 为 `Culture University` | train 相对参考新增 `university` 候选；编辑对齐为 **替换**，不是插入。三个条件另有 `cultural` / `Culture` 差异 | 末尾单复数及参考可靠性，声学未裁定 |

6 个事件中，train 相对基线没有改变任何参考目标词的命中状态。第 6 个来自无目标词参考，影响的是错误候选计数。前 3 个可以确认是严格文字口径与预测写法的差异；这只解释指标来源，不能证明录音听辨正确。后 3 个保留为声学未知。**5 次主指标漏识和 1 次新增替换均原样保留，没有以诊断为由重新出分。**

## 原始证据与本地回听

机器可核验记录见 [six_event_audit.json](audits/ascend-remaining-v1/six_event_audit.json)，包含参考、三个条件、重新计算的计数、音频头信息及文件哈希。转写片段源于 [CAiRE/ASCEND 固定 revision](https://huggingface.co/datasets/CAiRE/ASCEND/tree/737e9800ae31be9932ba8464c80366559bd28424)，这些参考转写沿用 **CC BY-SA 4.0**；原创代码的 MIT 许可不覆盖数据许可。

从项目根目录生成本地回听包（音频仍留在原数据目录，不进入 Git）：

```bash
python scripts/make_glossary_review_packet.py \
  --manifest experiments/glossary/data/ascend-validation-pilot200-train-v2/manifest.jsonl \
  --run-dir runs/glossary-ascend-val200-train-vocab30-cpu-001 \
  --output-dir runs/ascend-six-event-review-v1
```

打开生成的 `index.html`，先播放再展开参考与预测；如已知答案，如实注明。模板中的听辨字段初始为空。任何后续人工结论另存补充审查，不能覆盖冻结主结果，也不能据此修改本轮剩余样本或词表。
