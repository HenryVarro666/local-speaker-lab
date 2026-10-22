# 工程验证与待办

环境最近核验：2026-10-20，Apple M4 Pro / 24 GiB / macOS arm64，Python 3.12.15。本页验证本地候选，不代表远程 CI、干净安装或部署已通过。

## 验证记录

| 检查 | 结果与范围 |
| --- | --- |
| 原测试基线 | 初次 19 failed / 440 passed / 1 skipped；GNU realpath 命令级 PATH 后剩 3 failed / 456 passed；修正 venv Python 解引用后 460 passed / 1 skipped / 22 subtests |
| 最新全量测试 | **493 passed / 1 skipped / 22 subtests passed，12.95 秒，退出码 0**；原始日志与退出码在本机 bootstrap-20261020/final-tests.log 和 final-tests-exit.json |
| Python 静态检查 | Ruff E4/E7/E9/F 检查新增模块、worker、打包清单及对应测试通过；git diff --check 通过 |
| 依赖一致性 | pip check 通过；隔离 ASR uv pip check 检查 97 包，通过 |
| 完成后恢复 | 真实 20 条成功结果保留；worker 日志仍为 1 个、内容哈希不变；没有重新加载模型 |
| 模型与源码 | 最后两个条件的模型文件及实际 ASR 源码指纹相同；本地固定 snapshot 加载，未隐式拉取 aligner |
| 浏览器 | 临时独立 Chrome 上检查最终 002 回放；20 播放器、3 个 metadata 可读、无解码/JS 错误；390px 无横向溢出；审阅初始全未选、缺审阅者不能导出 |
| 公开汇总 | 无音频、原参考、预测和本机绝对路径；[统计](SUMMARY.json)与[来源指纹](INPUTS.json)可公开审查 |
| 归档入口链接 | voice_ai 六份入口文档检查 64 个本地链接，零失效；最终更新后两项源码包/链接检查再次通过 |
| 旧实验 | glossary 的协议、脚本与计分源码无 diff；725 条任务未访问/启动/重启，实时完成状态未知 |

最终文档接入后，已有 source-archive 测试发现新报告未被打包，触发 2 项失败（491 passed）。按既有保护方式显式添加已审查的四个报告文件与 README，不递归包含实验目录；后续未审查报告、原音和原始预测仍不进入源码包。修复后针对打包与仓库检查为 13 passed / 1 skipped，再运行完整套件，结果见上表。初始集成失败日志另存 pre-package-fix-final-tests.log。

测试运行使用命令级 GNU realpath 路径，不改系统配置：

```bash
PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH .venv-dev/bin/python -m pytest -q -rfs
uvx --from ruff ruff check --select E4,E7,E9,F src/speaker_lab/duplex src/speaker_lab/runner.py scripts/package.py tests/duplex_helpers.py tests/test_duplexlab_*.py tests/test_worker_contracts.py
.venv-dev/bin/python -m pip check
uv pip check --python .venvs/asr-mac/bin/python
```

## 尚未验收

- 人工实际听音、源参考纠正和主观音质。页面与单元测试不产生真实人工标注。
- 原生流式、TTS/声音输出、真人麦克风、Avatar、工具 Agent、训练导出及任何 ASR 微调。
- 音频审计的损坏/零帧、精确 1/12 秒、有停顿及幅度不同的专项边界测试。
- 遗留 running 的进程核实与自动恢复；当前保守拒绝。
- 每条识别计时、模型加载分项、严格冷启动、稳定 p95、MPS/MLX/CUDA，以及独立真人泛化评测。

代码仍在本地 feat/duplexlab 的未提交改动中；实际推理源文件已保存在 run/source/。本轮没有访问 apex、修改其他服务、提交、推送或部署。
