# 参与开发

先读[快速开始](docs/GETTING_STARTED.md)和[架构说明](docs/architecture.md)，确认变更属于单人词表实验还是双人工作流。核心包仍为 `src/speaker_lab/`；模块整理与模型效果实验分别记录，便于复查每次变化的作用。

## 开发环境与测试

核心开发和 CI 覆盖 Python **3.11 / 3.12**；Ubuntu 模型环境使用 Python 3.12。在独立虚拟环境中先安装固定构建和测试工具，再安装项目：

```bash
python -m pip install --upgrade --only-binary=:all: -r requirements/bootstrap.txt
python -m pip install --build-constraint requirements/bootstrap.txt -r requirements/dev.txt
python -m pip install --no-build-isolation --editable ".[dev]"
```

Windows 与 Ubuntu 创建、激活环境的完整命令见[快速开始](docs/GETTING_STARTED.md)。完整轻量回归需要 FFmpeg 和 Gradio；Ubuntu/Python 3.12 的可复现安装、检查与 CI 范围见[开发与验证](docs/development.md)。模型包按工作流保留在 ASR、分离等独立环境，开发检查无需合并这些依赖。Ubuntu 的环境要求与锁定快照见 [requirements/README.md](requirements/README.md)。

训练词表提取与固定抽样的合成测试使用标准库 `unittest`，不下载数据或模型：

```bash
python -m unittest discover -s tests -p test_train_glossary.py -v
python -m unittest discover -s tests -p test_select_glossary_pilot.py -v
```

其余现有测试主要使用 pytest。运行与改动相关的测试；需要完整轻量回归时执行：

```bash
python -m pytest -q
```

单元/合成测试检查计分、抽样、接口和失败处理。CI 使用 pytest 在 Windows / Ubuntu、Python 3.11 / 3.12 上运行完整的可移植测试，并在固定的 Ubuntu 应用环境检查 Gradio、FFmpeg、sdist / wheel 的实际构建安装。分发测试会禁用网络下载并在新环境、仓库外启动帮助入口；不要用 checkout 的 `PYTHONPATH` 掩盖缺失文件。

真实模型加载、目标设备兼容性、推理速度和识别质量使用独立运行或验收记录，写明机器、配置、模型 revision 与实际执行范围。双人真实录音验收见[完整指南](docs/TWO_SPEAKER_GUIDE.md)。

## 实验规则

词表规则、样本选择、提示、模型配置和计分方式应在查看本轮预测前冻结，并保存来源与 SHA256。当前协议见[词表实验](experiments/glossary/README.md)和[200 条冻结协议](experiments/glossary/protocols/ascend-val200-train-vocab30-v2/)。

- 词条候选只来自允许的训练文本。禁止按验证/测试参考答案、错误案例或覆盖率手选、补充或重排词条；无关词表采用独立固定来源与预先规定的排除、匹配规则。
- 按固定 ID、分组、时长和种子抽样。禁止用参考词汇或模型好坏筛选验证样本；扩大样本时保持已经冻结的规则，并核对原样本的包含关系。
- 参考转写仅供离线计分，不作为 Qwen 提示。三组使用同一音频，整批固定词表，保留未经后验修改的原始预测。
- 分析保留既定归一化、最长完整词条匹配和编辑对齐。不要为了改善结果增加缩写、同义词、翻译或人工修复。
- 对照必须配对完整。运行中仅检查进度和失败状态；按当前分析计划，600 次计时推理完整后才进行主要结果分析。
- 若发现通用实现错误，以新版本规则、配置或协议修复，保留原产物和原因。新版本与旧版本分别报告，不能把对验证结果的调整称作训练集规则。

## 文件和运行产物

每次推理使用新的 `runs/<实验名>/`，提取和抽样也使用新输出目录。重新计分或分析使用新的输出文件。工具拒绝覆盖已有产物时，选择新的明确名称，不删除或重写原目录来绕过检查。

运行过程中保留清单、配置、词表和脚本的冻结版本；改进说明文档不影响正在执行的输入。更新配置时保留可复核的旧哈希，区分“运行时使用的版本”和“分析时读取的源码版本”。

模型权重、音频、完整数据副本、虚拟环境和大运行日志保留在本地。共享实验时优先提供小型协议、ID、词表、数据来源与哈希；尊重语音资料的访问权限。不要提交凭证或将其写入 TOML、日志与文档。上层 `sources/` 是只读的同步参考目录。

## 提交清晰的变更

说明具体问题、最终行为、验证方法和仍未验证的设备或模型。涉及实验时列出变化的因素与冻结输入；涉及文档时检查相对链接、命令和状态描述。不要把依赖解析、合成测试或正在运行的实验写成已完成的真实模型效果结论。
