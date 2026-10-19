# 仓库整体审查记录

审查日期：2026-10-19；源码版本：0.1.4。范围包括模块边界、模型扩展协议、依赖构建、CI 与公开发布。修复保留现有 CLI、Gradio、独立模型环境和 0.1.3 → 0.1.4 升级入口。

## 发现与修复

| 问题 | 影响 | 修复与验证入口 |
| --- | --- | --- |
| 同一阶段存在旧 `response.json` | 新 worker 没写响应也可能使用上次成功结果 | 启动前清理旧响应，严格验证新 JSON 封套；`tests/test_worker_contracts.py` |
| 插件请求/响应缺少边界校验 | 无效请求可能先加载模型，非有限数和错误结构继续传播 | 插件导入前检查请求，限制结果对象及标准 JSON；worker 契约回归 |
| 分离仅检查来源个数 | 无效、重复或格式错误的文件直到匹配模型加载后才报错 | 在匹配前核对不同绝对文件、16 kHz/单声道/有限值/时长；`tests/test_pipeline_contracts.py` |
| ASR 边缘容差和匹配元数据解释不完整 | 区间裁剪后可能倒置，字符串布尔值或无效 margin 被误用 | 统一裁剪并验证时间，严格检查可选元数据；流水线契约回归 |
| 构建工具与第三方 sdist 隔离构建版本漂移 | 同样源码可能使用不同构建环境 | 独立 `requirements/bootstrap.txt`、固定构建后端、pip build constraints；安装与恢复测试 |
| CI 的跨平台矩阵只运行 unittest 子集 | 大量 pytest 函数只在单个平台执行 | 矩阵执行完整离线 pytest，各自保存报告；Ubuntu 锁环境另跑应用集成 |
| 部署替身沿用测试主机的 Python 版本 | 在 3.11 上提前触发目标部署的 3.12 守卫，未执行预期阶段 | 测试独立模拟 Ubuntu/Python 3.12，保留产品版本要求；通过真实远程矩阵核验 |
| Windows 测试依赖隐式 shell 路径和链接路径表示 | 假命令目录未正确进入 PATH，`readlink` 的扩展路径前缀造成误判 | 显式将假命令目录转换为 shell 绝对路径，并比较升级前后保存的链接目标 |
| 测试从 checkout 导入，未验证分发安装 | 可能遗漏 wheel/sdist 文件或依赖源码路径 | 真正执行 ZIP → sdist → wheel → 纯新 venv、仓库外 CLI/lazy import；`tests/test_distribution.py` |
| 安装 Gradio 后未构造界面 | API 不兼容可能在用户打开 UI 时才暴露 | 无服务、无模型的 Blocks 构建冒烟 |
| ZIP 递归收集跟随链接且时间戳不固定 | 可能读到项目外文件，重复构建校验和不同，失败破坏旧包 | 严格源码枚举、拒绝链接、固定元数据、临时文件原子替换；`tests/test_package.py` |
| Windows 发布的校验文件使用 CRLF | GNU `sha256sum -c` 将回车当作文件名字符而失败 | 固定 `.sha256` 为 LF，跨平台回归并实际执行 GNU 校验 |
| 公开入口缺少集中引用与许可证范围 | 原创适配、上游模型和数据来源不清 | [MIT](../LICENSE)、[上游与数据说明](../THIRD_PARTY_NOTICES.md)、[CITATION.cff](../CITATION.cff) |

## 验证边界

轻量回归不安装 GPU 模型运行时、不下载权重，也不以模拟后端的输出评价分离或识别质量。CI 的全套输入和命令见[开发与验证](development.md)及 [workflow](../.github/workflows/tests.yml)，远程状态以 [GitHub Actions](https://github.com/HenryVarro666/local-speaker-lab/actions) 对应提交为准。

Ubuntu + RTX 5080 的驱动、原生解码、CUDA 算子和真实录音仍由[一键部署与验收](deployment.md)在目标机器执行；该硬件验收与仓库 CI 分开报告。
