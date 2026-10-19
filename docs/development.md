# 开发与验证

推荐在 Ubuntu 24.04 / Python 3.12 开发。应用层及轻量测试不需要 PyTorch、HF_TOKEN 或模型权重；GPU 后端各有独立环境，不要把四套模型依赖装进同一个开发环境。

## 本地轻量环境

从仓库根目录运行：

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv ffmpeg libsndfile1

python3.12 -m venv .venv-dev
.venv-dev/bin/python -m pip install --upgrade --only-binary=:all: -r requirements/bootstrap.txt
.venv-dev/bin/python -m pip install --build-constraint requirements/bootstrap.txt -r requirements/locks/app-linux-py312.txt -r requirements/dev.txt
.venv-dev/bin/python -m pip install --no-build-isolation --no-deps --editable .
.venv-dev/bin/python -m pip check

.venv-dev/bin/python -m pytest -q -rs
bash -n deploy_ubuntu.sh
bash -n scripts/install_ubuntu.sh
bash -n scripts/run.sh
```

上面安装应用层锁文件，包含 Gradio；构建工具和测试工具也有固定清单，第三方包需要构建时使用相同的构建约束。只做核心开发时，可以在固定构建工具安装后使用 `pip install --build-constraint requirements/bootstrap.txt -r requirements/dev.txt` 和 `pip install --no-build-isolation --editable ".[dev]"`，省去 Gradio；可选界面测试会明确跳过。

Windows 可使用自己的 Python 3.11 / 3.12 虚拟环境安装核心开发环境，Bash 测试需要 Git Bash。原生 POSIX 信号及符号链接权限相关测试可能跳过，应以测试输出为准；FFmpeg 未安装时对应真实解码测试会跳过，Ubuntu integration 必须安装 FFmpeg。Ubuntu 部署及 GPU 验收仍在目标机器执行。

## 验证层级

| 层级 | 实际验证内容 | 入口 |
| --- | --- | --- |
| 轻量单元/集成 | 时轴、协议、真实本地子进程、FFmpeg、HTTP替身、安装/升级控制流程 | `python -m pytest` |
| 分发安装 | 实际源 ZIP 解压，构建 sdist 及 wheel，再在无源码路径、无继承包的新环境安装本地 wheel，检查包位置、版本及 CLI/UI/worker 帮助入口 | `python -m pytest tests/test_distribution.py` |
| 可选界面构建 | 使用已安装的 Gradio 构造真实 Blocks，不启动服务器或加载模型 | `python -m pytest tests/test_repository.py` |
| 环境诊断 | 各环境依赖、NumPy ABI、TorchCodec 原生解码、CUDA 卷积/BF16/SDPA/STFT | `.venvs/app/bin/python scripts/doctor.py` |
| 模型代码冒烟 | ClearVoice 小型随机权重网络和真实 NumPy 接口 | `.venvs/separate/bin/python scripts/smoke_models.py --device cuda:0` |
| 完整模型验收 | 真实双人录音、四个 worker、分离、对齐转写、Ollama纪要与引用 | 一键入口或 `.venvs/app/bin/python scripts/acceptance.py recording.wav` |
| 人工质量检查 | 人物身份、串音、文字准确率、纪要事实与引用 | 回听音轨并对照产物 |

## GitHub Actions

同一 CI 工作流包含两个独立作业，均不需要 HF 凭证或模型权重：

| 作业 | 环境与依赖 | 检查范围 |
| --- | --- | --- |
| core | `ubuntu-latest`、`windows-latest` × Python 3.11、3.12；先安装固定构建及开发工具，再安装 `.[dev]` | 用 pytest 同时运行函数测试和 unittest 类，覆盖跨平台控制流程及真实分发构建安装；缺少可选 Gradio / FFmpeg 时明确跳过相应测试 |
| integration | Ubuntu 24.04 / Python 3.12；安装系统 FFmpeg、libsndfile1、固定构建工具、完整应用锁文件、`requirements/dev.txt`，再用 `--no-build-isolation --no-deps --editable .` 安装项目 | `pip check`、Bash 语法、`compileall`、全量 pytest（包括界面构建、真实解码和分发安装），以及源 ZIP 的 SHA-256 和 CRC 校验 |

两个作业在检查失败时也上传已经生成的 JUnit 报告；integration 另外保存 Python 版本和依赖快照。源包构建和校验成功后才上传发布 bundle。Actions 固定到审阅过的提交 SHA，仅有仓库读取权限，checkout 不保留 Git 凭证。缓存键包含相应的构建、开发和应用依赖清单。所有 editable 安装都关闭隐式构建隔离，分发测试禁止从网络获取包，以核对显式构建工具确实够用。

两个作业均不安装 NVIDIA 驱动、不调用部署入口、不下载模型，也不执行 GPU 模型推理；integration 只安装 FFmpeg 和 libsndfile1 系统依赖。

这些文件提供 CI 配置；仓库推送到 GitHub 后才会实际运行，不能将本地检查视为远程 CI 已通过。本地 Windows 验证限于可运行的轻量测试、Bash 语法和源包检查；信号及符号链接相关跳过项以测试输出为准。Ubuntu 安装、驱动重启恢复和真实 RTX 5080 推理仍须通过目标机器的一键验收及人工回听确认。

## 扩展与依赖

增加模型前先看 [后端协议](model-backends.md)。保持模型框架惰性导入、子进程环境隔离和原音频时间轴；不要在 Gradio 回调中直接加载模型。分离与匹配的 job ID 必须唯一并完整覆盖请求，非法结果应在启动下一模型前失败。

默认锁定环境见 [依赖维护](../requirements/README.md)。不要因为新增模型需要另一套 NumPy 或 Transformers 就替换所有环境版本；为新路线单独声明环境和配置，再用相同录音对比。

## 构建发布文件

```bash
.venv-dev/bin/python -m build --no-isolation
.venv-dev/bin/python scripts/package.py
cd dist
sha256sum -c local-speaker-lab.zip.sha256
../.venv-dev/bin/python -m zipfile -t local-speaker-lab.zip
```

`build` 生成标准 sdist / wheel；`package.py` 生成包含部署材料的 `dist/local-speaker-lab.zip`、匹配的 `.zip.sha256` 以及独立 `deploy_ubuntu.sh`。ZIP 中包含代码、配置、文档、CI 和测试；运行数据、用户词表、环境、模型、凭证和日志不在源包中。完整 Ubuntu 部署使用源 ZIP；wheel 的包安装测试验证 Python 分发边界，不会替代多环境安装器或将模型环境合并到一个包。
