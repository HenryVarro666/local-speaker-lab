# 本轮工程与实验验证

最近环境核验：2026-10-20，America/Chicago。下列验证仅覆盖本次离线 pilot，不代表部署、实时流式或实际人工审阅通过。

| 项目 | 实际验证 |
| --- | --- |
| TDD | 数据选择/导出、训练启动保护、配对比较、时长交换审计、完整步骤/有限梯度检查、人工纠正导出边界均观察到预期失败后实现；当前针对性23项通过 |
| 完整本地回归 | **534 passed / 3 skipped / 22 subtests passed，16.60秒，退出码0**；原始日志 continuation-final-tests.log 和退出码 continuation-final-tests-exit.json |
| 语言确认 | 新speaker13/14各20条，两条件各40/40完成；配对总词数239一致；不重用旧5条作确认结论 |
| 数据与预算 | 训练池1465、dev40、test100、新语言40；speaker/ID/文件/PCM隔离；各臂500，mixed250/zh125/en125及相同时长桶；总时长差0.4547%，交集229，无需时长交换 |
| 实际训练 | 独立8/4 smoke完成1step并真实重载转写；两个500/40正式模型各63step、epoch1，日志loss/grad_norm/eval_loss有限，均重载并转写完整100条 |
| 真实训练参数 | 从三个checkpoint各自training_args.bin核对seed42、BF16、batch1、grad_acc8、epoch1、lr2e-5、worker0/persistent false；eval batch8 |
| 测试与独立复算 | 三模型同100条/N1045；Mac用本地原音、冻结协议与原始SQLite结果重算各组分数和1000次配对区间，与H100逐项一致，无模型重跑 |
| 来源与源码 | 固定模型/数据revision、Parquet/manifest/导出/模型文件哈希；执行时ASR及训练源码副本归档。原glossary协议、脚本与计分器未改 |
| 页面 | 私有三模型回放100个案例/播放器；抽查3个音频metadata无解码错误，过滤显示18改善/13退化/69相同编辑数；390px无横向溢出，JS错误0；未声称人工听音 |
| 依赖 | Mac dev pip check、独立ASR uv pip check，以及H100独立环境uv pip check检查122包均通过 |
| 静态与入口 | Ruff E4/E7/E9/F、git diff --check通过；工作区六份相关文档128个本地链接零失效；公开图与私有页面截图已实际查看 |
| 资源与退出 | GPU0已有Avatar任务，改用物理GPU1；只停止本次初始协调器及其后代并保留现场。正式任务结束GPU1恢复14MiB，独立ASR/训练进程已退出，GPU0既有进程保留；未访问apex |

全量测试会包含同工作区另行开发的 scheduling 模块；其可选 aiohttp、原Gradio未安装时单独列出跳过，不计为已通过行为。本轮没有修改或撤销该模块的并行工作。

训练源版本与工程边界：本轮训练全用source_reference，没有真实人工纠正。实际执行源码保存在私有training-source/及TRAINING_SOURCE.json；模型训练完成后新增的“已回听纠正可解决needs_review导出”回归修复没有用于本次训练，不把测试构造审阅当真实标签。后续显式UTF-8文本IO使跨平台行为更明确；本轮Mac/Linux实际产物本来就是UTF-8，分数未因此重跑。

122包完整GPU环境版本快照在[duplexlab-h100-py312.txt](../../../../requirements/locks/duplexlab-h100-py312.txt)，只对应本次Linux/Python/CUDA组合；不是任意干净主机安装保证。正式与smoke导出的每个JSONL哈希、角色、原音及标签前缀均在Mac复验一致。

本机证据在 runs/duplexlab/bootstrap-20261020/、runs/duplexlab/confirm40-* 与 runs/duplexlab/h100-continuation-v1/；GPU1采样记录为2秒间隔，最高9623MiB，非完整allocator峰值。独立H100数据盘保留环境、原音和checkpoint。公开[SUMMARY.json](SUMMARY.json)与图不含原文、预测、原音、本机/服务器路径或凭据。

尚未验收：人工实际回听、更多说话人和训练seed、近重复/说话人识别去重、原生流式兼容/延迟、真人麦克风和声音输出；未制作90秒演示录像或部署。所有当前差值区间跨零，不能将工程通过当作筛选有效性的统计证明。没有提交、推送或子代理。

最终文档更新后，源码包与文档链接两项检查再次通过；静态检查和diff格式仍通过。END_HEALTH快照确认本次模型进程已退出，物理GPU1为14MiB，GPU0既有进程仍在；本次SSH复用连接已关闭。
