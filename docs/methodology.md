# 测量方法与数据来源

服务器实验执行于 2026-11-15（Asia/Shanghai）；JSON 中 utc 保留 UTC，因此部分记录日期为 2026-11-14。公开版整理在同一天完成，未重新执行服务器实验。

## 计时与统计

计时共 43 个独立进程，另有 4 个 Profiler 进程。baseline 12 个、screen 10 个、confirm 21 个；每计时进程预热 12 次，采集 64 个模型样本与 32 个进程内 E2E 样本，共 2752 / 1376 个样本。

- 模型：输入已驻 NPU；计时前同步，执行模型，结束时同步。排除分词、输入传输和输出传输。
- E2E：分词、H2D、模型、D2H 与同步；不含 HTTP 网络、排队、并发和模型加载。
- enqueue_wall：从模型调用开始到返回主机控制权，可能受框架与队列背压影响，不是纯 CPU 或纯提交时间。
- 分组统计：先对每个进程求 P50，再求三个 P50 的中位数；没有把全部样本混在一起求 pooled P50。
- 图中的误差条：三个进程 P50 的最小至最大值，不是置信区间。screen 只有一轮，不能当作正式三轮确认。
- 数据来源：固定性能请求前 128 条、B8/S256，43 个进程 data_sha256 相同；cgroup 节流计数增量和稳态编译增量均为零。

A 与 D/T 是分阶段对照。三轮能观察基本波动，不能排除全部时段漂移。图模式 FP32/BF16 在同一确认阶段交错运行。首次图编译归入预热；它仍是在线服务启动时需要承担的成本。

## 文件地图

所有结果位于 [results/2026-11-15](../results/2026-11-15/)。

| 文件 | 内容 |
|---|---|
| runs/*/result.json | 43 个计时进程的原始样本与元数据，加 4 个 profile_only 元数据 |
| process_summary.csv | 43 行逐进程统计 |
| group_summary.json / .csv | 21 个 phase/group/variant 汇总，screen 单列 |
| micro_matched.json | 36 个微基准分组、3600 个原始同步样本 |
| micro_summary.csv | 3 个序列长度 × 4 个实现的三轮汇总 |
| micro_accuracy_fp64.json | 相同 seed/shape/mask 下的独立精度归约；不是重新计时 |
| profiles.json | 四组 Profiler 的 kernel、Computing、gap 和 host 算子计数 |
| profiler/*/*.csv | kernel 和 step 时间明细，目录不含原主机名或 PID |
| quality.json | 四组候选对冻结 CPU FP32 参考的完整验收输出 |
| SHA256SUMS.json | 本目录公开数据文件的 SHA256 |

使用 Python 标准库即可复算：

~~~powershell
python tools/verify_results.py
~~~

脚本核对文件 hash、数据集 hash、原始样本统计、43/21 数量、筛查与确认分离、micro 汇总、Profiler kernel/Computing/gap 及质量状态。它不调用 NPU，不会凭汇总数字重建原始样本。

重绘需要 matplotlib 和 numpy：

~~~powershell
python -m pip install -r requirements-figures.txt
python tools/plot_results.py
~~~

## Profiler 的边界

每组采集 3 个 active steps。kernel duration 合计与 Computing 是不同来源的指标，不能假设完全相等。gap 按每步 kernel 起止区间并集计算；只描述首个 kernel 到最后一个 kernel 之间未被这些 kernel 覆盖的区间，不是整卡利用率或独立 CPU 时间。

公开版保留 kernel/step CSV，因此设备侧统计可以重算。host 算子调用次数保留为经原 trace 汇总的计数；为了避免分发主机标识和不必要的大文件，公开版不包含完整 trace_view.json，无法仅凭公开文件重新抽取 host 调用序列。

## 脱敏与证据保留

本仓库没有复制原日志整包、服务器账号/地址、认证文件、远程控制脚本或模型权重。逐进程 JSON 用白名单保留科学测量相关字段，移除 PID 和完整线程列表；CPU 亲和范围作为实验变量保留。quality 中原参考目录改成复现用相对说明；Profiler 输出目录改为实验名。时延、原始采样、指标、精度设置、模型提交和输入 hash 均未改写。

原始记录中的 embedding 快速一致性检查使用 FP32 归约，并且只覆盖性能输入；完整质量结论使用 quality.json，微基准精度结论使用独立 FP64 归约文件。不要将这些检查混成一个质量门槛。

模型 manifest 的 SHA256 固定了本次下载文件集合；其他镜像缓存若同时下载了额外权重格式，manifest 文件集合可能不同。复现时应记录自己的完整 manifest，不能只改写 hash 让它看起来一致。

## 公共包装器与原实验的关系

原模型迁移代码和实验数学路径来自已执行的工程。公开版新增 --lab/--root/--cpus 参数、拒绝覆盖检查、候选白名单、数据审计和绘图。公开包装器经过本地检查，没有重新在 NPU 执行。原实验结果对应的是已封存版本；修改脚本或环境后产生的新数据必须另存目录并注明版本。
