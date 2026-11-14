# 官方资料与采用范围

资料检索覆盖昇腾官网、GitCode Ascend、GitCode CANN 及模型作者。GitCode 动态页面无法稳定读取时，使用同一 Ascend 项目的官方 GitHub 镜像核对；不同分支不自动视为相同版本。下列链接支持接口与实验设计，不代替本仓库实测。

| 来源 | 用途与边界 |
|---|---|
| [BGE 固定模型卡](https://huggingface.co/BAAI/bge-large-zh-v1.5/blob/79e7739b6ab944e86d6171e44d24c997fc1e0116/README.md) | 模型身份、query 前缀、CLS/L2 用法；权重单独下载 |
| [Ascend PyTorch 配套表](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) / [官方镜像](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) | 配套组合核对；master 会变化，本案例以实测冻结版本为准 |
| [官方 CANN 容器](https://github.com/Ascend/cann-container-image/blob/main/README.md) | 拉取命令、设备和驱动挂载方式；本项目只映射两张分配卡 |
| [AscendHub 镜像入口](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884) | 用户指定 CANN 镜像来源；该动态页面未完整抽取正文，实际拉取成功记录来自原实验 |
| [TorchAir GitCode](https://gitcode.com/Ascend/torchair) | PyTorch 图模式推理；本课通过 torch.compile 调用 |
| [FIA GitCode 文档，26.0.0](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) / [官方镜像](https://raw.githubusercontent.com/Ascend/op-plugin/26.0.0/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) | BNSD/BSH、mask、sparse_mode 和返回值；实际支持仍以安装版探针和模型验收为准 |
| [CPU_AFFINITY_CONF 官网](https://www.hiascend.com/doc_center/source/en/Pytorch/2600/comref/Envvariables/docs/en/environment_variable_reference/CPU_AFFINITY_CONF.md) | 自动亲和参数；手工 taskset 的核号来自本机拓扑 |
| [TASK_QUEUE_ENABLE 官网](https://www.hiascend.com/doc_center/source/en/Pytorch/2600/comref/Envvariables/docs/en/environment_variable_reference/TASK_QUEUE_ENABLE.md) | 两级流水与适用限制；参数在进程启动前设置 |
| [昇腾在线推理实践](https://www.hiascend.com/developer/techArticles/20260303-1) | 在线 PyTorch 路线背景，不沿用其模型性能 |
| [Ascend text-embeddings-inference](https://gitcode.com/Ascend/text-embeddings-inference) | BGE 任务在昇腾生态中的选型线索；本工程不是 TEI/ATB 实现 |
| [CANN GitCode 组织](https://gitcode.com/cann) / [ops-transformer](https://gitcode.com/cann/ops-transformer) | 融合注意力实现检索范围；9.1.0-beta.1 tiling 搜索结果未取得完整正文，未用其推断本机 9.1.0 的具体内核因果 |
| [Transformers v5.3.0 BERT](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/bert/modeling_bert.py) | 注意力模块签名、BF16 softmax 参考和返回契约 |

更细的源码提交、获取日期和原来源内容 SHA256 见 [sources.json](sources.json)。torch_npu 源码核对提交为 334dadc9a4a80a9c04ac6ec3c92182ccf525d46a；op-plugin 为 8b9c8534fa41eff367a41c155843daa530ab3a08。源码解释与运行时观测相互核对，不把源码中的一种可能路径当作已经测到的执行路径。

课程的 CLI、数据集、质量阈值、测量设计、图表和结果解读为本项目内容。第三方源码及网页以链接引用，未将整篇文档纳入本仓库许可证。
