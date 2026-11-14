# 第三方材料

本仓库 LICENSE 适用于原创教学代码、教程、图表与合成数据，不覆盖以下第三方组件：

- BAAI/bge-large-zh-v1.5：模型由 BAAI 发布，权重不随仓库分发。下载前查看固定模型仓库的模型卡和许可。
- PyTorch、torch_npu、TorchAir、CANN、Transformers、tokenizers 等：通过依赖安装或指定镜像取得，各自许可与分发条件独立适用。
- 昇腾和模型作者的技术文档、源码：在 docs/sources.md 中链接引用，未将全文复制并重新授权。

代码中的 FIA 适配器按官方 API 和安装版 Transformers 接口编写。本仓库不是 Ascend、Huawei、BAAI 或 Hugging Face 的官方项目；这些名称用于标识硬件、软件和模型。

合成教学数据包含虚构政策，仅用于功能、数值一致性和小规模检索验收，不作为真实服务承诺或生产评测集。
