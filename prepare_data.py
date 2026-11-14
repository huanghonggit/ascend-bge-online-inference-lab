"""Generate a small, original synthetic dataset; not an official benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

# Fictional policies at a fictional training centre. Each FAQ has two paraphrases.
FAQ = [
("账号激活", "星河学习中心的新学员收到邀请邮件后，打开激活页面，填写学员编号并设置密码。邀请过期时联系班级助教重新发送。", "第一次登录学习中心如何开通账号？", "激活邮件里的邀请失效了怎么办？"),
("忘记密码", "忘记学习平台密码时，在登录页面选择找回密码，使用注册邮箱接收验证码，验证后重新设置密码。", "我忘了学习平台登录密码，怎样找回？", "用注册邮箱可以重设密码吗？"),
("双重验证", "账号可以在安全设置中绑定验证器。更换手机前应保存恢复码；旧手机无法使用时，由助教核对身份后重置双重验证。", "换手机后收不到验证器代码怎么办？", "在哪里给学习账号开启双重验证？"),
("修改邮箱", "修改注册邮箱需要先登录个人资料页，验证旧邮箱，再填写新邮箱并完成验证。旧邮箱不可用时向助教提交资料变更申请。", "个人资料里的联系邮箱能修改吗？", "旧邮箱不能用了，如何变更注册邮箱？"),
("课程回放", "已报名课程的录播可在课程页的回放栏目查看。视频会在直播结束后整理上传，回放仅对本期学员开放。", "直播结束后在哪里看录像？", "错过上课时间还能补看课程吗？"),
("作业提交", "实操作业在课程页的作业栏提交。压缩包需包含代码、实验记录和结论，命名为学员编号与实验名称。", "实验代码和报告应该提交到哪里？", "作业压缩包需要包含哪些文件？"),
("延期申请", "无法按时提交作业的学员可以在作业截止前向助教申请延期，说明原因与计划完成日期，审批结果会记录在作业页。", "来不及交作业可以申请延期吗？", "延期申请要写哪些信息？"),
("结课证书", "完成必做实验并通过助教验收后，可以在个人学习记录中下载电子结课证书。姓名有误时先更正个人资料。", "通过课程考核后从哪里下载证书？", "结课证书上的名字不正确怎么处理？"),
("实验预约", "共享实验机需要提前在资源预约页选择时段。预约成功后页面显示服务器地址、账号和分配设备，资源仅在预约时段可用。", "怎样预约一台用于实操的服务器？", "预约成功后去哪里看服务器地址？"),
("预约取消", "不能使用已预约的实验时段时，请在资源预约页取消预约，把资源释放给其他学员。临时取消不会自动延长其他预约。", "已经预约的机器不想用了怎么取消？", "如何释放我预订但用不到的实验时段？"),
("连接实验机", "Windows 用户可以用 PowerShell 中的 ssh 命令连接实验机。服务器地址和端口以预约页为准，首次连接先核对主机指纹。", "Windows 电脑怎样远程登录实验服务器？", "PowerShell 连实验机应该用什么命令？"),
("上传文件", "向实验机上传小型代码包可使用 scp，目标目录选择个人工作目录。大文件由教师统一分发到课程公共目录。", "怎么把本地写好的脚本传到实验机？", "上传实验代码包应该放在哪个目录？"),
("指定设备", "实验账号只能使用预约页分配的设备。容器内的逻辑编号可能不同于宿主机编号，应通过环境检查确认后指定设备。", "容器里的设备编号为什么和宿主机不一样？", "实操时应该选择哪一张加速卡？"),
("释放资源", "结束实验前先保存代码与结果，再正常退出推理进程。确认没有自己的后台任务后，归还预约资源；不要终止其他人的进程。", "实验做完以后要怎样归还计算资源？", "退出服务器前需要检查什么？"),
("磁盘配额", "每位学员有独立磁盘配额。空间不足时先清理个人目录中可重建的缓存和重复结果，原始数据与模型权重由教师统一管理。", "实验目录磁盘空间满了怎么办？", "模型占空间太多，应该自己复制多份吗？"),
("结果备份", "重要实验结果应复制到个人持久化目录，再下载到本地备份。临时容器销毁后，未写入持久化目录的内容可能丢失。", "容器删除前如何保存实验报告和结果？", "临时目录里的文件会一直保留吗？"),
("下载模型", "课堂模型由教师提前下载并固定版本，学员从公共只读模型目录加载。不要在课堂上重复下载或修改公共权重文件。", "上课时每个人都要重新下载模型吗？", "课程的预训练权重从哪里读取？"),
("版本记录", "每次实验记录 Python、框架、设备插件、底层运行库版本，以及模型版本和数据校验值，方便之后复现。", "复现实验时需要保留哪些版本信息？", "为什么实验报告里要写模型和数据的校验值？"),
("环境冲突", "如果依赖包版本冲突，应切换到教师提供的课程环境，保留错误日志。不要直接升级共享系统中的全部软件包。", "安装 Python 依赖出现版本冲突怎么办？", "能不能通过全量升级共享环境修复报错？"),
("数据格式", "检索实验的文档和查询使用 UTF-8 编码的 JSONL，每行包含唯一编号和文本；相关性标注单独保存为查询与文档编号对应表。", "检索实验数据文件每一行应该写什么？", "查询和正确文档的对应关系存在哪里？"),
("空文本", "在线向量接口拒绝空字符串和只有空白的文本，调用方应先检查输入。非空长文本会按照课程规定的最大长度截断。", "只传空格给文本向量接口会怎样？", "文本超过模型允许长度怎么处理？"),
("查询指令", "语义检索实验给查询添加统一指令，文档不添加查询指令。所有对照实验保持该规则一致，避免预处理差异影响结果。", "文档和查询都需要添加检索提示词吗？", "不同优化方案能改变查询前缀吗？"),
("数值检查", "迁移后先检查向量形状、有限值和归一化范数，再与参考向量比较余弦相似度及绝对误差，最后检查检索结果。", "模型迁移后如何确认向量计算正确？", "只看程序没报错就能判断迁移成功吗？"),
("性能计时", "设备计算可能异步执行，测量前后需要同步。模型计时与包含分词、数据传输的端到端计时分开记录，预热不混入稳态统计。", "为什么推理计时要加设备同步？", "模型耗时和端到端耗时分别包含什么？"),
("首次编译", "图模式第一次执行可能包含编译，应单独记录首次调用耗时。固定输入形状充分预热后，再统计多次稳态运行。", "图模式首次调用很慢该怎么计入结果？", "为什么不能拿第一次图编译时间当稳态延迟？"),
("性能分析", "性能分析工具用来查看主机下发、设备算子和数据复制。采集少量有代表性的步骤，正式速度测试应关闭分析采集。", "如何定位推理慢在主机还是设备？", "开启性能分析时测到的速度能直接用作正式成绩吗？"),
("降低精度", "尝试降低浮点精度后要重新检查数值和检索质量。出现质量退化时保留失败记录并回到之前通过检查的配置。", "换用低精度后还需要检查检索结果吗？", "性能变快但向量质量变差该怎么处理？"),
("注意力掩码", "编码器的双向注意力需要屏蔽补齐位置，同时保留有效词元之间的双向关系。不同接口对布尔掩码的含义可能相反。", "更换注意力算子时为什么要检查掩码方向？", "编码器能直接使用生成模型的因果掩码吗？"),
("服务访问", "课堂演示服务默认只监听本机回环地址。远程学员通过 SSH 端口转发访问服务，按接口要求发送文本列表并接收向量。", "本地电脑如何访问服务器上的课堂向量服务？", "演示接口默认监听什么地址？"),
("请求排队", "单进程演示服务依次处理请求。并发请求可能排队，客户端总耗时包含排队和网络时间，不能直接等同于模型计算时间。", "服务有很多请求时为什么客户端延迟会上升？", "客户端延迟为什么不等于设备上的计算耗时？"),
("实验结论", "实验报告写清优化假设、改动、正确性检查、性能对照与结论。没有提速的实验同样可以通过验收，只要测量方法可靠。", "优化没有提速还能通过实验验收吗？", "一份完整的调优报告需要写哪些内容？"),
("技术求助", "遇到问题向助教提供复现命令、配置、完整错误栈和环境摘要。日志中不要包含访问令牌、个人密码或其他认证信息。", "向助教报告实验故障要提供哪些信息？", "提交错误日志前需要去掉什么内容？"),
]


def generate(out):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError("Dataset already exists; use --verify or choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = [], [], []
    for i, (title, text, a, b) in enumerate(FAQ, 1):
        did = f"d{i:03d}"
        corpus.append({"id": did, "text": title + "。" + text})
        for j, query in enumerate((a, b)):
            qid = f"q{(i-1)*2+j+1:03d}"
            queries.append({"id": qid, "text": query})
            qrels.append(qid + "\t" + did)
    smoke_texts = ["你好", "如何预约实验服务器？", "How do I reset my password?", "PyTorch 在线推理，NPU 910B3，2026年。", "空白  与\t标点！？测试", "数字0123456789和符号+-=/", "超长输入截断与语义检索。" * 80, "同一批次中的短句"]
    smoke = [{"id": f"s{i:03d}", "text": t} for i, t in enumerate(smoke_texts, 1)]
    perf = [{"id": f"p{i:04d}", "text": corpus[i % len(corpus)]["text"] * (1, 2, 4, 12)[(i // len(corpus)) % 4]} for i in range(1024)]
    groups = {"corpus": corpus, "queries": queries, "smoke": smoke, "perf_requests": perf}
    for name, values in groups.items():
        (out / (name + ".jsonl")).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
    (out / "qrels.tsv").write_text("query_id\tdoc_id\n" + "\n".join(qrels) + "\n", encoding="utf-8")
    manifest = {"provenance": "original synthetic teaching data; fictional policies; not an official evaluation set",
        "counts": {name: len(values) for name, values in groups.items()},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.suffix in (".jsonl", ".tsv")}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


def verify_data(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Dataset fingerprint mismatch: {name}")
    result = {"verified": True, "counts": manifest["counts"]}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify_data(args.out) if args.verify else generate(args.out)
