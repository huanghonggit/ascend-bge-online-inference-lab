import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .attention import replace_bert_attention

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
REQUIRED_TRANSFORMERS = "5.3.0"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                             allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    if parent:
        cfg = {**load_config(path.parent / parent), **cfg}
    if cfg["engine"] not in ("eager", "torchair"):
        raise ValueError("engine must be eager or torchair")
    if cfg["attention"] not in ("eager", "fia"):
        raise ValueError("attention must be eager or fia")
    if cfg["dtype"] not in ("fp32", "bf16"):
        raise ValueError("dtype must be fp32 or bf16")
    if cfg["padding"] not in ("max_length", "longest"):
        raise ValueError("unsupported padding strategy")
    if cfg["engine"] == "torchair" and cfg["padding"] != "max_length":
        raise ValueError("Graph lab requires fixed padding=max_length")
    if cfg["internal_format"] and cfg["engine"] != "torchair":
        raise ValueError("Internal weight experiment requires TorchAir")
    if cfg["attention"] == "fia" and cfg["dtype"] != "bf16":
        raise ValueError("FIA lab uses bf16 only")
    if not 2 <= cfg["max_length"] <= 512 or cfg["batch_size"] < 1:
        raise ValueError("Invalid maximum length or batch size")
    return cfg


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Empty dataset or duplicate IDs: {path}")
    validate_texts([r["text"] for r in rows])
    return rows


def validate_texts(texts):
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("texts must be a non-empty list")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Every text must be a non-empty string")


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def package_versions():
    packages = ("torch", "torch-npu", "transformers", "tokenizers", "numpy", "huggingface-hub", "safetensors", "torchair")
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def model_identity(model_dir):
    root = Path(model_dir)
    manifest = root / "model_manifest.json"
    if manifest.exists():
        # Full weight re-hashing is a separate pre-class command, outside timing.
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return {"manifest_sha256": sha256_file(manifest), "repo_id": data["repo_id"],
                "revision": data["revision"], "integrity_check": "run prepare_model.py --verify before class"}
    raise FileNotFoundError(f"Missing {manifest}; run prepare_model.py --model ... --fingerprint")


class EmbeddingModule(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, return_dict=False)[0]
        return F.normalize(states[:, 0, :].float(), p=2, dim=-1)


class Engine:
    def __init__(self, cfg, model_dir):
        if importlib.metadata.version("transformers") != REQUIRED_TRANSFORMERS:
            raise RuntimeError(f"This lab pins transformers=={REQUIRED_TRANSFORMERS}; use an isolated environment")
        self.cfg = cfg
        self.model_dir = str(Path(model_dir).resolve())
        self.identity = model_identity(model_dir)
        if cfg["device"].startswith("npu"):
            import torch_npu  # Register before constructing torch.device("npu:0").
        self.device = torch.device(cfg["device"])
        if self.device.type == "npu":
            torch.npu.set_device(self.device)
            if not torch.npu.is_available():
                raise RuntimeError("NPU unavailable")
            self.device_name = torch.npu.get_device_name(self.device)
        elif self.device.type == "cpu":
            if cfg["engine"] != "eager" or cfg["attention"] != "eager":
                raise ValueError("TorchAir/FIA require a real NPU; there is no CPU fallback")
            self.device_name = platform.processor() or "CPU"
        else:
            raise ValueError("This lab accepts cpu or npu:<index>")
        torch.set_num_threads(cfg["cpu_threads"])
        torch.manual_seed(20261114)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer.padding_side = "right"
        encoder = AutoModel.from_pretrained(model_dir, local_files_only=True,
                   attn_implementation="eager", dtype=torch.float32).eval()
        encoder.config.use_cache = False
        encoder.config.output_attentions = False
        encoder.config.output_hidden_states = False
        # CLS pooling does not use BertPooler; remove its unused dense/tanh work in ALL variants.
        if getattr(encoder, "pooler", None) is not None:
            encoder.pooler = None
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg["dtype"]]
        encoder = encoder.to(device=self.device, dtype=dtype)
        self.replaced_layers = 0
        if cfg["attention"] == "fia":
            if not hasattr(torch_npu, "npu_fused_infer_attention_score"):
                raise RuntimeError("The installed torch_npu lacks FIA")
            self.replaced_layers = replace_bert_attention(encoder)
        if cfg["internal_format"]:
            import torchair
            if not hasattr(torchair, "use_internal_format_weight"):
                raise RuntimeError("TorchAir has no use_internal_format_weight API")
            torchair.use_internal_format_weight(encoder)
        self.module = EmbeddingModule(encoder).eval()
        self.compiled_graphs = 0
        self.run = self.module
        if cfg["engine"] == "torchair":
            import torchair
            from torch import _dynamo
            _dynamo.config.suppress_errors = False
            backend = torchair.get_npu_backend()
            def counting_backend(gm, example_inputs):
                compiled = backend(gm, example_inputs)
                self.compiled_graphs += 1
                return compiled
            self.run = torch.compile(self.module, backend=counting_backend,
                                     dynamic=False, fullgraph=True)
        self.sync()
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.embedding_dim = encoder.config.hidden_size

    def sync(self):
        if self.device.type == "npu":
            torch.npu.synchronize(self.device)

    def metadata(self):
        return {"config": self.cfg, "packages": package_versions(),
                "model": self.identity, "device_name": self.device_name,
                "embedding_dim": self.embedding_dim,
                "replaced_attention_layers": self.replaced_layers,
                "compiled_graphs": self.compiled_graphs,
                "load_and_setup_ms": self.load_ms}

    def tokenize(self, texts, kind="query"):
        validate_texts(texts)
        if kind not in ("query", "passage") or len(texts) > self.cfg["batch_size"]:
            raise ValueError("Unsupported kind or batch too large")
        prefix = self.cfg["query_prefix"] if kind == "query" else ""
        actual = len(texts)
        padded = list(texts) + [texts[-1]] * (self.cfg["batch_size"] - actual)
        result = self.tokenizer([prefix + t for t in padded], return_tensors="pt",
              truncation=True, max_length=self.cfg["max_length"], padding=self.cfg["padding"])
        result = dict(result)
        if "token_type_ids" not in result:
            result["token_type_ids"] = torch.zeros_like(result["input_ids"])
        result = {name: result[name] for name in ("input_ids", "attention_mask", "token_type_ids")}
        if not torch.all(result["attention_mask"].sum(-1) > 0):
            raise ValueError("Fully masked sequences are forbidden")
        return result, actual

    def to_device(self, encoded):
        # Preserve int64 token IDs / masks; only model weights and activations use bf16.
        return {k: v.to(self.device) for k, v in encoded.items()}

    @torch.inference_mode()
    def encode(self, texts, kind="query"):
        validate_texts(texts)
        vectors, token_records = [], []
        for part in batches(list(texts), self.cfg["batch_size"]):
            host, n = self.tokenize(part, kind)
            # Canonical unpadded IDs make correctness checks independent of batch padding.
            for i in range(n):
                length = int(host["attention_mask"][i].sum())
                token_records.append({k: v[i, :length].tolist() for k, v in host.items()})
            device_inputs = self.to_device(host)
            result = self.run(**device_inputs)
            vectors.append(result[:n].float().cpu().numpy())
        self.sync()
        if self.cfg["engine"] == "torchair" and self.compiled_graphs < 1:
            raise RuntimeError("No TorchAir graph was compiled")
        return np.concatenate(vectors), digest_json(token_records)
