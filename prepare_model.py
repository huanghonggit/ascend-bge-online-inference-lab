"""Freeze a public snapshot and hash all model/tokenizer files. No NPU needed."""
import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = "BAAI/bge-large-zh-v1.5"
REVISION = "79e7739b6ab944e86d6171e44d24c997fc1e0116"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_files(root):
    return sorted(p for p in root.iterdir() if p.is_file() and
                  p.name != "model_manifest.json" and p.suffix in (".json", ".bin", ".safetensors", ".txt"))


def fingerprint(root, repo_id=REPO, revision=REVISION):
    root = Path(root)
    files = model_files(root)
    if not (root / "config.json").exists() or not any(p.suffix in (".bin", ".safetensors") for p in files):
        raise ValueError("Missing config.json or model weights")
    data = {"repo_id": repo_id, "revision": revision,
            "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in files}}
    (root / "model_manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return data


def verify(root):
    root = Path(root)
    data = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    if set(data["files"]) != {p.name for p in model_files(root)}:
        raise RuntimeError("Model file list changed")
    for name, expected in data["files"].items():
        path = root / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Fingerprint mismatch: {name}")
    return {"verified": True, "revision": data["revision"], "file_count": len(data["files"])}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/bge-large-zh-v1.5")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--fingerprint", action="store_true")
    mode.add_argument("--verify", action="store_true")
    p.add_argument("--revision", default=REVISION)
    a = p.parse_args()
    if a.verify:
        result = verify(a.model)
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
            p.error("--revision must be an immutable 40-character commit, not main")
        if a.download:
            from huggingface_hub import snapshot_download
            snapshot_download(REPO, revision=a.revision, local_dir=a.model,
                allow_patterns=["config.json", "pytorch_model.bin", "*.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"])
        result = fingerprint(a.model, revision=a.revision)
    print(json.dumps(result, ensure_ascii=False, indent=2))
