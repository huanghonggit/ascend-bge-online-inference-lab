"""CLI: python lab.py --help. Run from the lab directory on the NPU server."""
import argparse
import csv
import json
import subprocess
from pathlib import Path


def doctor(args):
    import torch
    from bge_lab.core import package_versions, save_json
    result = {"packages": package_versions(), "requested_device": args.device}
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        result["npu"] = {"available": torch.npu.is_available(), "visible_devices": torch.npu.device_count(), "name": torch.npu.get_device_name(args.device)}
        try:
            smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
            result["npu_smi"] = smi.stdout + smi.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["npu_smi"] = str(exc)
        x = torch.randn(32, 32, device=args.device, dtype=torch.bfloat16)
        y = x @ x
        torch.npu.synchronize()
        result["matmul_finite"] = bool(torch.isfinite(y).all().cpu())
        if args.with_optimizers:
            from bge_lab.attention import npu_fia
            q = torch.randn(2, 16, 32, 64, device=args.device, dtype=torch.bfloat16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            blocked = torch.zeros(2, 1, 32, 32, device=args.device, dtype=torch.bool)
            blocked[0, :, :, 20:] = True
            with torch.inference_mode():
                reference = ((q.float() @ k.float().transpose(-1, -2)) / 8.0).masked_fill(blocked, -float("inf")).softmax(-1) @ v.float()
                candidate = npu_fia(q, k, v, blocked, 16, 1/8)
                maxerr = float((candidate.float()-reference).abs().max().cpu())
                result["fia_smoke_max_abs_error"] = maxerr
                result["fia_smoke_passed"] = bool(torch.isfinite(candidate).all().cpu()) and maxerr < 0.05
                if not result["fia_smoke_passed"]:
                    save_json(args.out, result)
                    raise RuntimeError("FIA smoke failed; inspect error and software compatibility")
                import torchair
                result["torchair_module"] = {"file": torchair.__file__, "version": getattr(torchair, "__version__", None)}
                layer = torch.nn.Linear(64, 64).to(device=args.device, dtype=torch.bfloat16).eval()
                expected = layer(q)
                compiled = torch.compile(layer, backend=torchair.get_npu_backend(), dynamic=False, fullgraph=True)
                actual = compiled(q)
                torch.npu.synchronize()
                result["torchair_smoke_max_abs_error"] = float((actual.float()-expected.float()).abs().max().cpu())
                result["torchair_smoke_passed"] = torch.allclose(actual, expected, atol=0.05, rtol=0.02)
                if not result["torchair_smoke_passed"]:
                    save_json(args.out, result)
                    raise RuntimeError("TorchAir smoke failed")
    elif args.device == "cpu":
        result["cpu_matmul_finite"] = bool(torch.isfinite(torch.ones(2, 2) @ torch.ones(2, 2)).all())
    else:
        raise ValueError("Use cpu or npu:<index>")
    save_json(args.out, result)
    return result


def summarize(root, out):
    rows = []
    for p in sorted(Path(root).glob("*/bench.json")):
        b = json.loads(p.read_text(encoding="utf-8"))
        qp = p.parent / "quality.json"
        q = json.loads(qp.read_text(encoding="utf-8")) if qp.exists() else {}
        cfg = b["metadata"]["config"]
        binding = b.get("quality_binding", {"status": "NOT_CHECKED", "sha256": None})
        from bge_lab.core import sha256_file
        quality_bound = qp.exists() and binding["sha256"] == sha256_file(qp)
        quality_status = binding["status"] if quality_bound else "NOT_CHECKED_OR_CHANGED"
        rows.append({"experiment": cfg["name"], "device": cfg["device"], "dtype": cfg["dtype"],
             "engine": cfg["engine"], "attention": cfg["attention"], "internal_format": cfg["internal_format"],
             "batch": cfg["batch_size"], "max_length": cfg["max_length"], "padding": cfg["padding"],
             "dataset_sha256": b["dataset_sha256"], "quality_status": quality_status,
             "eligible_steady_timing": b["eligible_steady_timing"],
             "model_p50_ms": b["model"]["p50_batch_ms"], "model_p95_ms": b["model"]["p95_batch_ms"],
             "e2e_p50_ms": b["e2e"]["p50_batch_ms"], "e2e_p95_ms": b["e2e"]["p95_batch_ms"],
             "e2e_processing_texts_s": b["e2e"]["serial_processing_texts_per_s"],
             "first_forward_ms": b["first_forward_ms"],
             "peak_allocated_MiB": None if b["memory"]["peak_allocated_bytes"] is None else b["memory"]["peak_allocated_bytes"] / 1024**2,
             "compiled_graphs": b["metadata"]["compiled_graphs"],
             "recall5": q.get("retrieval", {}).get("recall5"), "mrr10": q.get("retrieval", {}).get("mrr10")})
    if not rows:
        raise ValueError("No */bench.json files found")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"experiments": len(rows), "output": str(out), "sorting": "by directory name, not performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--device", default="npu:0")
    d.add_argument("--with-optimizers", action="store_true")
    d.add_argument("--out", default="runs/environment.json")
    for name in ("infer", "quality", "bench", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/npu_fp32.json")
        p.add_argument("--model", default="models/bge-large-zh-v1.5")
        p.add_argument("--device", default=None)
        if name == "infer":
            p.add_argument("--texts", nargs="+", required=True)
            p.add_argument("--kind", choices=("query", "passage"), default="query")
            p.add_argument("--out", default="runs/infer.json")
        else:
            p.add_argument("--data", default="data" if name == "quality" else "data/perf_requests.jsonl")
            p.add_argument("--out", required=True)
        if name == "quality":
            p.add_argument("--reference")
            p.add_argument("--gate", default="configs/quality_gate.json")
        if name == "bench":
            p.add_argument("--repeat", type=int, default=3)
            p.add_argument("--warmup", type=int, default=10)
        if name == "profile":
            p.add_argument("--detailed", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--candidate", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--gate", default="configs/quality_gate.json")
    c.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--runs", required=True)
    s.add_argument("--out", default="runs/summary.csv")
    a = parser.parse_args()
    if a.command == "doctor":
        result = doctor(a)
    elif a.command == "summarize":
        result = summarize(a.runs, a.out)
    elif a.command == "compare":
        from bge_lab.core import save_json
        from bge_lab.quality import compare_artifacts
        result = compare_artifacts(a.candidate, a.reference, json.loads(Path(a.gate).read_text(encoding="utf-8")))
        save_json(a.out, result)
        if not result["passed"]:
            raise RuntimeError("Quality comparison failed")
    else:
        from bge_lab.core import Engine, load_config, save_json
        cfg = load_config(a.config)
        if a.device:
            cfg["device"] = a.device
        engine = Engine(cfg, a.model)
        if a.command == "infer":
            vectors, _ = engine.encode(a.texts, a.kind)
            result = {"shape": list(vectors.shape), "vectors": vectors.tolist(), "metadata": engine.metadata()}
            save_json(a.out, result)
            result = {"shape": list(vectors.shape), "output": a.out, "metadata": engine.metadata()}
        elif a.command == "quality":
            from bge_lab.quality import evaluate
            result = evaluate(engine, a.data, a.out, a.reference, a.gate)
        elif a.command == "bench":
            from bge_lab.measure import benchmark
            result = benchmark(engine, a.data, a.out, a.repeat, a.warmup)
        else:
            from bge_lab.measure import capture_profile
            result = capture_profile(engine, a.data, a.out, a.detailed)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
