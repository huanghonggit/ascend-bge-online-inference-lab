import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from .core import batches, read_jsonl, save_json, sha256_file


def summarize_times(times, count):
    values = np.asarray(times, dtype=np.float64)
    return {"calls": len(values), "texts": count, "sum_ms": float(values.sum()),
            "mean_batch_ms": float(values.mean()), "p50_batch_ms": float(np.percentile(values, 50)),
            "p95_batch_ms": float(np.percentile(values, 95)),
            "serial_processing_texts_per_s": float(count / (values.sum() / 1000))}


@torch.inference_mode()
def warmup(engine, texts, steps=10):
    host, _ = engine.tokenize(texts)
    inputs = engine.to_device(host)
    engine.sync()
    start = time.perf_counter()
    result = engine.run(**inputs)
    engine.sync()
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(steps):
        result = engine.run(**inputs)
    engine.sync()
    del result, inputs
    return first_ms


@torch.inference_mode()
def benchmark(engine, data_path, out_dir, repeat=3, warmup_steps=10):
    out_dir = Path(out_dir)
    if (out_dir / "bench.json").exists():
        raise FileExistsError("bench.json exists; use a new --out directory")
    if repeat < 1 or warmup_steps < 1:
        raise ValueError("repeat and warmup must be positive")
    quality_binding = {"status": "NOT_CHECKED", "sha256": None}
    quality_file = out_dir / "quality.json"
    if quality_file.exists():
        quality = json.loads(quality_file.read_text(encoding="utf-8"))
        if quality["metadata"]["config"] != engine.cfg or quality["metadata"]["model"] != engine.identity:
            raise ValueError("Quality and benchmark must use the same config and model in one output directory")
        if quality["status"] not in ("passed", "reference_created"):
            raise ValueError("Quality has failed or is incomplete; timing is not accepted")
        quality_binding = {"status": quality["status"], "sha256": sha256_file(quality_file)}
    rows = read_jsonl(data_path)
    parts = list(batches(rows, engine.cfg["batch_size"]))
    first_ms = warmup(engine, [r["text"] for r in parts[0]], warmup_steps)
    graphs_before = engine.compiled_graphs
    if engine.device.type == "npu":
        torch.npu.reset_peak_memory_stats(engine.device)
    raw, per_pass, lengths = [], [], []
    for rep in range(repeat):
        model_times, e2e_times = [], []
        temp = []
        for batch_index, part in enumerate(parts):
            host, n = engine.tokenize([r["text"] for r in part])
            tokens = int(host["attention_mask"][:n].sum())
            shape = list(host["input_ids"].shape)
            if rep == 0:
                lengths.extend(host["attention_mask"][:n].sum(-1).tolist())
            inputs = engine.to_device(host)
            engine.sync()  # H2D and tokenization excluded from the model interval.
            start = time.perf_counter()
            result = engine.run(**inputs)
            engine.sync()
            elapsed = (time.perf_counter() - start) * 1000
            model_times.append(elapsed)
            del result, inputs
            temp.append({"repeat": rep, "batch_index": batch_index, "texts": n,
                         "useful_tokens": tokens, "computed_token_slots": shape[0] * shape[1],
                         "shape_b": shape[0], "shape_s": shape[1], "model_ms": elapsed})
        # Same dataset, same order. A separate pass keeps model and application boundaries clear.
        for batch_index, part in enumerate(parts):
            texts = [r["text"] for r in part]
            engine.sync()
            start = time.perf_counter()
            host, n = engine.tokenize(texts)
            inputs = engine.to_device(host)
            result = engine.run(**inputs)[:n].float().cpu().numpy()
            engine.sync()  # CPU result is ready. No HTTP/network or retrieval in this interval.
            elapsed = (time.perf_counter() - start) * 1000
            e2e_times.append(elapsed)
            temp[batch_index]["e2e_ms"] = elapsed
            del result, inputs
        raw.extend(temp)
        per_pass.append({"repeat": rep, "model": summarize_times(model_times, len(rows)),
                         "e2e": summarize_times(e2e_times, len(rows))})
    memory = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    if engine.device.type == "npu":
        memory = {"peak_allocated_bytes": int(torch.npu.max_memory_allocated(engine.device)),
                  "peak_reserved_bytes": int(torch.npu.max_memory_reserved(engine.device))}
    changed = engine.compiled_graphs - graphs_before
    payload = {"metadata": engine.metadata(), "dataset_sha256": sha256_file(data_path),
        "quality_binding": quality_binding, "dataset_texts": len(rows), "repeat": repeat, "warmup_steps": warmup_steps,
        "first_forward_ms": first_ms, "steady_graph_count_delta": changed,
        "eligible_steady_timing": changed == 0,
        "model": summarize_times([r["model_ms"] for r in raw], len(rows)*repeat),
        "e2e": summarize_times([r["e2e_ms"] for r in raw], len(rows)*repeat),
        "per_pass": per_pass, "memory": memory,
        "post_truncation_length": {"min": min(lengths), "max": max(lengths),
           "p50": float(np.percentile(lengths, 50)), "p95": float(np.percentile(lengths, 95))},
        "total_useful_tokens_per_pass": sum(r["useful_tokens"] for r in raw if r["repeat"] == 0),
        "total_computed_token_slots_per_pass": sum(r["computed_token_slots"] for r in raw if r["repeat"] == 0),
        "measurement": "Synchronized serial batch latency; no profiler; throughput is texts/summed processing time. Not concurrent-service capacity. Model interval excludes tokenizer, H2D, D2H. E2E includes tokenizer, H2D, forward/pooling/normalize, D2H; excludes HTTP and retrieval."}
    save_json(out_dir / "bench.json", payload)
    with (out_dir / "latency.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    if changed:
        raise RuntimeError("New graph compiled during timing; timings rejected. Check inputs/shape/guards.")
    return payload


@torch.inference_mode()
def capture_profile(engine, data_path, out_dir, detailed=False):
    if engine.device.type != "npu":
        raise ValueError("NPU profiling requires an NPU; use CPU tests for code validation only")
    import torch_npu
    out_dir = Path(out_dir)
    if (out_dir / "profile_metadata.json").exists():
        raise FileExistsError("Profile output exists; choose a new --out")
    rows = read_jsonl(data_path)
    parts = list(batches(rows, engine.cfg["batch_size"]))
    warmup(engine, [r["text"] for r in parts[0]], 5)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir / "trace")),
        record_shapes=detailed, profile_memory=detailed, with_stack=detailed,
    ) as prof:
        for step in range(5):
            part = parts[step % len(parts)]
            with torch.profiler.record_function("LAB.tokenize"):
                host, n = engine.tokenize([r["text"] for r in part])
            with torch.profiler.record_function("LAB.H2D"):
                inputs = engine.to_device(host)
            with torch.profiler.record_function("LAB.forward_pool_normalize"):
                result = engine.run(**inputs)
            with torch.profiler.record_function("LAB.D2H"):
                _ = result[:n].cpu().numpy()
            engine.sync()
            prof.step()
    payload = {"metadata": engine.metadata(), "schedule": "wait=1,warmup=1,active=3,repeat=1",
               "detailed": detailed, "use": "diagnosis only; use benchmark command for timings"}
    save_json(out_dir / "profile_metadata.json", payload)
    return payload
