import json
from pathlib import Path
import numpy as np
from .core import read_jsonl, save_json, sha256_file


def read_qrels(path, query_ids, corpus_ids):
    rel = {q: set() for q in query_ids}
    corpus_set = set(corpus_ids)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("query_id"):
            continue
        q, d = line.split("\t")
        if q not in rel or d not in corpus_set:
            raise ValueError("qrels references an unknown ID")
        rel[q].add(d)
    if any(not docs for docs in rel.values()):
        raise ValueError("Every query must have at least one relevant document")
    return rel


def retrieval(query, corpus, query_ids, corpus_ids, rel):
    scores = query @ corpus.T
    ranked = np.argsort(-scores, axis=1, kind="stable")
    r1, r5, mrr, top = [], [], [], []
    for i, qid in enumerate(query_ids):
        docs = [corpus_ids[j] for j in ranked[i]]
        relevant = rel[qid]
        r1.append(len(set(docs[:1]) & relevant) / len(relevant))
        r5.append(len(set(docs[:5]) & relevant) / len(relevant))
        hits = [j + 1 for j, doc in enumerate(docs[:10]) if doc in relevant]
        mrr.append(1.0 / min(hits) if hits else 0.0)
        top.append({"query_id": qid, "top5": docs[:5], "relevant": sorted(relevant)})
    return {"recall1": float(np.mean(r1)), "recall5": float(np.mean(r5)),
            "mrr10": float(np.mean(mrr))}, top


def finite_stats(array):
    finite = bool(np.isfinite(array).all())
    return {"finite": finite, "max_norm_error": float(np.abs(np.linalg.norm(array, axis=1) - 1).max()) if finite else None}


def compare_artifacts(candidate_dir, reference_dir, gate):
    candidate_dir, reference_dir = Path(candidate_dir), Path(reference_dir)
    c = json.loads((candidate_dir / "quality.json").read_text(encoding="utf-8"))
    r = json.loads((reference_dir / "quality.json").read_text(encoding="utf-8"))
    # Reject incomparable data, instructions, tokenizer outputs and model snapshots.
    for key in ("ids", "data_hashes", "token_hashes", "semantic_contract"):
        if c[key] != r[key]:
            raise ValueError(f"Reference mismatch: {key}")
    if c["metadata"]["model"] != r["metadata"]["model"]:
        raise ValueError("Model identity mismatch")
    comparisons, checks = {}, {}
    with np.load(candidate_dir / "embeddings.npz", allow_pickle=False) as ca, np.load(reference_dir / "embeddings.npz", allow_pickle=False) as ra:
        for group in ("corpus", "queries", "smoke"):
            a, b = ca[group].astype(np.float64), ra[group].astype(np.float64)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                comparisons[group] = {"finite_and_same_shape": False}
                checks[group] = False
                continue
            cosine = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
            stats = {"finite_and_same_shape": True, "mean_cosine": float(cosine.mean()),
                     "min_cosine": float(cosine.min()), "max_abs_error": float(np.abs(a-b).max()),
                     "max_norm_error": float(np.abs(np.linalg.norm(a, axis=1)-1).max())}
            comparisons[group] = stats
            checks[group] = (stats["mean_cosine"] >= gate["min_mean_cosine"] and
                             stats["min_cosine"] >= gate["min_sample_cosine"] and
                             stats["max_abs_error"] <= gate["max_abs_error"] and
                             stats["max_norm_error"] <= gate["max_norm_error"])
    recall_drop = r["retrieval"]["recall5"] - c["retrieval"]["recall5"]
    mrr_drop = r["retrieval"]["mrr10"] - c["retrieval"]["mrr10"]
    checks["retrieval"] = recall_drop <= gate["max_recall5_drop"] and mrr_drop <= gate["max_mrr10_drop"]
    return {"passed": all(checks.values()), "checks": checks, "vectors": comparisons,
            "recall5_drop": recall_drop, "mrr10_drop": mrr_drop,
            "reference": str(reference_dir), "gate": gate}


def evaluate(engine, data_dir, out_dir, reference=None, gate_path=None):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if (out_dir / "quality.json").exists():
        raise FileExistsError("quality.json exists; use a new --out directory")
    rows = {group: read_jsonl(data_dir / (group + ".jsonl")) for group in ("corpus", "queries", "smoke")}
    ids = {group: [r["id"] for r in batch] for group, batch in rows.items()}
    rel = read_qrels(data_dir / "qrels.tsv", ids["queries"], ids["corpus"])
    arrays, token_hashes = {}, {}
    for group, batch in rows.items():
        arrays[group], token_hashes[group] = engine.encode([r["text"] for r in batch], "query" if group == "queries" else "passage")
    metrics, top = retrieval(arrays["queries"], arrays["corpus"], ids["queries"], ids["corpus"], rel)
    payload = {"metadata": engine.metadata(), "ids": ids,
        "semantic_contract": {"max_length": engine.cfg["max_length"], "query_prefix": engine.cfg["query_prefix"], "pooling": "CLS_FP32_L2", "padding_side": "right"},
        "data_hashes": {name: sha256_file(data_dir / name) for name in ("corpus.jsonl", "queries.jsonl", "smoke.jsonl", "qrels.tsv")},
        "token_hashes": token_hashes, "retrieval": metrics,
        "vector_checks": {name: finite_stats(v) for name, v in arrays.items()},
        "status": "reference_created" if reference is None else "pending_comparison"}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "embeddings.npz", **arrays)
    save_json(out_dir / "retrieval_top5.json", top)
    save_json(out_dir / "quality.json", payload)
    if not all(item["finite"] for item in payload["vector_checks"].values()):
        payload["status"] = "failed_nonfinite"
        save_json(out_dir / "quality.json", payload)
        raise RuntimeError("NaN/Inf detected; candidate is rejected")
    if reference:
        gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
        comparison = compare_artifacts(out_dir, reference, gate)
        payload["comparison"] = comparison
        payload["status"] = "passed" if comparison["passed"] else "failed"
        save_json(out_dir / "quality.json", payload)
        if not comparison["passed"]:
            raise RuntimeError(f"Quality gate failed; inspect {out_dir / 'quality.json'}")
    return payload
