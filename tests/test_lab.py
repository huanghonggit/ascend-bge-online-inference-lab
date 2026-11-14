"""CPU checks validate plumbing and mask mathematics, not Ascend operator execution."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from bge_lab.attention import replace_bert_attention, FusedBertSelfAttention
from bge_lab.core import Engine, load_config, read_jsonl, save_json, validate_texts
from bge_lab.quality import compare_artifacts, evaluate, retrieval
from bge_lab.measure import benchmark
from prepare_model import fingerprint, verify
from lab import summarize

ROOT = Path(__file__).resolve().parents[1]


def reference_attention(q, k, v, blocked, heads, scale):
    scores = (q @ k.transpose(-1, -2)) * scale
    if blocked is not None:
        scores = scores.masked_fill(blocked, -float("inf"))
    return scores.softmax(-1) @ v


def create_tiny_model(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vocab = {v: i for i, v in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "cpu", "npu", "为", "这", "个", "句", "子", "生", "成", "表", "示", "以", "用", "于", "检", "索", "相", "关", "文", "章", "：", "学", "习", "模", "型", "如", "何", "推", "理", "迁", "移", "账", "号", "你", "好"])}
    backend = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=512)
    tok.save_pretrained(root)
    torch.manual_seed(14)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=4, intermediate_size=64, max_position_embeddings=512,
                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    cfg._attn_implementation = "eager"
    model = BertModel(cfg).eval()
    model.save_pretrained(root)
    fingerprint(root, repo_id="local/tiny-random-bert-for-tests", revision="0"*40)
    return model


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(14)
        cfg = BertConfig(vocab_size=48, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0)
        cfg._attn_implementation = "eager"
        cls.original = BertModel(cfg).eval()

    def test_full_model_equivalence_with_and_without_padding(self):
        for mask in (torch.ones(2, 7, dtype=torch.long),
                     torch.tensor([[1,1,1,1,1,1,1], [1,1,1,0,0,0,0]])):
            with self.subTest(mask=mask.tolist()):
                model = copy.deepcopy(self.original)
                self.assertEqual(replace_bert_attention(model, reference_attention), 2)
                x = torch.tensor([[2,6,7,8,9,10,3], [2,8,3,0,0,0,0]])
                with torch.inference_mode():
                    expected = self.original(x, attention_mask=mask).last_hidden_state
                    actual = model(x, attention_mask=mask).last_hidden_state
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_future_tokens_are_visible_and_mask_blocks_keys(self):
        original = copy.deepcopy(self.original.encoder.layer[0].attention.self)
        with torch.no_grad():
            for linear in (original.query, original.key, original.value):
                linear.weight.copy_(torch.eye(32))
                linear.bias.zero_()
        adapter = FusedBertSelfAttention(original, reference_attention).eval()
        x = torch.zeros(1, 4, 32)
        x[:, -1] = 1
        with torch.inference_mode():
            full = adapter(x, torch.zeros(1,1,1,4))[0]
            blocked = adapter(x, torch.tensor([[[[0.,0.,0.,-10000.]]]]))[0]
        self.assertGreater(float(full[0,0].norm()), 0.1)
        torch.testing.assert_close(blocked[0,0], torch.zeros(32))

    def test_adapter_rejects_incompatible_modes(self):
        original = copy.deepcopy(self.original.encoder.layer[0].attention.self)
        adapter = FusedBertSelfAttention(original, reference_attention)
        x = torch.randn(1,4,32)
        with self.assertRaisesRegex(ValueError, "inference-only"):
            adapter(x)
        adapter.eval()
        with self.assertRaisesRegex(ValueError, "cross-attention"):
            adapter(x, encoder_hidden_states=x)
        with self.assertRaisesRegex(ValueError, "4-D"):
            adapter(x, attention_mask=torch.ones(1,4,dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "attention probability"):
            adapter(x, output_attentions=True)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="ascend-bge-tests-")
        cls.path = Path(cls.tmp.name)
        create_tiny_model(cls.path / "model")
        cfg = load_config(ROOT / "configs/cpu_reference.json")
        cfg.update(batch_size=4, max_length=32, cpu_threads=2)
        cls.cfg = cfg
        cls.engine = Engine(cfg, cls.path / "model")
        cls.data = ROOT / "data"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_manifest_detects_changes(self):
        self.assertTrue(verify(self.path / "model")["verified"])
        f = self.path / "model" / "extra.txt"
        f.write_text("changed", encoding="utf-8")
        try:
            with self.assertRaisesRegex(RuntimeError, "file list"):
                verify(self.path / "model")
        finally:
            f.unlink()

    def test_tail_padding_order_and_integer_inputs(self):
        texts = ["hello world", "模型推理", "如何学习", "你好", "npu cpu"]
        host, n = self.engine.tokenize(texts[:2])
        self.assertEqual(n, 2)
        self.assertEqual(tuple(host["input_ids"].shape), (4, 32))
        self.assertEqual(host["input_ids"].dtype, torch.int64)
        together, _ = self.engine.encode(texts)
        separate = np.concatenate([self.engine.encode([t])[0] for t in texts])
        np.testing.assert_allclose(together, separate, atol=1e-6, rtol=1e-5)
        self.assertEqual(together.shape, (5,32))
        np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1, atol=2e-6)
        with self.assertRaises(ValueError):
            validate_texts(["  "])

    def test_quality_comparison_and_data_contract(self):
        ref, cand = self.path / "reference", self.path / "candidate"
        evaluate(self.engine, self.data, ref)
        result = evaluate(self.engine, self.data, cand, ref, ROOT / "configs/quality_gate.json")
        self.assertEqual(result["status"], "passed")
        gate = json.loads((ROOT / "configs/quality_gate.json").read_text(encoding="utf-8"))
        result["semantic_contract"]["query_prefix"] = "wrong"
        save_json(cand / "quality.json", result)
        with self.assertRaisesRegex(ValueError, "semantic_contract"):
            compare_artifacts(cand, ref, gate)
        result["semantic_contract"]["query_prefix"] = self.cfg["query_prefix"]
        save_json(cand / "quality.json", result)
        with np.load(cand / "embeddings.npz") as data:
            arrays = {k: data[k] for k in data.files}
        arrays["smoke"][0,0] = np.nan
        np.savez_compressed(cand / "embeddings.npz", **arrays)
        self.assertFalse(compare_artifacts(cand, ref, gate)["passed"])

    def test_benchmark_counts_and_summary(self):
        short = self.path / "perf.jsonl"
        rows = read_jsonl(self.data / "perf_requests.jsonl")[:9]
        short.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)+"\n", encoding="utf-8")
        out = self.path / "bench-runs" / "cpu-test"
        b = benchmark(self.engine, short, out, repeat=2, warmup_steps=1)
        self.assertEqual(b["model"]["calls"], 6)
        self.assertEqual(b["e2e"]["texts"], 18)
        self.assertTrue(b["eligible_steady_timing"])
        self.assertEqual(b["total_computed_token_slots_per_pass"], 3*4*32)
        self.assertGreater(b["first_forward_ms"], 0)
        summary = summarize(out.parent, self.path / "summary.csv")
        self.assertEqual(summary["experiments"], 1)
        with self.assertRaises(FileExistsError):
            benchmark(self.engine, short, out, repeat=1, warmup_steps=1)

    def test_synthetic_dataset_integrity(self):
        manifest = json.loads((self.data / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(read_jsonl(self.data / "corpus.jsonl")), 32)
        self.assertEqual(len(read_jsonl(self.data / "queries.jsonl")), 64)
        self.assertEqual(len(read_jsonl(self.data / "perf_requests.jsonl")), 1024)
        self.assertTrue(manifest)

    def test_no_cpu_fallback_for_fia(self):
        cfg = dict(self.cfg, dtype="bf16", attention="fia")
        with self.assertRaisesRegex(ValueError, "no CPU fallback"):
            Engine(cfg, self.path / "model")

    def test_retrieval_known_answer(self):
        metrics, _ = retrieval(np.eye(3), np.eye(3), ["q1","q2","q3"], ["d1","d2","d3"],
            {"q1":{"d1"},"q2":{"d2"},"q3":{"d3"}})
        self.assertEqual(metrics, {"recall1":1.0,"recall5":1.0,"mrr10":1.0})


    def test_quality_benchmark_config_binding(self):
        out = self.path / "binding-test"
        evaluate(self.engine, self.data, out)
        qp = out / "quality.json"
        quality = json.loads(qp.read_text(encoding="utf-8"))
        quality["metadata"]["config"]["max_length"] += 1
        save_json(qp, quality)
        with self.assertRaisesRegex(ValueError, "same config"):
            benchmark(self.engine, self.data / "perf_requests.jsonl", out, 1, 1)

    def test_http_real_request_and_validation(self):
        import threading
        import urllib.request
        import urllib.error
        from http.server import HTTPServer
        from serve import handler_for
        server = HTTPServer(("127.0.0.1", 0), handler_for(self.engine))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = "http://127.0.0.1:" + str(server.server_port)
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                self.assertEqual(json.load(response)["status"], "ready")
            data = json.dumps({"texts": ["如何学习", "hello world"], "kind": "query"}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url + "/embed", data=data, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.load(response)
                self.assertEqual(result["shape"], [2,32])
                expected, _ = self.engine.encode(["如何学习", "hello world"])
                np.testing.assert_allclose(result["embeddings"], expected, atol=1e-6)
            invalid = urllib.request.Request(url + "/embed", data=b'{"texts":[""]}', headers={"Content-Type":"application/json"})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(invalid, timeout=5)
            self.assertEqual(cm.exception.code, 400)
            cm.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
