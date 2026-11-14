"""Single-process classroom HTTP demo. Model stays loaded; requests are serial."""
import argparse
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from bge_lab.core import Engine, load_config, validate_texts

MAX_BODY_BYTES = 1024 * 1024


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"error": "Use GET /health or POST /embed"})
                return
            self.reply(200, {"status": "ready", "metadata": engine.metadata(),
                             "request_execution": "single process, serial"})

        def do_POST(self):
            if self.path != "/embed":
                self.reply(404, {"error": "Use POST /embed"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    raise ValueError("Content-Length must be 1..1048576 bytes")
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                texts, kind = payload.get("texts"), payload.get("kind", "query")
                validate_texts(texts)
                if len(texts) > engine.cfg["batch_size"]:
                    raise ValueError("Request exceeds configured batch_size")
                if kind not in ("query", "passage"):
                    raise ValueError("kind must be query or passage")
            except (ValueError, UnicodeDecodeError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            try:
                started = time.perf_counter()
                vectors, _ = engine.encode(texts, kind)
                elapsed = (time.perf_counter() - started) * 1000
                self.reply(200, {"shape": list(vectors.shape), "embeddings": vectors.tolist(),
                                 "processing_ms": elapsed, "kind": kind,
                                 "timing_note": "Server encode time only; excludes JSON/network/queueing"})
            except Exception:
                traceback.print_exc()
                self.reply(500, {"error": "Inference failed; inspect server log"})
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/bge-large-zh-v1.5")
    p.add_argument("--config", default="configs/online_b1.json")
    p.add_argument("--device")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.device:
        cfg["device"] = a.device
    engine = Engine(cfg, a.model)
    engine.encode(["服务预热"], "query")
    engine.encode(["模型已加载"], "passage")
    server = HTTPServer((a.host, a.port), handler_for(engine))
    print(json.dumps({"ready": True, "host": a.host, "port": a.port, "metadata": engine.metadata()},
                     ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
