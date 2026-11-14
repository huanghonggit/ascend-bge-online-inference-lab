"""Verify a serial B=1 HTTP service; always save results and reap its own process group."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

import psutil


def pct(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    i = int(pos)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--external-stop-file", type=Path)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Run the server-side check in the Linux lab container")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.external_stop_file and args.external_stop_file.exists():
        parser.error("External stop file already exists; choose a new path")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))

    def request(method, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}" + path, data=data,
                                     method=method, headers={"Content-Type": "application/json; charset=utf-8"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                status, body = response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status, body = exc.code, json.loads(exc.read().decode("utf-8"))
        return {"status": status, "body": body, "roundtrip_ms": (time.perf_counter()-started)*1000}

    def save(name, result):
        (out/name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def group_members(pgid):
        members = []
        for proc in psutil.process_iter(["pid", "status", "cmdline"]):
            try:
                if os.getpgid(proc.pid) == pgid and proc.status() != psutil.STATUS_ZOMBIE:
                    members.append({"pid": proc.pid, "status": proc.status(), "argv": proc.cmdline()})
            except (ProcessLookupError, PermissionError, psutil.Error):
                continue
        return members

    command = [sys.executable, "serve.py", "--config", args.config, "--host", "127.0.0.1", "--port", str(args.port)]
    result = {"command": command, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "requests": [], "functional_passed": False, "passed": False}
    with (out/"service.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pgid = process.pid
        result["process_group_id"] = pgid
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Service exited; inspect service.log")
                try:
                    health = request("GET", "/health")
                    if health["status"] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.5)
            else:
                raise TimeoutError("Service readiness timeout")
            result["health"] = health
            for i in range(30):
                payload = {"texts": ["如何预约实验服务器？" if i % 2 == 0 else "实验完成后，请将结果保存在个人工作目录。"],
                           "kind": "query" if i % 2 == 0 else "passage"}
                response = request("POST", "/embed", payload)
                body = response["body"]
                assert response["status"] == 200 and body["shape"] == [1, 1024], response
                vector = body["embeddings"][0]
                norm = math.sqrt(sum(x*x for x in vector))
                assert all(math.isfinite(x) for x in vector) and abs(norm-1) < 0.0002
                result["requests"].append({"index": i, "kind": payload["kind"], "status": response["status"],
                                           "norm": norm, "processing_ms": body["processing_ms"],
                                           "roundtrip_ms": response["roundtrip_ms"]})
                if i == 0:
                    save("sample_response.json", {"request": payload, "response": response})
            invalid = [{"texts": []}, {"texts": ["甲", "乙"]}, {"texts": ["正常输入"], "kind": "invalid"},
                       ["non-object"], {"texts": ["   "]}]
            result["invalid_requests"] = [request("POST", "/embed", value) for value in invalid]
            assert all(response["status"] == 400 for response in result["invalid_requests"])
            result["health_after"] = request("GET", "/health")
            assert result["health_after"]["status"] == 200
            for field in ("processing_ms", "roundtrip_ms"):
                values = [row[field] for row in result["requests"]]
                result[field] = {"count": len(values), "p50": pct(values, .5), "p95": pct(values, .95),
                                 "mean": statistics.mean(values)}
            result["functional_passed"] = True
            save("http_results.before_shutdown.json", result)
            if args.external_stop_file:
                print("READY_FOR_EXTERNAL_PROBE", flush=True)
                deadline = time.monotonic() + 300
                while not args.external_stop_file.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Service exited while waiting for external probe")
                    time.sleep(0.5)
                result["external_stop_seen"] = args.external_stop_file.exists()
                if not result["external_stop_seen"]:
                    raise TimeoutError("External probe completion flag not received in 300 seconds")
        except BaseException as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            traceback.print_exc()
        finally:
            result["shutdown_signals"] = []
            for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 8), (signal.SIGKILL, 5)):
                if process.poll() is not None and not group_members(pgid):
                    break
                result["shutdown_signals"].append(sig.name)
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    process.poll()
                    if process.returncode is not None and not group_members(pgid):
                        break
                    time.sleep(0.2)
            process.poll()
            result["service_exit_code"] = process.returncode
            result["remaining_processes"] = group_members(pgid)
            result["child_stopped"] = process.returncode is not None and not result["remaining_processes"]
            result["graceful_shutdown"] = result["service_exit_code"] == 0 and result["shutdown_signals"] == ["SIGINT"] and result["child_stopped"]
            result["passed"] = result["functional_passed"] and "error" not in result and result["child_stopped"]
            result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save("http_results.json", result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("requests", "health", "health_after", "invalid_requests")}, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
