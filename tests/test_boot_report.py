"""Boot report (src/boot_report.py) and its job (src/handler.py), on lines from real vLLM 0.32 logs."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import boot_report  # noqa: E402
import handler  # noqa: E402

# From two vLLM 0.32.0rc1.dev348 boots on a B200 (Qwen3.6-35B-A3B, then RecursiveSI/flint).
QWEN_BOOT = """\
(EngineCore pid=2345) INFO 10-22 02:45:01 [core.py:133] Initializing a V1 LLM engine (v0.32.0rc1.dev348+ge7c846767) with config: model='Qwen/Qwen3.6-35B-A3B'
(APIServer pid=1482) INFO 10-22 02:44:54 [scheduler.py:310] Chunked prefill is enabled with max_num_batched_tokens=16384.
(EngineCore pid=2345) INFO 10-22 02:45:34 [default_loader.py:501] Loading weights took 5.47 seconds
(EngineCore pid=2345) INFO 10-22 02:45:36 [model_runner.py:429] Model loading took 64.61 GiB memory and 31.783849 seconds
(EngineCore pid=2345) INFO 10-22 02:45:52 [monitor.py:62] [backbone] torch.compile took 15.09 s in total
(EngineCore pid=2345) INFO 10-22 02:46:14 [gpu_worker.py:763] Available KV cache memory: 94.73 GiB
(EngineCore pid=2345) INFO 10-22 02:46:14 [kv_cache_utils.py:1101] GPU KV cache size: 3,912,727 tokens, Maximum concurrency for 36,864 tokens per request: 106.14x
(EngineCore pid=2345) INFO 10-22 02:46:40 [model_runner.py:1140] Graph capturing finished in 7 secs, took 0.71 GiB
(EngineCore pid=2345) INFO 10-22 02:46:47 [model_runner.py:1140] Graph capturing finished in 1 secs, took 0.55 GiB
(EngineCore pid=2345) INFO 10-22 02:46:51 [core.py:395] init engine (profile, create kv cache, warmup model) took 74.71 s (compilation: 15.09 s)
"""


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(boot_report, "ENABLED", True)
    monkeypatch.setattr(boot_report, "_launches", [])
    monkeypatch.setattr(boot_report, "_relaunches", [])
    monkeypatch.setattr(boot_report, "_healthy_at", None)
    monkeypatch.setattr(boot_report, "_startup_error", False)
    monkeypatch.setattr(boot_report, "_cache_fallback", None)
    monkeypatch.setattr(boot_report, "_gpu", {"name": "NVIDIA L40S", "memory_mib": [46068], "devices": []})
    return boot_report


def boot(br, text=QWEN_BOOT):
    br.note_launch(["vllm", "serve", "--host", "127.0.0.1", "--model", "Qwen/Qwen3.6-35B-A3B"])
    for line in text.splitlines(keepends=True):
        br.observe(line)


def test_parses_vllm_timing_and_memory_lines(enabled):
    boot(enabled)
    enabled.note_healthy()
    r = enabled.report()
    assert r["vllm"] == {
        "vllm_version": "0.32.0rc1.dev348+ge7c846767",
        "max_num_batched_tokens": 16384,
        "weights_load_s": 5.47,
        "model_load_gib": 64.61,
        "model_load_s": 31.783849,
        "torch_compile_s": 15.09,
        "kv_cache_gib": 94.73,
        "kv_cache_tokens": 3912727,
        "max_concurrency_at_tokens": 36864,
        "max_concurrency_x": 106.14,
        "graph_capture_s": 8.0,          # two capture passes, summed
        "graph_capture_gib": pytest.approx(1.26),
        "engine_init_s": 74.71,
    }
    assert r["launches"] == 1 and r["relaunches"] == [] and r["startup_error"] is False
    assert r["launch_to_healthy_s"] is not None and r["launch_to_healthy_s"] >= 0
    assert r["gpu"] == "NVIDIA L40S" and r["argv"][:2] == ["vllm", "serve"]


def test_startup_plan_and_compile_cache_flags(enabled):
    boot(enabled, "INFO Applying persisted startup plan (fingerprint abc): kv_cache_memory_bytes=1 (1.00 GiB)\n"
                  "INFO [backbone] Directly load the compiled graph(s) for compile range (1, 2048) from the cache, took 0.712 s\n")
    assert enabled.report()["flags"] == ["startup_plan_applied", "compile_cache_hit"]


def test_a_relaunch_starts_a_fresh_record(enabled):
    boot(enabled)
    enabled.note_relaunch("oom")
    enabled.note_launch(["vllm", "serve", "--enforce-eager"])
    enabled.observe("INFO init engine (profile, create kv cache, warmup model) took 12.50 s\n")
    r = enabled.report()
    assert r["launches"] == 2 and r["relaunches"] == ["oom"]
    assert r["vllm"] == {"engine_init_s": 12.5}  # nothing carried over from the first launch
    assert r["argv"] == ["vllm", "serve", "--enforce-eager"]


def test_off_by_default_records_nothing(monkeypatch):
    monkeypatch.setattr(boot_report, "ENABLED", False)
    monkeypatch.setattr(boot_report, "_launches", [])
    boot_report.note_launch(["vllm", "serve"])
    boot_report.observe("INFO init engine (profile, create kv cache, warmup model) took 12.50 s\n")
    assert boot_report._launches == []


def run(job):
    async def collect():
        return [x async for x in handler.handler(job)]
    return asyncio.run(collect())


def test_report_job_returns_the_record(enabled, monkeypatch):
    boot(enabled)
    monkeypatch.setattr(handler, "boot_report", enabled)
    monkeypatch.setattr(handler, "startup_error", "X failed inside vLLM's implementation of the model")
    out = run({"input": {"day0_boot_report": True}})
    assert len(out) == 1 and out[0]["day0_boot_report"]["vllm"]["engine_init_s"] == 74.71


def test_report_job_when_off_says_how_to_turn_it_on(monkeypatch):
    monkeypatch.setattr(handler, "boot_report", None)
    monkeypatch.setattr(handler, "startup_error", None)
    out = run({"input": {"day0_boot_report": True}})
    assert out[0]["error"]["message"].startswith("The boot report is off. Set DAY0_BOOT_REPORT=1")


def test_other_jobs_still_get_the_startup_error(monkeypatch):
    monkeypatch.setattr(handler, "boot_report", None)
    monkeypatch.setattr(handler, "startup_error", "cause")
    out = run({"input": {"prompt": "hi"}})
    assert out == [{"error": {"message": "cause", "type": "startup_error", "code": None}}]


def test_gpu_info_reads_nvidia_smi_and_drops_uuids(monkeypatch):
    monkeypatch.setattr(boot_report, "_gpu", None)
    out = {
        ("--query-gpu=name,memory.total", "--format=csv,noheader,nounits"):
            ["NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887"],
        ("-L",): ["GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (UUID: GPU-1234)",
                  "  MIG 2g.48gb      Device  0: (UUID: MIG-5678)"],
    }
    monkeypatch.setattr(boot_report, "_smi", lambda *a: out[a])
    info = boot_report._gpu_info()
    assert info["name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert info["memory_mib"] == [97887]
    assert info["devices"] == ["GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition", "MIG 2g.48gb      Device  0:"]


def test_cache_fallback_is_reported(enabled):
    boot(enabled)
    assert enabled.report()["compile_cache_fallback"] is None
    enabled.note_cache_fallback("not writable")
    r = enabled.report()
    assert r["compile_cache_fallback"] == "not writable" and r["gpu_memory_mib"] == [46068]
