"""Boot report for configuration sweeps (opt-in: DAY0_BOOT_REPORT=1).

A sweep that compares worker settings needs engine-side start times: the
Serverless job API's delayTime also contains scheduling and image pull, which
vary by tens of seconds between cold starts and drown out what a setting
changes. With DAY0_BOOT_REPORT=1 the worker keeps a small record of its own
boot (when it launched vLLM, when vLLM answered /health, any relaunch) and of
the timing and memory lines vLLM prints while it starts. A job whose input is
{"day0_boot_report": true} returns that record instead of running a request.

Off by default: with the variable unset nothing is parsed, nothing is kept and
the report job answers with an error saying how to turn it on.
"""

import os
import re
import subprocess
import time
from typing import Any, Callable, Optional

ENABLED = os.getenv("DAY0_BOOT_REPORT", "").strip().lower() in {"1", "true", "yes", "on"}
REPORT_VERSION = 1

# Close enough to process start for a boot measured in tens of seconds.
PROCESS_START = time.time()


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _int(text: str) -> int:
    return int(text.replace(",", ""))


# (field, pattern, converters for each group). Matched against every line vLLM
# prints during startup; for a field seen twice the later line wins, except the
# fields in _SUMMED, which add up (vLLM can capture graphs in more than one pass).
_PATTERNS: list[tuple[str, "re.Pattern[str]", tuple[Callable[[str], Any], ...]]] = [
    ("vllm_version", re.compile(r"LLM engine \(v([^)\s]+)\)"), (str,)),
    ("max_num_batched_tokens", re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)"), (_int,)),
    ("weights_load_s", re.compile(r"Loading weights took ([\d.]+) seconds"), (_num,)),
    ("model_load", re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"), (_num, _num)),
    ("torch_compile_s", re.compile(r"torch\.compile took ([\d.]+) s in total"), (_num,)),
    ("graph_capture", re.compile(r"Graph capturing finished in ([\d.]+) secs?, took ([\d.]+) GiB"), (_num, _num)),
    ("kv_cache_gib", re.compile(r"Available KV cache memory: ([\d.]+) GiB"), (_num,)),
    ("kv_cache_tokens", re.compile(r"GPU KV cache size: ([\d,]+) tokens"), (_int,)),
    ("max_concurrency", re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"), (_int, _num)),
    ("engine_init_s", re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"), (_num,)),
]
_SUMMED = {"graph_capture"}
# Markers whose presence is what matters.
_FLAGS: list[tuple[str, "re.Pattern[str]"]] = [
    ("startup_plan_applied", re.compile(r"Applying persisted startup plan")),
    ("startup_plan_not_applied", re.compile(r"Startup plan not applied")),
    ("startup_plan_saved", re.compile(r"Saved startup plan to")),
    ("compile_cache_hit", re.compile(r"Directly load (?:the compiled graph\(s\).*from the cache|AOT compilation from path)")),
]

_launches: list[dict] = []
_healthy_at: Optional[float] = None
_relaunches: list[str] = []
_startup_error: bool = False
_gpu: Optional[str] = None


def note_launch(argv: list[str]) -> None:
    """A new `vllm serve` started; lines seen from now on belong to it."""
    if ENABLED:
        _launches.append({"at": time.time(), "argv": list(argv), "vllm": {}, "flags": []})


def observe(line: str) -> None:
    """Record any timing or memory line from vLLM's output (one line at a time)."""
    if not ENABLED or not _launches:
        return
    current = _launches[-1]
    for field, pattern, convert in _PATTERNS:
        m = pattern.search(line)
        if not m:
            continue
        values = [f(g) for f, g in zip(convert, m.groups())]
        value: Any = values[0] if len(values) == 1 else values
        if field in _SUMMED and field in current["vllm"]:
            value = [a + b for a, b in zip(current["vllm"][field], value)]
        current["vllm"][field] = value
    for flag, pattern in _FLAGS:
        if flag not in current["flags"] and pattern.search(line):
            current["flags"].append(flag)


def note_healthy() -> None:
    global _healthy_at
    if ENABLED:
        _healthy_at = time.time()


def note_relaunch(reason: str) -> None:
    """The worker is relaunching vLLM with changed settings (oom, disk, revision)."""
    if ENABLED:
        _relaunches.append(reason)


def note_startup_error() -> None:
    global _startup_error
    if ENABLED:
        _startup_error = True


def _gpu_name() -> Optional[str]:
    global _gpu
    if _gpu is None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout.strip().splitlines()
            _gpu = ", ".join(sorted(set(out))) if out else ""
        except (OSError, subprocess.SubprocessError):
            _gpu = ""
    return _gpu or None


def report() -> dict:
    """The record for the report job; plain JSON types only."""
    last = _launches[-1] if _launches else None
    vllm = dict(last["vllm"]) if last else {}
    flat: dict[str, Any] = {}
    for key, value in vllm.items():
        if key == "model_load":
            flat["model_load_gib"], flat["model_load_s"] = value
        elif key == "graph_capture":
            flat["graph_capture_s"], flat["graph_capture_gib"] = value
        elif key == "max_concurrency":
            flat["max_concurrency_at_tokens"], flat["max_concurrency_x"] = value
        else:
            flat[key] = value
    launched = last["at"] if last else None
    return {
        "version": REPORT_VERSION,
        "gpu": _gpu_name(),
        "process_start": PROCESS_START,
        "launches": len(_launches),
        "relaunches": list(_relaunches),
        "startup_error": _startup_error,
        "launched_at": launched,
        "healthy_at": _healthy_at,
        "launch_to_healthy_s": round(_healthy_at - launched, 2) if (_healthy_at and launched) else None,
        "process_to_healthy_s": round(_healthy_at - PROCESS_START, 2) if _healthy_at else None,
        "vllm": flat,
        "flags": list(last["flags"]) if last else [],
        "argv": list(last["argv"]) if last else [],
    }
