"""Environment-sampling throughput benchmark: serial baseline vs proposed parallel architecture.

This benchmark measures *environment sampling throughput only*. It does not
involve PPO, reward normalisation, evaluation callbacks or any other part of the
learning pipeline: a fixed, seeded action stream drives the ego aircraft while
the opponent is driven by a policy checkpoint taken from an opponent pool, which
is exactly the work the environment performs during rollout collection.

Two arms are compared:

  serial    ``scripts/benchmark/self_play_wrapper_legacy.py`` around a single
            ``gym.make`` environment in a single process. The opponent policy is
            coupled into the environment step, so every decision step performs
            one single-sample neural-network forward pass in the same process
            that also integrates the flight dynamics. This is the original
            implementation described in the paper.

  parallel  ``src/environments/NN_vec_env.py`` (the proposed architecture): a
            pure-dynamics environment replicated over ``--workers`` subprocesses,
            with the opponent observations of all sub-environments collected and
            sent to the accelerator as one batch per parallel step.

Both arms use the same task, the same opponent pool, the same device and the
same ego action stream, so the difference between them is architectural.

Throughput is reported in two units, because a decision step and a JSBSim
integration step differ by a factor of ``JSBSIM_DT_HZ / agent_interaction_freq``:

  * decision steps per second   (one ``env.step`` of the agent, 10 Hz)
  * integration steps per second (one JSBSim FDM integration step, 60 Hz)

Example
-------
    python -m scripts.benchmark.bench_env_throughput --mode both \
        --workers 30 --steps 20000 --repeats 3 --device cuda \
        --pool opponent_pool/pool3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

ENV_ID = "F16-TrackingTask-stage1-NoFG-v0"
JSBSIM_DT_HZ = 60          # JSBSim FDM integration frequency
AGENT_FREQ_HZ = 10         # agent interaction frequency
SIM_STEPS_PER_AGENT_STEP = JSBSIM_DT_HZ // AGENT_FREQ_HZ  # == 6

ACTION_SEED = 20240927
# A benign, reproducible ego command stream: small surface deflections with a
# moderate throttle, so that episodes stay in a normal flight regime and the
# measured load is dominated by sampling rather than by repeated crashes.
SURFACE_RANGE = 0.25
THROTTLE_RANGE = (0.4, 0.8)


def cpu_topology() -> str:
    """Describe the CPU as physical cores / logical processors, flagging hybrid parts.

    On Intel hybrid parts (e.g. i9-14900K) the two core classes have different
    maximum frequencies, which matters when interpreting the parallel scaling
    curve, so the peak frequencies per core class are reported as well.
    """
    import glob

    cores = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology"):
        cpu = path.split("/")[-2]
        try:
            with open(os.path.join(path, "physical_package_id")) as f:
                pkg = f.read().strip()
            with open(os.path.join(path, "core_id")) as f:
                core = f.read().strip()
        except OSError:
            continue
        cores.setdefault((pkg, core), cpu)

    by_freq = {}
    for cpu in cores.values():
        try:
            with open(f"/sys/devices/system/cpu/{cpu}/cpufreq/cpuinfo_max_freq") as f:
                mhz = int(f.read().strip()) // 1000
        except OSError:
            mhz = None
        by_freq[mhz] = by_freq.get(mhz, 0) + 1

    # Intel hybrid parts expose two core classes with distinct frequency ceilings.
    perf = sum(n for mhz, n in by_freq.items() if mhz and mhz >= 5000)
    eff = sum(n for mhz, n in by_freq.items() if mhz and mhz < 5000)
    peak_perf = max([mhz for mhz in by_freq if mhz and mhz >= 5000], default=None)
    peak_eff = max([mhz for mhz in by_freq if mhz and mhz < 5000], default=None)

    if perf and eff:
        return (f"{len(cores)} physical cores / {os.cpu_count()} logical processors; "
                f"{perf} performance cores up to {peak_perf} MHz, "
                f"{eff} efficiency cores up to {peak_eff} MHz")
    return f"{len(cores)} physical cores / {os.cpu_count()} logical processors"


def make_action_stream(n_steps: int, n_envs: int, seed: int = ACTION_SEED) -> np.ndarray:
    """Deterministic ego action stream, identical for every arm and repeat."""
    rng = np.random.default_rng(seed)
    acts = rng.uniform(-SURFACE_RANGE, SURFACE_RANGE, size=(n_steps, n_envs, 4))
    acts[..., 3] = rng.uniform(THROTTLE_RANGE[0], THROTTLE_RANGE[1], size=(n_steps, n_envs))
    return acts.astype(np.float64)


# --------------------------------------------------------------------------- #
# Arm A: serial baseline
# --------------------------------------------------------------------------- #
def build_serial_env(pool_root: str, device: str):
    import gymnasium as gym
    import jsb_env.jsbgym_m  # noqa: F401  (registers the env ids)
    from scripts.benchmark.self_play_wrapper_legacy import SelfPlayWrapper

    env = gym.make(ENV_ID)
    env = SelfPlayWrapper(env, pool_roots=pool_root, model_num=0, device=device)
    return env


# --------------------------------------------------------------------------- #
# Arm B: proposed parallel architecture
# --------------------------------------------------------------------------- #
def build_parallel_env(pool_root: str, workers: int, start_method: str):
    import gymnasium as gym
    import jsb_env.jsbgym_m  # noqa: F401
    from src.environments.NN_vec_env import NNVecEnv

    def _make(rank: int):
        def _init():
            env = gym.make(ENV_ID)
            env.reset(seed=rank)
            return env
        return _init

    return NNVecEnv(
        [_make(i) for i in range(workers)],
        start_method=start_method,
        pool_roots=pool_root,
        model_num=0,
    )


def build_serial_subproc_env(pool_root: str, workers: int, device: str, start_method: str):
    """Historical pre-fix configuration.

    At git revision ``b8148e1`` the opponent wrapper was installed as the
    ``wrapper_class`` of ``make_vec_env``, so when ``create_env`` was called with
    its then-default ``vec_env_cls=SubprocVecEnv`` the ``SelfPlayWrapper`` was
    constructed *inside every worker process*. Each of the ``workers``
    subprocesses therefore loaded the opponent policies itself and performed one
    single-sample inference per decision step. This arm reproduces that wiring.
    """
    import gymnasium as gym
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def _make(rank: int):
        def _init():
            import jsb_env.jsbgym_m  # noqa: F401  (register env ids inside the worker)
            from scripts.benchmark.self_play_wrapper_legacy import SelfPlayWrapper
            env = gym.make(ENV_ID)
            env = SelfPlayWrapper(env, pool_roots=pool_root, model_num=0, device=device)
            env.reset(seed=rank)
            return env
        return _init

    return SubprocVecEnv([_make(i) for i in range(workers)], start_method=start_method)


def count_opponent_models(pool_root: str) -> int:
    n = 0
    if not os.path.isdir(pool_root):
        return 0
    for item in sorted(os.listdir(pool_root)):
        if os.path.exists(os.path.join(pool_root, item, "best_model.zip")):
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_serial_once(env, actions: np.ndarray, warmup: int) -> dict:
    """One timed repeat of the serial arm."""
    def drive(n: int):
        n_reset = 0
        for i in range(n):
            a = actions[i, 0]
            _, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                env.reset()
                n_reset += 1
        return n_reset

    env.reset()
    drive(min(warmup, len(actions)))

    t0 = time.perf_counter()
    resets = drive(len(actions))
    elapsed = time.perf_counter() - t0
    return {"steps": int(len(actions)), "elapsed_s": elapsed, "resets": resets}


def run_parallel_once(vec, actions: np.ndarray, warmup: int) -> dict:
    """One timed repeat of the parallel arm."""
    def drive(n: int, offset: int = 0):
        for i in range(n):
            vec.step(actions[offset + i])
        return 0

    vec.reset()
    warmup_acts = make_action_stream(max(warmup, 1), vec.num_envs, seed=ACTION_SEED + 1)
    drive(min(warmup, len(warmup_acts)), offset=0)

    t0 = time.perf_counter()
    drive(len(actions), offset=0)
    elapsed = time.perf_counter() - t0
    return {"steps": int(len(actions)) * vec.num_envs, "elapsed_s": elapsed, "resets": -1}


# --------------------------------------------------------------------------- #
# Hardware / software protocol
# --------------------------------------------------------------------------- #
def protocol(device: str, workers: int, pool_root: str, args) -> dict:
    import torch

    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_id": ENV_ID,
        "jsbsim_dt_hz": JSBSIM_DT_HZ,
        "agent_interaction_freq_hz": AGENT_FREQ_HZ,
        "sim_steps_per_agent_step": SIM_STEPS_PER_AGENT_STEP,
        "device": device,
        "workers": workers,
        "pool_root": pool_root,
        "opponent_models_in_pool": count_opponent_models(pool_root),
        "steps_per_repeat": args.steps,
        "repeats": args.repeats,
        "warmup_steps": args.warmup,
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": None,
        "cpu_topology": None,
        "gpu": None,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        info["cpu_topology"] = cpu_topology()
    except Exception:
        pass
    try:
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def summarise(records: list, workers: int) -> dict:
    rates = [r["steps"] / r["elapsed_s"] for r in records]
    med = statistics.median(rates)
    return {
        "decision_steps_per_s_median": med,
        "decision_steps_per_s_min": min(rates),
        "decision_steps_per_s_max": max(rates),
        "integration_steps_per_s_median": med * SIM_STEPS_PER_AGENT_STEP,
        "per_worker_decision_steps_per_s": med / workers if workers else med,
        "elapsed_s": [round(r["elapsed_s"], 3) for r in records],
        "resets": [r["resets"] for r in records],
        "repeats": len(records),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["serial", "serial_subproc", "parallel", "both", "all"], default="both")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--steps", type=int, default=20000, help="timed decision steps per repeat (per environment)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", default="experiments/20260423_154709_best_train_mlp/opponent_pool",
                    help="opponent pool directory (default: the pool used by the BDC-MLP run reported in the paper)")
    ap.add_argument("--start-method", default="fork")
    ap.add_argument("--out", default=None, help="output directory (default experiments/bench_env_throughput/<ts>)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        "experiments", "bench_env_throughput", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "protocol": protocol(args.device, args.workers, args.pool, args),
        "arms": {},
        "summary": {},
    }
    p = result["protocol"]
    print("=" * 78)
    print("Environment sampling throughput benchmark")
    print("=" * 78)
    for k in ("timestamp", "env_id", "jsbsim_dt_hz", "agent_interaction_freq_hz",
              "sim_steps_per_agent_step", "device", "workers", "pool_root",
              "opponent_models_in_pool", "steps_per_repeat", "repeats", "warmup_steps",
              "torch_num_threads", "torch_version", "cpu_model", "cpu_topology", "gpu"):
        print(f"  {k:34s}: {p.get(k)}")
    print("-" * 78, flush=True)

    # ---------------- serial arm ----------------
    if args.mode in ("serial", "both"):
        print("\n[serial] building environment (opponent policy coupled into the step loop)...", flush=True)
        t0 = time.perf_counter()
        env = build_serial_env(args.pool, args.device)
        build_s = time.perf_counter() - t0
        acts = make_action_stream(args.steps + args.warmup, 1)
        print(f"[serial] ready in {build_s:.1f} s, models loaded = {len(env.opponent_models)}", flush=True)

        recs = []
        for r in range(args.repeats):
            rec = run_serial_once(env, acts[:args.steps], args.warmup)
            recs.append(rec)
            print(f"[serial] repeat {r+1}/{args.repeats}: {rec['steps']} steps in "
                  f"{rec['elapsed_s']:.2f} s -> {rec['steps']/rec['elapsed_s']:.1f} decision steps/s "
                  f"({rec['resets']} episode resets)", flush=True)
        env.close()
        result["arms"]["serial"] = {"build_s": build_s, "records": recs,
                                    "summary": summarise(recs, 1)}

    # ---------------- parallel arm ----------------
    if args.mode in ("parallel", "both", "all"):
        print(f"\n[parallel] building {args.workers} subprocess environments...", flush=True)
        t0 = time.perf_counter()
        vec = build_parallel_env(args.pool, args.workers, args.start_method)
        build_s = time.perf_counter() - t0
        acts = make_action_stream(args.steps + args.warmup, args.workers)
        print(f"[parallel] ready in {build_s:.1f} s, models loaded = {len(vec.opponent_models)}", flush=True)

        recs = []
        for r in range(args.repeats):
            rec = run_parallel_once(vec, acts[:args.steps], args.warmup)
            recs.append(rec)
            print(f"[parallel] repeat {r+1}/{args.repeats}: {rec['steps']} steps in "
                  f"{rec['elapsed_s']:.2f} s -> {rec['steps']/rec['elapsed_s']:.1f} decision steps/s", flush=True)
        vec.close()
        result["arms"]["parallel"] = {"build_s": build_s, "records": recs,
                                      "summary": summarise(recs, args.workers)}

    # ---------------- historical multi-process serial arm ----------------
    if args.mode in ("serial_subproc", "all"):
        print(f"\n[serial_subproc] building {args.workers} subprocess environments, each with "
              f"its own opponent policy instance (pre-fix wiring)...", flush=True)
        t0 = time.perf_counter()
        vec = build_serial_subproc_env(args.pool, args.workers, args.device, args.start_method)
        build_s = time.perf_counter() - t0
        acts = make_action_stream(args.steps + args.warmup, args.workers)
        print(f"[serial_subproc] ready in {build_s:.1f} s", flush=True)

        recs = []
        for r in range(args.repeats):
            rec = run_parallel_once(vec, acts[:args.steps], args.warmup)
            recs.append(rec)
            print(f"[serial_subproc] repeat {r+1}/{args.repeats}: {rec['steps']} steps in "
                  f"{rec['elapsed_s']:.2f} s -> {rec['steps']/rec['elapsed_s']:.1f} decision steps/s", flush=True)
        vec.close()
        result["arms"]["serial_subproc"] = {"build_s": build_s, "records": recs,
                                            "summary": summarise(recs, args.workers)}

    # ---------------- summary ----------------
    arms = result["arms"]
    if "serial" in arms and "parallel" in arms:
        s = arms["serial"]["summary"]["decision_steps_per_s_median"]
        q = arms["parallel"]["summary"]["decision_steps_per_s_median"]
        result["summary"] = {
            "serial_decision_steps_per_s": s,
            "parallel_decision_steps_per_s": q,
            "speedup_decision_steps": q / s if s else None,
            "serial_integration_steps_per_s": s * SIM_STEPS_PER_AGENT_STEP,
            "parallel_integration_steps_per_s": q * SIM_STEPS_PER_AGENT_STEP,
            "speedup_integration_steps": q / s if s else None,
        }
        if "serial_subproc" in arms:
            ss = arms["serial_subproc"]["summary"]["decision_steps_per_s_median"]
            result["summary"]["serial_subproc_decision_steps_per_s"] = ss
            result["summary"]["speedup_vs_serial_subproc"] = q / ss if ss else None

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for arm in ("serial", "serial_subproc", "parallel"):
        if arm in arms:
            sm = arms[arm]["summary"]
            print(f"  {arm:14s}  {sm['decision_steps_per_s_median']:10.1f} decision steps/s   "
                  f"{sm['integration_steps_per_s_median']:12.1f} integration steps/s")
    if result["summary"]:
        sp = result["summary"]
        print(f"\n  speed-up parallel vs serial (1 process)        : "
              f"{sp['speedup_decision_steps']:.1f}x")
        if "speedup_vs_serial_subproc" in sp:
            print(f"  speed-up parallel vs serial (subproc, pre-fix) : "
                  f"{sp['speedup_vs_serial_subproc']:.1f}x")
        print(f"  (identical in both units: both arms use {SIM_STEPS_PER_AGENT_STEP} "
              f"integration steps per decision step)")
    print(f"\n  results -> {os.path.join(out_dir, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
