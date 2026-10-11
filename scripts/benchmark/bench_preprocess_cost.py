"""Measure the per-decision-step cost of the observation preprocessing pipeline.

This script produces the numbers quoted in the response to reviewer comment R2-2
(Section "Experimental Setup" of the paper): the cost of building the observation
inside a simulation worker, relative to the JSBSim integration it accompanies.

Protocol (same style as scripts/benchmark/bench_env_throughput.py):
  * a single worker processes only, driven by one fixed command sequence;
  * median of three runs of 1000 decision steps after a warm-up of 200 steps;
  * all variants are measured in the same process, so only the preprocessing differs.

Usage:
    python scripts/benchmark/bench_preprocess_cost.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402

import jsb_env.jsbgym_m  # noqa: E402,F401  (registers the env ids)
from src.environments.sensor_wrapper import SensorPerturbationWrapper  # noqa: E402

ENV_ID = "F16-TrackingTask-stage2-NoFG-v0"
N_STEPS = 1000
WARMUP = 200
REPEATS = 3


def make_env():
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    return env


def replay(env, commands):
    for a in commands:
        _obs, _r, terminated, truncated, _info = env.step(a)
        if terminated or truncated:
            env.reset()


def median_time(setup, commands):
    times = []
    for _ in range(REPEATS):
        env = make_env()
        setup(env)
        t0 = time.perf_counter()
        replay(env, commands)
        times.append(time.perf_counter() - t0)
        env.close()
    return statistics.median(times)


def main() -> None:
    env = make_env()
    commands = [env.action_space.sample() for _ in range(WARMUP + N_STEPS)]
    task = env.unwrapped.task
    sim = env.unwrapped.sim
    sim_steps = env.unwrapped.sim_steps_per_agent_step
    obs = np.asarray(env.reset(seed=0)[0])

    t_full = median_time(lambda e: None, commands)
    t_nonorm = median_time(
        lambda e: setattr(e.unwrapped.task, "observation_normalization", lambda x: x),
        commands,
    )

    # state assembly (both aircraft) + concatenation + bounded normalisation
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        own = np.array(task.State(*(sim[p] for p in task.state_variables)))
        opp = np.array(task.State(*(env.unwrapped.opponent_sim[p] for p in task.state_variables)))
        task.observation_normalization(np.concatenate([own, opp]))
    t_chain = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        for _ in range(sim_steps):
            sim.run()
    t_integration = (time.perf_counter() - t0) / N_STEPS

    # optional sensor-perturbation wrapper, per observation and per 30-env batch
    wrapper = SensorPerturbationWrapper(
        env, missing_prob=0.05, rel_error_std=0.02, prefixes=["target/", "oppo/"]
    )
    batch = np.tile(obs.astype(np.float64), (30, 1))
    wrapper.observation(batch)
    m = 3000
    t0 = time.perf_counter()
    for _ in range(m):
        wrapper.observation(obs.astype(np.float64))
    t_wrapper_1 = (time.perf_counter() - t0) / m
    t0 = time.perf_counter()
    for _ in range(m):
        wrapper.observation(batch)
    t_wrapper_30 = (time.perf_counter() - t0) / m

    # optional running-statistics normalisation layer (as used by the baseline)
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec_env = VecNormalize(
        DummyVecEnv([make_env]), norm_obs=True, norm_reward=False, clip_obs=10.0, training=True
    )
    vec_env.normalize_obs(batch)
    t0 = time.perf_counter()
    for _ in range(m):
        vec_env.normalize_obs(batch)
    t_vecnorm = (time.perf_counter() - t0) / m

    step = t_full / N_STEPS
    print(f"observation                       : {obs.shape} {obs.dtype}")
    print(f"state variables per aircraft      : {len(task.state_variables)}")
    print(f"integration steps per decision    : {sim_steps}")
    print(f"decision step, full pipeline      : {step*1e6:7.1f} us ({N_STEPS/t_full:7.1f} steps/s)")
    print(f"assembly + normalisation          : {t_chain*1e6:7.1f} us ({t_chain/step*100:5.1f}%)")
    print(f"  normalisation alone             : {(t_full-t_nonorm)/N_STEPS*1e6:7.1f} us "
          f"({(t_full-t_nonorm)/t_full*100:5.1f}%)")
    print(f"six JSBSim integration steps      : {t_integration*1e6:7.1f} us ({t_integration/step*100:5.1f}%)")
    print(f"sensor wrapper, one observation   : {t_wrapper_1*1e6:7.1f} us ({t_wrapper_1/step*100:5.1f}%)")
    print(f"sensor wrapper, 30-env batch      : {t_wrapper_30*1e6:7.1f} us ({t_wrapper_30/(30*step)*100:5.2f}%)")
    print(f"VecNormalize obs layer, 30-env    : {t_vecnorm*1e6:7.1f} us ({t_vecnorm/(30*step)*100:5.2f}%)")
    env.close()


if __name__ == "__main__":
    main()
