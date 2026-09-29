"""Implementation checks required by section 8 (hand-built states/tapes,
no statistical 'expected winner' checks)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm import params as P                                   # noqa: E402
from bm.policies import (ControllerC, FixedP, OracleD, OutputThreshold,   # noqa: E402
                         OutputTrend, Policy, WorkOnly, base_strategies, tuning_grid)
from bm.runner import run_episode, run_world                 # noqa: E402
from bm.tape import Tape, dev_seeds, make_tape               # noqa: E402
from bm.world import (OBSERVATION_FIELDS, FEEDBACK_FIELDS, Feedback, Observation,  # noqa: E402
                      World, capacity, sensor_vector)

SEED = 31000000
WORK, SERVICE = P.WORK, P.SERVICE


def fresh(mode="dynamic", seed=SEED):
    return World(make_tape(seed, mode))


# ---------------------------------------------------------------- admissibility
@pytest.mark.parametrize("R,k,ok", [
    (2.75, 5, True),
    (np.nextafter(2.75, 0.0), 5, False),
    (2.70, 5, False),
    (2.75, 4, False),
    (10.0, 4, False),
    (10.0, 5, True),
])
def test_service_admissibility(R, k, ok):
    w = fresh()
    w.R, w.k = R, k
    w.begin_step()
    assert w.observation().service_allowed is ok
    fb = w.apply(SERVICE)
    assert fb.action == (SERVICE if ok else WORK)


# ---------------------------------------------------------------- SERVICE physics
def test_service_physics():
    w = fresh()
    w.R, w.k, w.D, w.S, w.E, w.n_o, w.G = 10.0, 7, 0.9, 0.3, 0.8, 5, 12.0
    w.begin_step()
    fb = w.apply(SERVICE)
    assert fb.action == SERVICE and fb.u is None
    assert w.D == pytest.approx(0.16 * 0.9, abs=1e-15)
    assert w.S == 0.3
    assert w.R == pytest.approx(10.0 - 2.55, abs=1e-12)
    assert w.G == 12.0
    assert (w.k, w.E, w.n_o, w.n_s) == (0, 1.0, 0, 1)


# ---------------------------------------------------------------- WORK physics
def test_work_uses_pre_action_capacity_and_updated_D_for_damage():
    tape = make_tape(SEED, "dynamic")
    w = World(tape)
    w.D, w.S, w.R, w.E, w.n_o = 0.95, 0.47, 20.0, 0.9, 3
    D0, S0 = w.D, w.S
    w.begin_step()
    j = 0
    a, m, h, eps = tape.a[j], tape.m[j], tape.h[j], tape.eps_prod[j]
    c_pre = capacity(D0, S0)
    assert c_pre < 0.995 and w.pre_action_symptom == 1
    fb = w.apply(WORK)
    g = max(0.0, 1.42 * a * c_pre + eps)
    memory = min(0.95, max(0.86, 0.91 + 0.018 * (m - 1)))
    D1 = min(1.35, max(0.0, memory * D0 + (0.025 + 0.075 * a) * m + h))
    S1 = S0 + 0.026 * max(0, D1 - 0.58) + 0.115 * max(0, D1 - 0.82) ** 2
    assert w.G == pytest.approx(g, abs=1e-14)
    assert w.D == pytest.approx(D1, abs=1e-14)
    assert w.S == pytest.approx(S1, abs=1e-14)
    assert w.R == pytest.approx(min(52.0, 20.0 - 0.30 + g - 0.12), abs=1e-12)
    u = min(1.15, max(0.0, g / (1.42 * max(0.18, a))))
    assert fb.u == pytest.approx(u, abs=1e-14)
    assert w.E == pytest.approx(0.7 * 0.9 + 0.3 * u, abs=1e-14)
    assert w.n_o == 4 and w.k == P.K0 + 1
    # discriminating: capacity from post-action state would give a different output,
    # and damage from pre-action D would give a different S
    c_post = capacity(D1, S1)
    assert abs(max(0.0, 1.42 * a * c_post + eps) - g) > 1e-6
    S_wrong = S0 + 0.026 * max(0, D0 - 0.58) + 0.115 * max(0, D0 - 0.82) ** 2
    assert abs(S_wrong - S1) > 1e-6


def test_reserve_cap():
    w = fresh()
    w.R = 51.9
    w.begin_step()
    w.apply(WORK)
    assert w.R <= 52.0


# ---------------------------------------------------------------- death
def test_no_gain_after_death_and_metrics_frozen():
    tape = make_tape(SEED, "dynamic")
    w = World(tape)
    w.D, w.S = 1.2, 0.555
    res = run_episode(tape, WorkOnly(), world=w)
    assert res["died"] == 1 and res["survived"] == 0
    ds = res["death_step"]
    assert ds == 1 and res["steps_alive"] == 1
    assert res["n_work"] == 1 and w.t == 1
    assert res["G"] == w.G and res["final_S"] >= 0.56


def test_death_by_reserve():
    tape = make_tape(SEED, "dynamic")
    w = World(tape)
    w.R, w.D = 0.05, 0.1
    # zero availability for step 1 would guarantee death; instead just check the rule
    w.begin_step()
    w.apply(WORK)
    assert (not w.alive) == (w.R <= 0.0)


def test_death_on_last_step_is_distinguishable():
    tape = make_tape(SEED, "static")
    w = World(tape)
    w.t = 999
    w.D, w.S = 1.2, 0.5595
    res = run_episode(tape, WorkOnly(), world=w)
    assert w.t == 1000
    assert res["died"] == 1 and res["survived"] == 0
    assert res["death_step"] == 1000 and res["steps_alive"] == 1000
    assert res["G"] > 0.0       # gain on the death step is counted

    w2 = World(tape)
    w2.t = 999
    w2.D, w2.S = 0.1, 0.0
    res2 = run_episode(tape, WorkOnly(), world=w2)
    assert res2["survived"] == 1 and res2["died"] == 0 and math.isnan(res2["death_step"])
    assert res2["steps_alive"] == 1000


def test_after_death_world_refuses_steps():
    w = fresh()
    w.D, w.S = 1.2, 0.559
    w.begin_step()
    w.apply(WORK)
    assert not w.alive
    with pytest.raises(RuntimeError):
        w.begin_step()


# ---------------------------------------------------------------- fixed schedules
@pytest.mark.parametrize("p", P.FIXED_PERIODS)
def test_fixed_first_service_step1_then_p_work(p):
    tape = make_tape(SEED + 7919, "static")
    log = []
    res = run_episode(tape, FixedP(p), log_rows=log)
    actions = [r["action"] for r in log]
    assert actions[0] == SERVICE
    svc = [i for i, a in enumerate(actions) if a == SERVICE]
    if res["n_denied_service_requests"] == 0:
        gaps = np.diff(svc) - 1
        assert np.all(gaps == p)


# ---------------------------------------------------------------- controller C
def _obs(t, x, E=1.0, n_o=0, k=10, R=20.0, allowed=True):
    x = np.asarray(x, dtype=float)
    return Observation(t=t, x=x, R=R, k=k, service_allowed=allowed, E=E, n_o=n_o)


def test_C_derivative_zero_after_service_and_first_step():
    c = ControllerC("C")
    c.reset()
    x1 = np.linspace(-1, 1, 8)
    x2 = x1 * 0.5 + 0.1
    z1 = c.features(x1)
    assert np.all(z1[8:] == 0.0)
    z2 = c.features(x2)
    assert np.allclose(z2[8:], 4.0 * (x2 - x1))
    # through decide/feedback: SERVICE resets the previous observation
    c.reset()
    c.decide(_obs(1, x1))
    c.feedback(Feedback(action=SERVICE, u=None, E=1.0, n_o=0))
    c.decide(_obs(2, x2))
    z_next = c.queue[-1][1]
    assert np.all(z_next[8:] == 0.0)


def test_C_level_zeroes_derivative():
    c = ControllerC("C_level")
    c.reset()
    c.features(np.ones(8))
    z = c.features(-np.ones(8))
    assert np.all(z[8:] == 0.0) and np.all(z[:8] == -1.0)


def test_C_decision_rules():
    c = ControllerC("C")
    c.reset()
    assert c.decide(_obs(1, np.zeros(8), E=0.96, n_o=1)) == SERVICE   # reactive trigger
    c.reset()
    assert c.decide(_obs(1, np.zeros(8), E=0.96, n_o=0)) == WORK      # needs n_o > 0
    c.reset()
    c.w[0] = 4.0                                                       # p ~ 0.98
    c.n = 17
    assert c.decide(_obs(1, np.zeros(8))) == WORK                      # n < 18 -> risk 0
    c.n = 18
    assert c.decide(_obs(2, np.zeros(8))) == SERVICE


def _work_fb():
    return Feedback(action=WORK, u=1.0, E=1.0, n_o=1)


def test_C_queue_exact_t_minus_4_accepted():
    c = ControllerC("C")
    c.reset()
    for t in range(1, 5):
        c.decide(_obs(t, np.full(8, 0.01 * t)))
        c.feedback(_work_fb(), label=0)
    assert c.n == 0
    c.decide(_obs(5, np.zeros(8)))
    w_before = c.w.copy()
    c.feedback(_work_fb(), label=1)
    assert c.n == 1 and c.n_accepted == 1
    assert not np.allclose(w_before, c.w)
    assert c.queue[0][0] == 2


def test_C_queue_example_crossing_service_is_discarded():
    c = ControllerC("C")
    c.reset()
    c.decide(_obs(1, np.zeros(8)))
    c.feedback(_work_fb(), label=0)
    c.decide(_obs(2, np.zeros(8)))
    c.feedback(Feedback(action=SERVICE, u=None, E=1.0, n_o=0))   # no training on SERVICE
    for t in (3, 4, 5, 6):
        c.decide(_obs(t, np.zeros(8)))
        c.feedback(_work_fb(), label=1)
    # t=5 pops entry 1 (crossed SERVICE), t=6 pops entry 2 (the SERVICE step itself)
    assert c.n == 0 and c.n_discarded == 2
    c.decide(_obs(7, np.zeros(8)))
    c.feedback(_work_fb(), label=1)          # entry 3: after the SERVICE, exact t-4
    assert c.n == 1


def test_C_queue_stale_entry_is_discarded_not_caught_up():
    c = ControllerC("C")
    c.reset()
    # a SERVICE step does not pop, so an entry can become older than t-4
    c.queue.append((1, np.zeros(16), 0))
    c.queue.append((2, np.zeros(16), 0))
    c._t = 6
    c.feedback(_work_fb(), label=1)
    assert c.n == 1 and c.n_discarded == 1   # entry 1 stale (discarded), entry 2 exact
    assert len(c.queue) == 0


def test_C_training_update_formula():
    c = ControllerC("C")
    c.reset()
    c.w = np.linspace(-0.5, 0.5, 17)
    z = np.linspace(-1, 1, 16)
    w0 = c.w.copy()
    c._train(z, 1)
    phi = np.concatenate(([1.0], z))
    p = 1 / (1 + np.exp(-np.clip(phi @ w0, -12, 12)))
    reg = w0.copy(); reg[0] = 0.0
    grad = 2.2 * (1 - p) * phi - 0.0008 * reg
    expect = np.clip(w0 + 0.13 * grad / (0.8 + phi @ phi), -4, 4)
    assert np.allclose(c.w, expect, atol=1e-15)


def test_C_scrambled_uses_permuted_vector_and_its_difference():
    tape = make_tape(SEED, "dynamic")
    w = World(tape)
    w.begin_step()
    o1 = w.observation("scrambled")
    x1 = w.x_now.copy()
    assert np.allclose(o1.x, x1[tape.q[0]])
    c = ControllerC("C_scrambled")
    c.reset()
    c.decide(o1)
    fb = w.apply(WORK)
    c.feedback(fb, w.privileged_label())
    w.begin_step()
    o2 = w.observation("scrambled")
    c.decide(o2)
    z2 = c.queue[-1][1]
    assert np.allclose(z2[:8], w.x_now[tape.q[1]])
    assert np.allclose(z2[8:], 4 * (w.x_now[tape.q[1]] - x1[tape.q[0]]))


# ---------------------------------------------------------------- output-trend
def test_output_trend_delta():
    pol = OutputTrend(4, 0.94)
    pol.reset()
    assert pol.decide(_obs(1, np.zeros(8), E=1.0, n_o=0)) == WORK
    pol.feedback(Feedback(action=WORK, u=0.9, E=0.97, n_o=1))
    assert pol.delta == pytest.approx(0.97 - 1.0)
    # 0.97 + 4 * (-0.03) = 0.85 < 0.94
    assert pol.decide(_obs(2, np.zeros(8), E=0.97, n_o=1)) == SERVICE
    pol.feedback(Feedback(action=SERVICE, u=None, E=1.0, n_o=0))
    assert pol.delta == 0.0


# ---------------------------------------------------------------- wear regime
def test_static_multiplier_constant():
    for s in dev_seeds("static")[:10]:
        t = make_tape(s, "static")
        assert np.all(t.m == 0.92) and np.all(t.regime == 1)


def _runs(seq):
    runs, cur, n = [], seq[0], 0
    for v in seq:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = v, 1
    runs.append(n)
    return runs


def test_dynamic_minimum_dwell_and_switches_to_other_regime():
    n_switch = 0
    for s in dev_seeds("dynamic"):
        t = make_tape(s, "dynamic")
        runs = _runs(list(t.regime))
        assert all(r >= P.MIN_DWELL for r in runs[:-1])   # last run truncated by horizon
        n_switch += len(runs) - 1
        assert set(np.unique(t.m)) <= {0.48, 0.92, 1.65}
    assert n_switch > 0


# ---------------------------------------------------------------- sensors
def test_centered_channels_sum_to_zero_and_mapping():
    tape = make_tape(SEED, "dynamic")
    w = World(tape)
    for _ in range(50):
        w.begin_step()
        x = w.x_now
        raw = w.raw_now
        assert abs(x.sum()) < 1e-12
        placed = np.empty(8)
        for i in range(8):
            placed[tape.pi[i]] = tape.signs[tape.pi[i]] * raw[i]
        assert np.allclose(x, placed - placed.mean(), atol=1e-15)
        assert np.all(np.abs(raw) <= 1.7)
        w.apply(SERVICE if w.k >= 6 else WORK)


def test_sensor_formula_by_hand():
    pi = np.arange(8)
    signs = np.ones(8)
    eps = np.zeros(8)
    x, raw = sensor_vector(0.6, 0.75, 0.2, -0.3, 31, eps, pi, signs)
    expect = [(0.6 - 0.46) / 0.34, 0.0, (0.64 - 0.6) / 0.22, 0.2, 0.0, math.sin(1.0), -0.3, 0.0]
    assert np.allclose(raw, expect)
    assert np.allclose(x, np.array(expect) - np.mean(expect))


# ---------------------------------------------------------------- tape / order invariance
def test_tape_deterministic_and_seed_dependent():
    a = make_tape(SEED, "dynamic").fingerprint()
    b = make_tape(SEED, "dynamic").fingerprint()
    c = make_tape(SEED + 7919, "dynamic").fingerprint()
    assert a == b != c


def test_strategy_order_does_not_change_tape_or_results():
    settings = {"tuned_output": {"theta": 0.94}, "output_trend": {"L": 2, "theta": 0.965}}
    for mode in ("dynamic", "static"):
        seed = dev_seeds(mode)[3]
        fp0 = make_tape(seed, mode).fingerprint()
        fwd = run_world(seed, mode, base_strategies(settings))
        rev = run_world(seed, mode, list(reversed(base_strategies(settings))))
        assert make_tape(seed, mode).fingerprint() == fp0
        by = {r["strategy"]: r for r in rev}
        for r in fwd:
            r2 = by[r["strategy"]]
            for key, v in r.items():
                v2 = r2[key]
                if isinstance(v, float) and math.isnan(v):
                    assert math.isnan(v2)
                else:
                    assert v == v2, (r["strategy"], key)


# ---------------------------------------------------------------- information boundary
def test_observation_contains_only_permitted_fields():
    assert set(Observation.__dataclass_fields__) == set(OBSERVATION_FIELDS)
    assert set(Feedback.__dataclass_fields__) == set(FEEDBACK_FIELDS)
    hidden = {"D", "S", "c", "m", "a", "b", "regime", "capacity", "pi", "signs", "raw",
              "tape", "seed", "world", "y", "label", "h"}
    assert not (hidden & set(OBSERVATION_FIELDS))
    assert not (hidden & set(FEEDBACK_FIELDS))


def test_x_only_given_to_policies_that_declare_it():
    w = fresh()
    w.begin_step()
    assert w.observation(None).x is None
    assert w.observation("plain").x is not None
    for pol in [WorkOnly(), FixedP(7), OutputThreshold(0.965), OutputTrend(2, 0.9), OracleD()]:
        assert pol.x_mode is None
    obs = w.observation("plain")
    with pytest.raises(ValueError):
        obs.x[0] = 1.0            # read-only copy


def _refs(obj, seen=None):
    """All objects reachable from a policy's attributes."""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj)
    elif hasattr(obj, "__dict__"):
        items = list(vars(obj).values())
    else:
        try:
            from collections import deque
            items = list(obj) if isinstance(obj, deque) else []
        except TypeError:
            items = []
    for it in items:
        yield from _refs(it, seen)


def test_policies_hold_no_reference_to_world_or_tape():
    settings = {"tuned_output": {"theta": 0.94}, "output_trend": {"L": 2, "theta": 0.965}}
    tape = make_tape(SEED, "dynamic")
    for pol in base_strategies(settings) + tuning_grid():
        run_episode(tape, pol)
        for o in _refs(pol):
            assert not isinstance(o, (World, Tape)), pol.name


class _Spy(Policy):
    name = "spy"

    def __init__(self, x_mode=None, needs_D=False, needs_label=False):
        self.x_mode, self.needs_D, self.needs_label = x_mode, needs_D, needs_label
        self.seen_D = []
        self.seen_labels = []
        self.seen_x = []

    def decide(self, obs, D=None):
        self.seen_D.append(D)
        self.seen_x.append(obs.x)
        return SERVICE if obs.t % 9 == 0 else WORK

    def feedback(self, fb, label=None):
        self.seen_labels.append((fb.action, label))


def test_privileged_values_only_for_declared_exceptions():
    tape = make_tape(SEED, "dynamic")
    plain = _Spy()
    run_episode(tape, plain)
    assert all(d is None for d in plain.seen_D)
    assert all(lbl is None for _, lbl in plain.seen_labels)
    assert all(x is None for x in plain.seen_x)

    oracle = _Spy(needs_D=True)
    log = []
    run_episode(tape, oracle, log_rows=log)
    assert all(d is not None for d in oracle.seen_D)
    assert [r["D_pre"] for r in log] == oracle.seen_D          # pre-action D exactly
    assert all(lbl is None for _, lbl in oracle.seen_labels)

    lab = _Spy(needs_label=True)
    log = []
    run_episode(tape, lab, log_rows=log)
    assert all(d is None for d in lab.seen_D)
    for (act, lbl), row in zip(lab.seen_labels, log):
        if act == WORK:
            assert lbl == row["y_pre"]                          # pre-action symptom
        else:
            assert lbl is None                                  # no label on SERVICE


def test_world_guards_privileged_access_timing():
    w = fresh()
    w.begin_step()
    with pytest.raises(RuntimeError):
        w.privileged_label()                 # not before the action
    w.apply(SERVICE)
    with pytest.raises(RuntimeError):
        w.privileged_label()                 # not after SERVICE
    with pytest.raises(RuntimeError):
        w.privileged_D()                     # not after the action
    with pytest.raises(RuntimeError):
        OracleD().decide(_obs(1, None))      # oracle needs the explicit value
    w.begin_step()
    w.apply(WORK)
    assert w.privileged_label() in (0, 1)
