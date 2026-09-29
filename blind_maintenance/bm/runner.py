"""Episode runner: connects a World to a Policy through the restricted API,
collects end-of-episode metrics and (optionally) the analyzer step log."""
import math

from . import params as P
from .world import World

METRIC_COLUMNS = [
    "G", "survived", "died", "death_step", "steps_alive", "final_S", "final_R",
    "final_D", "n_work", "n_service", "B", "n_denied_service_requests",
    "alive_steps_r0", "alive_steps_r1", "alive_steps_r2",
    "services_r0", "services_r1", "services_r2", "n_train_C",
]

LOG_COLUMNS = [
    "t", "regime", "m", "b", "a", "D_pre", "S_pre", "R_pre", "k_pre", "E_pre", "n_o_pre",
    "c_pre", "y_pre", "allowed", "requested", "action", "g", "u", "R_post", "D_post",
    "S_post", "E_post", "n_o_post", "G_post", "alive_post",
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "policy_debug",
]


def run_episode(tape, policy, log_rows=None, world=None):
    """Run one policy on one world tape.  Returns a metrics dict.

    If `log_rows` is a list, one analyzer row per living step is appended
    (hidden state included - analyzer only, never passed to the policy).
    `world` lets tests start from a hand-prepared state (default: fresh World).
    """
    if world is None:
        world = World(tape)
    policy.reset()
    n_denied = 0
    while world.alive and world.t < P.T:
        world.begin_step()
        obs = world.observation(policy.x_mode)
        if log_rows is not None:
            pre = dict(t=world.t, regime=int(tape.regime[world.t - 1]),
                       m=float(tape.m[world.t - 1]), b=int(tape.b[world.t - 1]),
                       a=float(tape.a[world.t - 1]), D_pre=world.D, S_pre=world.S,
                       R_pre=world.R, k_pre=world.k, E_pre=world.E, n_o_pre=world.n_o,
                       c_pre=world.pre_action_capacity, y_pre=world.pre_action_symptom,
                       allowed=int(world.allowed_now))
            G_before = world.G
        if policy.needs_D:
            req = policy.decide(obs, D=world.privileged_D())
        else:
            req = policy.decide(obs)
        if req not in (P.WORK, P.SERVICE):
            raise ValueError(f"invalid action {req!r}")
        fb = world.apply(req)
        if req == P.SERVICE and fb.action == P.WORK:
            n_denied += 1
        label = None
        if policy.needs_label and fb.action == P.WORK:
            label = world.privileged_label()
        policy.feedback(fb, label)
        if log_rows is not None:
            x = world.x_now
            row = dict(pre)
            row.update(requested=req, action=fb.action, g=world.G - G_before,
                       u=(fb.u if fb.u is not None else math.nan), R_post=world.R,
                       D_post=world.D, S_post=world.S, E_post=world.E, n_o_post=world.n_o,
                       G_post=world.G, alive_post=int(world.alive))
            for i in range(P.N_CH):
                row[f"x{i}"] = float(x[i])
            dbg = policy.debug()
            row["policy_debug"] = ";".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                                           for k, v in dbg.items())
            log_rows.append(row)

    alive_steps = world.n_work + world.n_service
    B = (world.G - P.BASE_COST * alive_steps - P.WORK_COST * world.n_work
         - P.SERVICE_COST * world.n_service)
    return {
        "G": world.G,
        "survived": int(world.alive),
        "died": int(not world.alive),
        "death_step": world.death_step if world.death_step is not None else math.nan,
        "steps_alive": world.death_step if world.death_step is not None else P.T,
        "final_S": world.S,
        "final_R": world.R,
        "final_D": world.D,
        "n_work": world.n_work,
        "n_service": world.n_service,
        "B": B,
        "n_denied_service_requests": n_denied,
        "alive_steps_r0": world.alive_steps_regime[0],
        "alive_steps_r1": world.alive_steps_regime[1],
        "alive_steps_r2": world.alive_steps_regime[2],
        "services_r0": world.services_regime[0],
        "services_r1": world.services_regime[1],
        "services_r2": world.services_regime[2],
        "n_train_C": getattr(policy, "n", math.nan) if policy.needs_label else math.nan,
    }


def run_world(seed, mode, policies, log_rows_by_policy=None):
    """Run all policies on one world (one shared tape, separate states)."""
    from .tape import make_tape
    tape = make_tape(seed, mode)
    fp = tape.fingerprint()
    rows = []
    for pol in policies:
        log = None
        if log_rows_by_policy is not None:
            log = log_rows_by_policy.setdefault(pol.name + ("" if not pol.config else f"[{pol.config}]"), [])
        met = run_episode(tape, pol, log)
        row = {"mode": mode, "world_seed": seed, "strategy": pol.name,
               "config": pol.config, "tape_sha256": fp}
        row.update(met)
        rows.append(row)
    return rows
