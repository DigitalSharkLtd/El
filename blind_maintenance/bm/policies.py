"""Strategies (spec sections 6-7).

A policy only receives `Observation` and `Feedback` objects (plus the declared
privileged value for Oracle-D and controller C).  It never holds a reference
to the World or the Tape.

Interface
    x_mode      None | "plain" | "scrambled"   which sensor vector it receives
    needs_D     True only for Oracle-D
    needs_label True only for controller C (and its variants)
    reset()
    decide(obs, D=None) -> WORK | SERVICE      (requested action)
    feedback(fb, label=None)                   (after the action)
"""
import math
from collections import deque

import numpy as np

from . import params as P

WORK, SERVICE = P.WORK, P.SERVICE


class Policy:
    name = "policy"
    config = ""
    x_mode = None
    needs_D = False
    needs_label = False

    def reset(self):
        pass

    def decide(self, obs, D=None):
        raise NotImplementedError

    def feedback(self, fb, label=None):
        pass

    def debug(self):
        return {}


class WorkOnly(Policy):
    name = "work_only"

    def decide(self, obs, D=None):
        return WORK


class FixedP(Policy):
    """SERVICE when k >= p (k = WORK steps since the last SERVICE)."""

    def __init__(self, p):
        self.p = int(p)
        self.name = f"fixed_{self.p}"
        self.config = f"p={self.p}"

    def decide(self, obs, D=None):
        return SERVICE if obs.k >= self.p else WORK


class OutputThreshold(Policy):
    """Output-only (theta = 0.965) and Tuned-output: SERVICE if n_o > 0 and E < theta."""

    def __init__(self, theta, name=None):
        self.theta = float(theta)
        self.name = name or f"output_theta_{self.theta:g}"
        self.config = f"theta={self.theta:g}"

    def decide(self, obs, D=None):
        return SERVICE if (obs.n_o > 0 and obs.E < self.theta) else WORK


class OutputTrend(Policy):
    """SERVICE if n_o > 0 and E + L*min(Delta, 0) < theta.

    Delta = E_new - E_old of the last WORK; Delta = 0 after SERVICE (and at start).
    """

    def __init__(self, L, theta, name=None):
        self.L = float(L)
        self.theta = float(theta)
        self.name = name or f"output_trend_L{self.L:g}_theta_{self.theta:g}"
        self.config = f"L={self.L:g};theta={self.theta:g}"

    def reset(self):
        self.delta = 0.0
        self._E_before = None

    def decide(self, obs, D=None):
        self._E_before = obs.E
        score = obs.E + self.L * min(self.delta, 0.0)
        return SERVICE if (obs.n_o > 0 and score < self.theta) else WORK

    def feedback(self, fb, label=None):
        if fb.action == WORK:
            self.delta = fb.E - self._E_before
        else:
            self.delta = 0.0

    def debug(self):
        return {"delta": self.delta}


class OracleD(Policy):
    """Privileged threshold rule: SERVICE if pre-action D >= 0.53."""
    name = "oracle_D"
    config = f"D_threshold={P.ORACLE_D_THRESHOLD}"
    needs_D = True

    def decide(self, obs, D=None):
        if D is None:
            raise RuntimeError("Oracle-D requires the privileged D value")
        return SERVICE if D >= P.ORACLE_D_THRESHOLD else WORK


def _sigmoid(v):
    v = min(P.C_LOGIT_CLIP, max(-P.C_LOGIT_CLIP, v))
    return 1.0 / (1.0 + math.exp(-v))


class ControllerC(Policy):
    """Specified controller C (section 6) with variants
    "C" (plain x), "C_scrambled" (x[q_t] every step), "C_level" (derivative part zeroed).
    """
    needs_label = True

    def __init__(self, variant="C"):
        if variant not in ("C", "C_scrambled", "C_level"):
            raise ValueError(variant)
        self.variant = variant
        self.name = variant
        self.x_mode = "scrambled" if variant == "C_scrambled" else "plain"

    def reset(self):
        self.w = np.zeros(1 + 2 * P.N_CH)
        self.n = 0
        self.queue = deque()
        self.x_prev = None
        self.n_s = 0
        self._t = None
        self._last_risk = 0.0
        self._last_reactive = False
        self.n_accepted = 0
        self.n_discarded = 0

    def features(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.variant == "C_level" or self.x_prev is None:
            deriv = np.zeros(P.N_CH)
        else:
            deriv = P.C_DERIV_GAIN * (x - self.x_prev)
        self.x_prev = x.copy()          # updated on every living step
        return np.concatenate([x, deriv])

    def prob(self, z):
        phi = np.concatenate(([1.0], z))
        return _sigmoid(float(phi @ self.w))

    def decide(self, obs, D=None):
        z = self.features(obs.x)
        self._t = obs.t
        self.queue.append((obs.t, z, self.n_s))       # enqueue before deciding
        reactive = obs.n_o > 0 and obs.E < P.C_E_TRIGGER
        risk = 0.0 if self.n < P.C_MIN_EXAMPLES else self.prob(z)   # model before training
        self._last_risk = risk
        self._last_reactive = reactive
        return SERVICE if (reactive or risk > P.C_RISK_THRESHOLD) else WORK

    def _train(self, z_old, y):
        phi = np.concatenate(([1.0], z_old))
        p = _sigmoid(float(phi @ self.w))
        weight = P.C_POS_WEIGHT if y == 1 else 1.0
        reg = self.w.copy()
        reg[0] = 0.0
        grad = weight * (y - p) * phi - P.C_L2 * reg
        self.w = np.clip(self.w + P.C_LR * grad / (P.C_NORM_EPS + float(phi @ phi)),
                         -P.C_W_CLIP, P.C_W_CLIP)
        self.n += 1

    def feedback(self, fb, label=None):
        if fb.action == SERVICE:
            self.n_s += 1
            self.x_prev = None          # next derivative is zero; no training on SERVICE
            return
        if label is None:
            raise RuntimeError("controller C needs the delayed label after WORK")
        t = self._t
        while self.queue and self.queue[0][0] <= t - P.C_LABEL_DELAY:
            old_t, z_old, ns_old = self.queue.popleft()
            if old_t == t - P.C_LABEL_DELAY and ns_old == self.n_s:
                self._train(z_old, label)
                self.n_accepted += 1
            else:
                self.n_discarded += 1

    def debug(self):
        return {"risk": self._last_risk, "reactive": int(self._last_reactive),
                "n_train": self.n}


def base_strategies(settings=None):
    """The fixed list of strategies run on every world.

    settings: frozen tuned settings for one mode
        {"tuned_output": {"theta": ...}, "output_trend": {"L": ..., "theta": ...}}
    If None, tuned strategies are omitted (development grid is run separately).
    """
    out = [WorkOnly()]
    out += [FixedP(p) for p in P.FIXED_PERIODS]
    out += [OutputThreshold(P.OUTPUT_ONLY_THETA, name="output_only"),
            ControllerC("C"), ControllerC("C_scrambled"), ControllerC("C_level"),
            OracleD()]
    if settings is not None:
        out.append(OutputThreshold(settings["tuned_output"]["theta"], name="tuned_output"))
        out.append(OutputTrend(settings["output_trend"]["L"], settings["output_trend"]["theta"],
                               name="output_trend"))
    return out


def tuning_grid():
    """Development-only configuration grid for the two tuned comparison strategies."""
    grid = [OutputThreshold(th, name=f"tuned_output_grid") for th in P.THETA_GRID]
    grid += [OutputTrend(L, th, name=f"output_trend_grid")
             for L in P.L_GRID for th in P.THETA_GRID]
    return grid
