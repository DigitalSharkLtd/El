"""Hidden world state, step physics (section 5) and the observation API (section 4).

Policies never receive a World object.  The runner builds an `Observation`
(only permitted fields) and, for the two declared exceptions, passes the
privileged value explicitly:
  * Oracle-D receives the pre-action D;
  * controller C receives the delayed training label y_t after its WORK action.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import params as P
from .tape import Tape


def _clip(x, lo, hi):
    return hi if x > hi else (lo if x < lo else x)


@dataclass(frozen=True)
class Observation:
    """Everything an ordinary agent may see before choosing its action."""
    t: int                       # own step counter (derivable from own history)
    x: Optional[np.ndarray]      # centred sensor vector (None for output-only / fixed)
    R: float
    k: int
    service_allowed: bool
    E: float
    n_o: int


OBSERVATION_FIELDS = ("t", "x", "R", "k", "service_allowed", "E", "n_o")


@dataclass(frozen=True)
class Feedback:
    """What an agent learns after its action was executed."""
    action: int                  # executed action (WORK / SERVICE)
    u: Optional[float]           # normalised output after WORK, None after SERVICE
    E: float                     # updated output EMA
    n_o: int                     # updated number of EMA observations


FEEDBACK_FIELDS = ("action", "u", "E", "n_o")


def sensor_vector(d, a, ar, walk, t, eps, pi, signs):
    """Raw channels -> clip -> placement x_{pi(i)} = s_{pi(i)} raw_i -> centring."""
    raw = np.empty(P.N_CH)
    raw[0] = _clip((d - 0.46) / 0.34, -1.4, 1.4) + eps[0]
    raw[1] = _clip((d - 0.60) / 0.18, -1.4, 1.4) + eps[1]
    raw[2] = _clip((0.64 - d) / 0.22, -1.4, 1.4) + eps[2]
    raw[3] = ar + eps[3]
    raw[4] = _clip((a - 0.75) / 0.55, -1.2, 1.2) + eps[4]
    raw[5] = math.sin(t / 31.0) + eps[5]
    raw[6] = walk + eps[6]
    raw[7] = eps[7]
    np.clip(raw, -P.RAW_CLIP, P.RAW_CLIP, out=raw)
    x = np.empty(P.N_CH)
    x[pi] = signs[pi] * raw
    x -= x.mean()
    return x, raw


def capacity(D, S):
    c_D = _clip(1.0 - max(0.0, D - 0.72), 0.18, 1.0)
    c_S = _clip(1.0 - 1.55 * max(0.0, S - 0.45), 0.15, 1.0)
    return c_D * c_S


def service_allowed(R, k):
    return bool(R >= P.SERVICE_R_MIN and k >= P.SERVICE_K_MIN)


class World:
    def __init__(self, tape: Tape):
        self.tape = tape
        self.R = P.R0
        self.D = P.D0
        self.S = P.S0
        self.alive = True
        self.G = 0.0
        self.k = P.K0
        self.n_s = 0
        self.E = P.E0
        self.n_o = 0
        self.t = 0                     # last started step
        self.n_work = 0
        self.n_service = 0
        self.death_step = None
        self.alive_steps_regime = [0, 0, 0]
        self.services_regime = [0, 0, 0]
        self._phase = "idle"           # idle -> observed -> acted -> observed ...
        self._x = None
        self._raw = None
        self._c = None
        self._y = None
        self._allowed = None
        self._last_action = None

    # ----- step start: sensors, capacity, symptom -----
    def begin_step(self):
        if not self.alive:
            raise RuntimeError("dead world cannot start a step")
        if self._phase == "observed":
            raise RuntimeError("previous step not applied")
        if self.t >= P.T:
            raise RuntimeError("horizon reached")
        self.t += 1
        j = self.t - 1
        tp = self.tape
        d = _clip(self.D, 0.0, P.D_CLIP_MAX)
        self._x, self._raw = sensor_vector(d, tp.a[j], tp.ar[j], tp.walk[j], self.t,
                                           tp.eps_sensor[j], tp.pi, tp.signs)
        self._c = capacity(self.D, self.S)
        self._y = 1 if self._c < P.SYMPTOM_C else 0
        self._allowed = service_allowed(self.R, self.k)
        self._phase = "observed"

    # ----- permitted observation -----
    def observation(self, x_mode=None) -> Observation:
        if self._phase != "observed":
            raise RuntimeError("observation only between begin_step and apply")
        if x_mode is None:
            x = None
        elif x_mode == "plain":
            x = self._x.copy()
        elif x_mode == "scrambled":
            x = self._x[self.tape.q[self.t - 1]].copy()
        else:
            raise ValueError(x_mode)
        if x is not None:
            x.setflags(write=False)
        return Observation(t=self.t, x=x, R=self.R, k=self.k,
                           service_allowed=self._allowed, E=self.E, n_o=self.n_o)

    # ----- the two explicit privileged exceptions -----
    def privileged_D(self) -> float:
        """Oracle-D only: pre-action D of the current step."""
        if self._phase != "observed":
            raise RuntimeError("D is exposed only before the action of the current step")
        return self.D

    def privileged_label(self) -> int:
        """Controller C only: y_t = 1[c_t < 0.995] of the pre-action capacity,
        released after a WORK action has been executed."""
        if self._phase != "acted" or self._last_action != P.WORK:
            raise RuntimeError("label is released only after a WORK action")
        return self._y

    # ----- action -----
    def apply(self, requested: int) -> Feedback:
        if self._phase != "observed":
            raise RuntimeError("apply without begin_step")
        j = self.t - 1
        tp = self.tape
        action = P.SERVICE if (requested == P.SERVICE and self._allowed) else P.WORK
        reg = int(tp.regime[j])
        self.R -= P.BASE_COST
        u = None
        if action == P.WORK:
            a = float(tp.a[j])
            m = float(tp.m[j])
            g = max(0.0, P.PROD_GAIN * a * self._c + float(tp.eps_prod[j]))
            self.G += g
            self.R = min(P.R_MAX, self.R + g - P.WORK_COST)
            u = _clip(g / (P.PROD_GAIN * max(0.18, a)), 0.0, P.U_MAX)
            self.E = P.EMA_KEEP * self.E + P.EMA_NEW * u
            self.n_o += 1
            memory = _clip(0.91 + 0.018 * (m - 1.0), 0.86, 0.95)
            self.D = _clip(memory * self.D + (0.025 + 0.075 * a) * m + float(tp.h[j]),
                           0.0, P.D_CLIP_MAX)
            self.S += 0.026 * max(0.0, self.D - 0.58) + 0.115 * max(0.0, self.D - 0.82) ** 2
            self.k += 1
            self.n_work += 1
        else:
            self.R -= P.SERVICE_COST
            self.D = P.SERVICE_D_FACTOR * self.D
            self.n_s += 1
            self.k = 0
            self.E = 1.0
            self.n_o = 0
            self.n_service += 1
            self.services_regime[reg] += 1
        self.alive_steps_regime[reg] += 1
        self._last_action = action
        self._last_u = u
        self._phase = "acted"
        if self.R <= 0.0 or self.S >= P.S_DEATH:
            self.alive = False
            self.death_step = self.t
        return Feedback(action=action, u=u, E=self.E, n_o=self.n_o)

    # ----- analyzer-only helpers -----
    @property
    def pre_action_capacity(self):
        return self._c

    @property
    def pre_action_symptom(self):
        return self._y

    @property
    def allowed_now(self):
        return self._allowed

    @property
    def raw_now(self):
        return self._raw

    @property
    def x_now(self):
        return self._x
