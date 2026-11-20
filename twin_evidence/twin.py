"""Digital Twins layer: persistent scalar state per observable, updated by a gated Kalman step (Dossier §7.5).

gate  : d2 = (y - x_prior)^2 / (P_prior + u^2)        -> flag if d2 > 3.841 (95 %, 1 dof)
gain  : K  = P_prior / (P_prior + R),  R = R_factor * u^2  (PASS 1, WARN 2, FAIL -> no update)
state : x+ = x_prior + K (y - x_prior),  P+ = (1 - K) P_prior
drift : P_prior = P + Q,  Q = q6^2 * dt / 6  (q6 = process SD over six months, teaching assumption)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from .validate import Evaluation, to_months

CHI2_95_1DOF = 3.841

# process SD over six months (Dossier §7.5; same values as MDX.twin.DEFAULT_VARIABLES)
Q6 = {
    "roi_volume": lambda x: 0.01 * abs(x),
    "NAA_Cr": lambda x: 0.03,
    "Cho_Cr": lambda x: 0.03,
    "T1_ms": lambda x: 30.0,
}


@dataclass
class State:
    x: float
    P: float
    t: float  # months since baseline

    @property
    def sd(self) -> float:
        return math.sqrt(self.P)


@dataclass
class UpdateEvent:
    packet_id: str
    observable: str
    kind: str                 # init / update / reject
    action: str
    status: str
    y: float | None = None
    u: float | None = None
    R_gate: float | None = None
    R_gain: float | None = None
    dt_months: float | None = None
    Q: float | None = None
    prior: dict | None = None
    K: float | None = None
    d2: float | None = None
    d2_flag: bool | None = None
    z_change: float | None = None
    posterior: dict | None = None
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Twin:
    def __init__(self, months_per_q: float = 6.0, q6: dict | None = None):
        self.state: dict[str, State] = {}
        self.history: dict[str, list[State]] = {}
        self.log: list[UpdateEvent] = []
        self.months_per_q = months_per_q
        self.q6 = {**Q6, **(q6 or {})}  # extra observables (e.g. the scanner twin) bring their own drift

    def ingest(self, packet: dict, ev: Evaluation) -> UpdateEvent:
        obs, pid = packet.get("observable"), packet.get("packet_id")
        reasons = [f"{f.rule}: {f.reason}" for f in ev.findings if f.status != "PASS"]
        if ev.status == "FAIL" or ev.u_effective is None or not isinstance(packet.get("value"), (int, float)):
            e = UpdateEvent(pid, obs, "reject", "reject", ev.status, reasons=reasons or ["no usable value/uncertainty"])
            self.log.append(e)
            return e
        y, u = float(packet["value"]), float(ev.u_effective)
        R_gate = u * u
        R_gain = R_gate * float(ev.r_factor or 1.0)
        t = to_months(packet.get("relative_time"))
        s = self.state.get(obs)
        if s is None:  # first evidence initialises the state
            new = State(y, R_gain, t)
            e = UpdateEvent(pid, obs, "init", ev.action, ev.status, y, u, R_gate, R_gain, posterior={"x": y, "sd": new.sd}, reasons=reasons)
        else:
            dt = max(0.0, t - s.t)
            q = self.q6.get(obs, lambda x: 0.0)(s.x)
            Q = q * q * dt / self.months_per_q
            Pp = s.P + Q
            K = Pp / (Pp + R_gain)
            d2 = (y - s.x) ** 2 / (Pp + R_gate)
            new = State(s.x + K * (y - s.x), (1 - K) * Pp, t)
            # change test of the measurements (Dossier §3.10): z = delta / sqrt(u0^2 + u1^2)
            u0 = math.sqrt(self.history[obs][0].P) if self.history.get(obs) else None
            z = (y - self.history[obs][-1].x) / math.sqrt(u0 ** 2 + u ** 2) if u0 and len(self.history[obs]) == 1 else None
            e = UpdateEvent(pid, obs, "update", ev.action, ev.status, y, u, R_gate, R_gain, dt, Q,
                            {"x": s.x, "sd": math.sqrt(Pp)}, K, d2, d2 > CHI2_95_1DOF, z,
                            {"x": new.x, "sd": new.sd}, reasons)
        self.state[obs] = new
        self.history.setdefault(obs, []).append(new)
        self.log.append(e)
        return e

    def next_measurement(self, obs: str, sigma_m: float, r_factor: float = 1.0) -> float:
        """Dossier Eq. 7.3: posterior SD after an unbiased repeat with SD sigma_m (no extra drift)."""
        P = self.state[obs].P
        R = r_factor * sigma_m ** 2
        return math.sqrt(P * R / (P + R))
