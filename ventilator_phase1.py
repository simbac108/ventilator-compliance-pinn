"""
Phase 1 — Single-patient Ventilator Lung Model
================================================
Reproduces the RC lung model from the ventilator splitting paper.

Physics:
    Lung  →  RC circuit: C * dP_lung/dt = Q(t)
    Flow  →  Q(t) = (P_vent(t) - P_lung(t)) / R
    Combined ODE: dP_lung/dt = (P_vent(t) - P_lung(t)) / (R * C)

Units (matching the paper):
    Pressure  : cmH2O
    Flow      : mL/s
    Volume    : mL
    Time      : s
    Compliance: mL/cmH2O
    Resistance: cmH2O·s/mL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ════════════════════════════════════════════════════════════════════════════
# 1.  VENTILATOR CLASS  — pressure source (square wave)
# ════════════════════════════════════════════════════════════════════════════

class Ventilator:
    """
    Pressure-controlled ventilator.

    Delivers a square-wave pressure waveform:
        - Inspiratory phase: P = PIP  (duration = T_insp)
        - Expiratory phase:  P = PEEP (duration = T_exp)

    Parameters
    ----------
    pip   : Peak Inspiratory Pressure (cmH2O)
    peep  : Positive End-Expiratory Pressure (cmH2O)
    rr    : Respiratory Rate (breaths/min)
    ie    : Inspiratory-to-Expiratory ratio (e.g. 0.5 means 1:2)
    """

    def __init__(self, pip: float, peep: float, rr: float, ie: float = 1/3):
        self.pip  = pip
        self.peep = peep
        self.rr   = rr
        self.ie   = ie                          # I / (I+E)

        self.T_breath = 60.0 / rr              # total breath period (s)
        self.T_insp   = ie * self.T_breath      # inspiratory time (s)
        self.T_exp    = (1 - ie) * self.T_breath

    def pressure(self, t: float) -> float:
        """Return ventilator pressure at time t."""
        t_in_cycle = t % self.T_breath
        return self.pip if t_in_cycle < self.T_insp else self.peep

    def pressure_array(self, t: np.ndarray) -> np.ndarray:
        """Vectorised version for plotting."""
        return np.array([self.pressure(ti) for ti in t])

    def is_inspiratory(self, t: float) -> bool:
        return (t % self.T_breath) < self.T_insp

    def summary(self):
        print(f"Ventilator settings:")
        print(f"  PIP  = {self.pip} cmH2O")
        print(f"  PEEP = {self.peep} cmH2O")
        print(f"  RR   = {self.rr} bpm  →  T_breath = {self.T_breath:.2f}s")
        print(f"  I:E  = {self.ie:.2f}:{1-self.ie:.2f}")
        print(f"  T_insp = {self.T_insp:.2f}s  |  T_exp = {self.T_exp:.2f}s")


# ════════════════════════════════════════════════════════════════════════════
# 2.  LUNG CLASS  — RC circuit model
# ════════════════════════════════════════════════════════════════════════════

class Lung:
    """
    Single-compartment RC lung model.

    Modelled as a Hookean spring (compliance C) in series with
    a viscous dashpot (resistance R) — consistent with the paper.

    ODE:  dP_lung/dt = (P_vent(t) - P_lung(t)) / (R * C)
    Flow: Q(t) = (P_vent(t) - P_lung(t)) / R

    Parameters
    ----------
    compliance : C in mL/cmH2O  (typical ARDS: 20–40, healthy: 60–100)
    resistance : R in cmH2O·s/mL (typical ETT: ~5–15)
    label      : name for plots
    """

    def __init__(self, compliance: float, resistance: float, label: str = "Patient"):
        self.C     = compliance
        self.R     = resistance
        self.label = label

        # time constant τ = R*C — how fast the lung fills
        self.tau = resistance * compliance

    def ode(self, t: float, P_lung: float, ventilator: Ventilator) -> float:
        """
        Right-hand side of the lung ODE.
        dP_lung/dt = (P_vent - P_lung) / (R*C)
        """
        P_vent = ventilator.pressure(t)
        return (P_vent - P_lung) / self.tau

    def flow(self, P_vent: float, P_lung: float) -> float:
        """Q = (P_vent - P_lung) / R   [mL/s]"""
        return (P_vent - P_lung) / self.R

    def summary(self):
        print(f"\nLung — {self.label}:")
        print(f"  Compliance C = {self.C} mL/cmH2O")
        print(f"  Resistance R = {self.R} cmH2O·s/mL")
        print(f"  Time const τ = {self.tau:.2f} s")


# ════════════════════════════════════════════════════════════════════════════
# 3.  SIMULATOR CLASS  — integrates the ODE, computes clinical outputs
# ════════════════════════════════════════════════════════════════════════════

class VentilatorSimulator:
    """
    Integrates the lung ODE driven by the ventilator pressure waveform.

    Runs for N_breaths breath cycles and extracts:
        - P_lung(t) : lung pressure waveform
        - Q(t)      : flow waveform
        - V(t)      : volume waveform (integrated flow)
        - V_tidal   : steady-state tidal volume (mL)
        - P_peak    : peak lung pressure (cmH2O)
    """

    def __init__(self, ventilator: Ventilator, lung: Lung):
        self.vent = ventilator
        self.lung = lung

    def run(self, N_breaths: int = 8, dt: float = 0.001):
        """Simulate N_breaths breath cycles."""
        t_end = N_breaths * self.vent.T_breath
        t_eval = np.arange(0, t_end, dt)

        # Initial condition: lung at PEEP
        P0 = [self.vent.peep]

        # Integrate using scipy's RK45 (much more accurate than Euler)
        sol = solve_ivp(
            fun=lambda t, y: [self.lung.ode(t, y[0], self.vent)],
            t_span=(0, t_end),
            y0=P0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )

        t       = sol.t
        P_lung  = sol.y[0]
        P_vent  = self.vent.pressure_array(t)
        Q       = self.lung.flow(P_vent, P_lung)

        # Volume: integrate flow, reset at start of each breath
        V = np.zeros_like(Q)
        dt_arr = np.diff(t, prepend=t[0])
        for i in range(1, len(t)):
            t_in_cycle = t[i] % self.vent.T_breath
            if t_in_cycle < dt:             # new breath — reset
                V[i] = 0.0
            else:
                V[i] = V[i-1] + max(Q[i], 0) * dt_arr[i]   # only inspiratory flow

        return t, P_vent, P_lung, Q, V

    def steady_state_metrics(self, N_breaths: int = 8):
        """Return tidal volume and peak pressure at steady state (last breath)."""
        t, P_vent, P_lung, Q, V = self.run(N_breaths=N_breaths)

        # Extract last breath
        T = self.vent.T_breath
        mask = t >= (N_breaths - 1) * T

        V_tidal = V[mask].max()
        P_peak  = P_lung[mask].max()
        P_min   = P_lung[mask].min()

        return {
            "V_tidal (mL)"      : round(V_tidal, 1),
            "P_peak (cmH2O)"    : round(P_peak, 1),
            "P_min (cmH2O)"     : round(P_min, 1),
        }


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN — reproduce paper Figure 3 waveforms
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Ventilator settings from the paper (Fig 3) ──────────────────────────
    vent = Ventilator(pip=26, peep=13, rr=20, ie=1/3)
    vent.summary()

    # ── Two patients from the paper's benchtop validation (Table 2) ─────────
    # Paper reports compliance 18 mL/cmH2O (low) and 34 mL/cmH2O (medium)
    # Typical ETT resistance ~0.01 cmH2O·s/mL  (= 10 cmH2O·s/L in clinical units)
    # τ = R*C must be << T_breath (3s) for lung to fill meaningfully
    patient_A = Lung(compliance=18, resistance=0.01, label="Patient A (low compliance)")
    patient_B = Lung(compliance=34, resistance=0.01, label="Patient B (medium compliance)")

    patient_A.summary()
    patient_B.summary()

    # ── Run simulations ─────────────────────────────────────────────────────
    sim_A = VentilatorSimulator(vent, patient_A)
    sim_B = VentilatorSimulator(vent, patient_B)

    t_A, Pv_A, Pl_A, Q_A, V_A = sim_A.run(N_breaths=6)
    t_B, Pv_B, Pl_B, Q_B, V_B = sim_B.run(N_breaths=6)

    # ── Steady-state metrics ─────────────────────────────────────────────────
    print("\nSteady-state metrics:")
    metrics_A = sim_A.steady_state_metrics()
    metrics_B = sim_B.steady_state_metrics()
    print(f"  Patient A: {metrics_A}")
    print(f"  Patient B: {metrics_B}")
    print(f"\n  Paper reports: A ≈ 352-359 mL, B ≈ 566-567 mL")

    # ── Plot — mirroring paper Figure 3 layout ───────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "Single-Patient RC Lung Model — Reproducing Paper Figure 3\n"
        "Pressure-controlled ventilation: PIP=26, PEEP=13, RR=20 bpm",
        fontsize=12, fontweight='bold'
    )

    colors = {'A': 'black', 'B': 'gray', 'vent': 'steelblue'}
    last_n = 4   # show last 4 breaths for clarity

    T      = vent.T_breath
    t_show = t_A >= (6 - last_n) * T
    t_plot = t_A[t_show] - t_A[t_show][0]   # re-zero time axis

    # --- Pressure ---
    ax = axes[0]
    ax.plot(t_plot, Pv_A[t_show], color=colors['vent'],
            linewidth=1.2, linestyle='--', alpha=0.5, label='Ventilator P')
    ax.plot(t_plot, Pl_A[t_show], color=colors['A'],
            linewidth=2, label=f'Patient A (C={patient_A.C})')
    ax.plot(t_plot, Pl_B[t_show], color=colors['B'],
            linewidth=2, linestyle='--', label=f'Patient B (C={patient_B.C})')
    ax.set_ylabel('Pressure (cmH₂O)')
    ax.set_title('Airway Pressure')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    ax.set_ylim(8, 32)

    # --- Flow ---
    ax = axes[1]
    ax.plot(t_plot, Q_A[t_show], color=colors['A'], linewidth=2, label='Patient A')
    ax.plot(t_plot, Q_B[t_show], color=colors['B'],
            linewidth=2, linestyle='--', label='Patient B')
    ax.axhline(0, color='k', linewidth=0.8, linestyle=':')
    ax.set_ylabel('Flow (mL/s)')
    ax.set_title('Flow Rate')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    # --- Volume ---
    ax = axes[2]
    ax.plot(t_plot, V_A[t_show], color=colors['A'], linewidth=2,
            label=f"Patient A  →  Vt = {metrics_A['V_tidal (mL)']} mL")
    ax.plot(t_plot, V_B[t_show], color=colors['B'],
            linewidth=2, linestyle='--',
            label=f"Patient B  →  Vt = {metrics_B['V_tidal (mL)']} mL")
    ax.set_ylabel('Volume (mL)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Tidal Volume')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/ventilator_phase1.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved → ventilator_phase1.png")
