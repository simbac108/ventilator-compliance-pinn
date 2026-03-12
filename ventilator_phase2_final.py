"""
Phase 2 — Forward PINN for Single-Patient Lung Model  (v2 — normalised)
========================================================================
Key fix over v1: normalise time and pressure to [0,1] before training.

Without normalisation the ODE RHS is O(70 cmH2O/s) → huge residuals
→ loss gets stuck. With normalisation all quantities are O(1).

    t̃ = t / t_end              (time scaled to [0,1])
    P̃ = (P - PEEP)/(PIP-PEEP)  (pressure scaled to [0,1])

Normalised ODE:
    dP̃/dt̃ = alpha * (P̃_vent - P̃_lung),   alpha = t_end / tau

Option 3 for discontinuity:
    P_vent(t) precomputed from Phase 1 Ventilator class and interpolated
    at collocation points. Network only learns the smooth lung response.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

torch.manual_seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICS CLASSES  (Phase 1 — unchanged)
# ════════════════════════════════════════════════════════════════════════════

class Ventilator:
    def __init__(self, pip, peep, rr, ie=1/3):
        self.pip      = pip
        self.peep     = peep
        self.rr       = rr
        self.ie       = ie
        self.T_breath = 60.0 / rr
        self.T_insp   = ie * self.T_breath

    def pressure(self, t):
        return self.pip if (t % self.T_breath) < self.T_insp else self.peep

    def pressure_array(self, t):
        return np.array([self.pressure(ti) for ti in t])


class Lung:
    def __init__(self, compliance, resistance, label="Patient"):
        self.C     = compliance
        self.R     = resistance
        self.tau   = resistance * compliance
        self.label = label

    def ode(self, t, P_lung, ventilator):
        return (ventilator.pressure(t) - P_lung) / self.tau


class ODE_Solver:
    def __init__(self, ventilator, lung):
        self.vent = ventilator
        self.lung = lung

    def solve(self, N_breaths=6, dt=0.005):
        t_end  = N_breaths * self.vent.T_breath
        t_eval = np.arange(0, t_end, dt)
        sol    = solve_ivp(
            fun=lambda t, y: [self.lung.ode(t, y[0], self.vent)],
            t_span=(0, t_end),
            y0=[self.vent.peep],
            t_eval=t_eval,
            method='RK45', rtol=1e-7, atol=1e-9
        )
        return sol.t, sol.y[0]


# ════════════════════════════════════════════════════════════════════════════
# 2.  FORWARD PINN  (normalised)
# ════════════════════════════════════════════════════════════════════════════

class LungPINN(nn.Module):
    """
    Forward PINN — learns P_lung(t) from physics alone (no data).

    Normalisation (key fix):
        t̃ = t / t_end,   P̃ = (P - PEEP) / (PIP - PEEP)
        → all inputs/outputs in [0,1], gradients well-behaved

    Normalised ODE:
        dP̃/dt̃ = alpha * (P̃_vent - P̃_lung)
        alpha  = t_end / tau   (large number, but values are O(1))
    """

    def __init__(self, tau, peep, pip, t_all, Pvent_all,
                 hidden_layers=4, neurons=64):
        super().__init__()

        self.tau   = tau
        self.peep  = float(peep)
        self.pip   = float(pip)
        self.dP    = float(pip - peep)
        self.t_end = float(t_all[-1])
        self.alpha = self.t_end / tau      # ODE multiplier in norm. coords

        # Normalised reference arrays (fixed — not trained)
        t_norm     = t_all / self.t_end
        Pvent_norm = (Pvent_all - peep) / self.dP
        self.register_buffer('t_ref',     torch.tensor(t_norm,     dtype=torch.float32))
        self.register_buffer('Pvent_ref', torch.tensor(Pvent_norm, dtype=torch.float32))

        # MLP: input t̃ ∈ [0,1]  →  output P̃ ∈ [0,1]
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.history = {"total": [], "physics": [], "ic": []}

    # ── normalisation helpers ────────────────────────────────────────────────
    def _t_norm(self, t):
        return t / self.t_end

    def _P_denorm(self, P_tilde):
        return P_tilde * self.dP + self.peep

    # ── forward: raw time in → raw pressure out ───────────────────────────────
    def forward(self, t):
        return self._P_denorm(self.net(self._t_norm(t)))

    # ── P_vent interpolation (normalised space) ───────────────────────────────
    def _interp_Pvent_norm(self, t_tilde):
        """Linear interpolation of normalised P_vent at query points."""
        idx = torch.searchsorted(self.t_ref, t_tilde.squeeze(), right=True)
        idx = torch.clamp(idx, 1, len(self.t_ref) - 1)

        t0 = self.t_ref[idx - 1];  t1 = self.t_ref[idx]
        P0 = self.Pvent_ref[idx - 1]; P1 = self.Pvent_ref[idx]

        w = (t_tilde.squeeze() - t0) / (t1 - t0 + 1e-10)
        return (P0 + w * (P1 - P0)).unsqueeze(1)

    # ── physics loss (normalised ODE residual) ────────────────────────────────
    def physics_loss(self, t_col):
        """
        Residual of normalised ODE:
            dP̃/dt̃  -  alpha*(P̃_vent - P̃_lung)  =  0
        Everything is O(1) → stable.
        """
        t_n = self._t_norm(t_col).requires_grad_(True)
        P_n = self.net(t_n)

        dP_n_dt_n = torch.autograd.grad(
            P_n, t_n,
            grad_outputs=torch.ones_like(P_n),
            create_graph=True, retain_graph=True
        )[0]

        Pvent_n  = self._interp_Pvent_norm(t_n)
        residual = dP_n_dt_n - self.alpha * (Pvent_n - P_n)
        return torch.mean(residual ** 2)

    # ── IC loss: P̃(0) = 0  (i.e. P_lung(0) = PEEP) ──────────────────────────
    def ic_loss(self):
        t0  = torch.zeros(1, 1)
        P_n = self.net(self._t_norm(t0))
        return P_n ** 2

    # ── total loss ────────────────────────────────────────────────────────────
    def total_loss(self, t_col, lambda_ic=50.0):
        L_phys = self.physics_loss(t_col)
        L_ic   = self.ic_loss()
        L_tot  = L_phys + lambda_ic * L_ic
        return L_tot, L_phys, L_ic


# ════════════════════════════════════════════════════════════════════════════
# 3.  TRAINER
# ════════════════════════════════════════════════════════════════════════════

class PINNTrainer:
    def __init__(self, pinn, t_end, N_col=2000, lambda_ic=50.0):
        self.pinn      = pinn
        self.lambda_ic = lambda_ic
        t_col          = np.linspace(0, t_end, N_col).reshape(-1, 1)
        self.t_col     = torch.tensor(t_col, dtype=torch.float32)

    def train_adam(self, epochs=5000, lr=1e-3):
        print(f"  Adam — {epochs} epochs …")
        opt   = optim.Adam(self.pinn.parameters(), lr=lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9997)

        for epoch in range(epochs):
            opt.zero_grad()
            L_tot, L_phys, L_ic = self.pinn.total_loss(self.t_col, self.lambda_ic)
            L_tot.backward()
            opt.step(); sched.step()

            self.pinn.history["total"].append(L_tot.item())
            self.pinn.history["physics"].append(L_phys.item())
            self.pinn.history["ic"].append(L_ic.item())

            if epoch % 1000 == 0:
                print(f"    [{epoch:5d}]  L={L_tot.item():.3e}  "
                      f"L_phys={L_phys.item():.3e}  "
                      f"L_ic={L_ic.item():.3e}")

    def train_lbfgs(self, max_iter=500):
        print(f"  L-BFGS — up to {max_iter} iterations …")
        opt = optim.LBFGS(
            self.pinn.parameters(), max_iter=max_iter,
            tolerance_grad=1e-9, tolerance_change=1e-12,
            history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            L_tot, L_phys, L_ic = self.pinn.total_loss(self.t_col, self.lambda_ic)
            L_tot.backward()
            self.pinn.history["total"].append(L_tot.item())
            self.pinn.history["physics"].append(L_phys.item())
            self.pinn.history["ic"].append(L_ic.item())
            return L_tot

        opt.step(closure)
        print(f"    Final loss: {self.pinn.history['total'][-1]:.3e}")

    def train(self, adam_epochs=5000, lbfgs_iter=500):
        print("Training …")
        self.train_adam(epochs=adam_epochs)
        self.train_lbfgs(max_iter=lbfgs_iter)
        print("Done.\n")


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Setup ────────────────────────────────────────────────────────────────
    vent      = Ventilator(pip=26, peep=13, rr=20, ie=1/3)
    patient_A = Lung(compliance=18, resistance=0.01, label="Patient A")

    print(f"tau   = {patient_A.tau:.4f} s")
    print(f"alpha = t_end/tau = {5 * vent.T_breath / patient_A.tau:.1f}  "
          f"(ODE multiplier in normalised coords)")

    # ── Ground truth ─────────────────────────────────────────────────────────
    N_breaths = 5
    solver    = ODE_Solver(vent, patient_A)
    t_true, P_true = solver.solve(N_breaths=N_breaths, dt=0.002)
    t_end          = t_true[-1]
    P_vent_full    = vent.pressure_array(t_true)

    # ── Build and train PINN ──────────────────────────────────────────────────
    pinn = LungPINN(
        tau=patient_A.tau, peep=vent.peep, pip=vent.pip,
        t_all=t_true, Pvent_all=P_vent_full,
        hidden_layers=4, neurons=64
    )
    print(f"Network parameters: {sum(p.numel() for p in pinn.parameters())}\n")

    trainer = PINNTrainer(pinn, t_end=t_end, N_col=2000, lambda_ic=50.0)
    trainer.train(adam_epochs=5000, lbfgs_iter=500)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    t_test = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        P_pred = pinn(t_test).numpy().flatten()

    error = np.abs(P_pred - P_true)
    print(f"Max absolute error : {error.max():.4f} cmH2O")
    print(f"Mean absolute error: {error.mean():.4f} cmH2O")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Phase 2 — Forward PINN (normalised)\n"
        f"Patient A: C={patient_A.C} mL/cmH₂O, R={patient_A.R}, τ={patient_A.tau}s",
        fontsize=13, fontweight='bold'
    )

    # Panel 1: full waveform
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.plot(t_true, P_vent_full, color='steelblue', lw=1,
             alpha=0.4, label='P_vent (known input)')
    ax1.plot(t_true, P_true,  'k-',  lw=2,   label='ODE solver (truth)')
    ax1.plot(t_true, P_pred,  'r--', lw=1.8, label='PINN prediction', alpha=0.9)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Pressure (cmH₂O)')
    ax1.set_title('Lung Pressure: PINN vs Ground Truth')
    ax1.legend(); ax1.grid(True, alpha=0.4)

    # Panel 2: zoom last breath
    ax2 = fig.add_subplot(2, 2, 3)
    mask = t_true >= (N_breaths - 1) * vent.T_breath
    ax2.plot(t_true[mask], P_true[mask],  'k-',  lw=2,   label='ODE solver')
    ax2.plot(t_true[mask], P_pred[mask],  'r--', lw=1.8, label='PINN')
    ax2.fill_between(t_true[mask], P_true[mask], P_pred[mask],
                     alpha=0.2, color='red', label='Error')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Pressure (cmH₂O)')
    ax2.set_title('Zoom: Last Breath (steady state)')
    ax2.legend(); ax2.grid(True, alpha=0.4)

    # Panel 3: loss history
    ax3 = fig.add_subplot(2, 2, 4)
    iters = range(len(pinn.history["total"]))
    ax3.semilogy(iters, pinn.history["total"],   label='Total',   color='steelblue')
    ax3.semilogy(iters, pinn.history["physics"], label='Physics', color='tomato',   ls='--')
    ax3.semilogy(iters, pinn.history["ic"],      label='IC',      color='seagreen', ls=':')
    ax3.axvline(5000, color='gray', ls='--', alpha=0.6, label='Adam → L-BFGS')
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss History')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('ventilator_phase2.png', dpi=150, bbox_inches='tight')
    print("Plot saved → ventilator_phase2.png")
