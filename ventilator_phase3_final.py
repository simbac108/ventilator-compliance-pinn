"""
Phase 3 v3 — Inverse PINN with Analytical Warm Start
=====================================================
Root cause of v1/v2 failure: the physics loss landscape is degenerate.
Making C → infinity drives alpha → 0, which shrinks the physics residual
regardless of waveform quality. Gradient descent exploits this every time.

Solution: don't ask gradient descent to find C from scratch.
Instead, analytically estimate C directly from the waveform's time
constant, then use the PINN to refine that estimate.

Analytical estimate (from RC circuit impulse response):
    During inspiration: P_lung(t) = P_vent - (P_vent - P_PEEP)*exp(-t/tau)
    At end of inspiration (t = T_insp):
        P_peak = P_vent - (P_vent - P_PEEP)*exp(-T_insp/tau)
    Solving for tau:
        tau = -T_insp / ln((P_vent - P_peak) / (P_vent - P_PEEP))
        C   = tau / R

This gives a clinically useful estimate from just two numbers:
    P_peak (measurable from sensor) and T_insp (known ventilator setting).

The PINN then refines this estimate using the full waveform.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# 1. PHYSICS CLASSES
# ============================================================================

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
        self.C   = compliance
        self.R   = resistance
        self.tau = resistance * compliance

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


# ============================================================================
# 2. ANALYTICAL C ESTIMATOR
# ============================================================================

def estimate_C_analytical(t_obs, P_obs, vent, R):
    """
    Estimate compliance C from the inspiratory pressure waveform.

    Method: nonlinear least-squares fit of the RC step response to the
    observed pressure during the inspiratory phase of the last 3 breaths.

        P_lung(t) = PIP - (PIP - P_e) * exp(-t / tau)

    where P_e is the measured lung pressure at the start of inspiration
    (NOT assumed to be PEEP -- at steady state the lung doesn't fully
    return to PEEP during expiration when tau is comparable to T_exp).

    Limitation: when tau << T_insp (stiff lung, C=18 case), the lung
    saturates quickly and the exponential portion of the waveform occupies
    only the first fraction of T_insp. With noise, the fit becomes less
    precise. The PINN refines from this initialisation using the full
    waveform shape.
    """
    from scipy.optimize import curve_fit

    T = vent.T_breath
    t_max = t_obs[-1]
    taus = []

    for b in range(1, 4):   # last 3 breaths
        t_start = t_max - b * T
        t_end   = t_start + vent.T_insp
        mask = (t_obs >= t_start) & (t_obs <= t_end)
        if mask.sum() < 6:
            continue

        ti = t_obs[mask] - t_obs[mask][0]   # reset to 0 at breath start
        Pi = P_obs[mask]
        P_e = float(np.median(Pi[:4]))       # start-of-breath pressure

        # Fit P(t) = PIP - (PIP - P_e)*exp(-t/tau)
        # Only use unsaturated portion: where lung is still >5% below PIP
        gap_thresh = vent.pip - 0.05 * (vent.pip - P_e)
        use = Pi < gap_thresh
        if use.sum() < 4:
            use[:min(8, len(use))] = True    # if fully saturated, use first 8 pts

        def model(t_arr, tau_fit):
            return vent.pip - (vent.pip - P_e) * np.exp(-t_arr / tau_fit)

        try:
            popt, _ = curve_fit(model, ti[use], Pi[use],
                                p0=[0.3], bounds=(0.005, 5.0),
                                maxfev=2000)
            taus.append(float(popt[0]))
        except Exception:
            pass

    tau_est = float(np.median(taus)) if taus else 0.3
    C_est   = float(np.clip(tau_est / R, 5.0, 150.0))

    print(f"  Analytical estimate (exponential curve fit):")
    print(f"    tau_est = {tau_est:.4f} s  ->  C_est = {C_est:.2f} mL/cmH2O")

    return C_est, tau_est


# ============================================================================
# 3. INVERSE PINN
# ============================================================================

class InverseLungPINN(nn.Module):
    """
    Inverse PINN: learns P_lung(t) and refines compliance C.

    Initialised from the analytical estimate — so C starts close to
    the truth and the PINN only needs to refine, not search.
    """

    def __init__(self, R, peep, pip, t_all, Pvent_all,
                 C_init, C_min=5.0, C_max=150.0,
                 hidden_layers=4, neurons=64):
        super().__init__()

        self.R     = R
        self.peep  = float(peep)
        self.pip   = float(pip)
        self.dP    = float(pip - peep)
        self.t_end = float(t_all[-1])
        self.C_min = C_min
        self.C_max = C_max

        # Initialise C_raw from analytical estimate
        frac      = np.clip((C_init - C_min) / (C_max - C_min), 0.01, 0.99)
        C_raw_0   = float(np.log(frac / (1.0 - frac)))
        self.C_raw = nn.Parameter(torch.tensor([C_raw_0], dtype=torch.float32))

        # Fixed normalised P_vent reference
        t_norm     = t_all / self.t_end
        Pvent_norm = (Pvent_all - peep) / self.dP
        self.register_buffer('t_ref',     torch.tensor(t_norm,     dtype=torch.float32))
        self.register_buffer('Pvent_ref', torch.tensor(Pvent_norm, dtype=torch.float32))

        # MLP
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.history = {
            "total": [], "data": [], "physics": [], "ic": [], "C_inferred": []
        }

    @property
    def C(self):
        return self.C_min + (self.C_max - self.C_min) * torch.sigmoid(self.C_raw)

    @property
    def alpha(self):
        return self.t_end / (self.R * self.C)

    def _t_norm(self, t):    return t / self.t_end
    def _P_denorm(self, Pn): return Pn * self.dP + self.peep

    def forward(self, t):
        return self._P_denorm(self.net(self._t_norm(t)))

    def _interp_Pvent_norm(self, t_tilde):
        idx = torch.searchsorted(self.t_ref, t_tilde.squeeze(), right=True)
        idx = torch.clamp(idx, 1, len(self.t_ref) - 1)
        t0  = self.t_ref[idx-1];  t1 = self.t_ref[idx]
        P0  = self.Pvent_ref[idx-1]; P1 = self.Pvent_ref[idx]
        w   = (t_tilde.squeeze() - t0) / (t1 - t0 + 1e-10)
        return (P0 + w * (P1 - P0)).unsqueeze(1)

    def data_loss(self, t_obs, P_obs):
        return torch.mean((self.forward(t_obs) - P_obs) ** 2)

    def physics_loss(self, t_col):
        """
        Reformulated residual — divides by alpha to remove degeneracy:
            (R*C / t_end) * dP̃/dt̃  -  (P̃_vent - P̃_lung)  =  0

        Now C appears in the NUMERATOR. Large C makes the left side
        large (not small), so the degenerate C→∞ solution is blocked.
        """
        t_n = self._t_norm(t_col).requires_grad_(True)
        P_n = self.net(t_n)

        dP_n_dt_n = torch.autograd.grad(
            P_n, t_n,
            grad_outputs=torch.ones_like(P_n),
            create_graph=True, retain_graph=True
        )[0]

        Pvent_n  = self._interp_Pvent_norm(t_n)

        # Reformulated: (R*C/t_end)*dP̃/dt̃ - (P̃_vent - P̃_lung) = 0
        # Equivalent ODE but C now in numerator — blocks large-C degeneracy
        inv_alpha = (self.R * self.C) / self.t_end
        residual  = inv_alpha * dP_n_dt_n - (Pvent_n - P_n)
        return torch.mean(residual ** 2)

    def ic_loss(self):
        t0  = torch.zeros(1, 1)
        P_n = self.net(self._t_norm(t0))
        return P_n ** 2

    def total_loss(self, t_col, t_obs, P_obs, lambda_phys=1.0, lambda_ic=50.0):
        L_data = self.data_loss(t_obs, P_obs)
        L_phys = self.physics_loss(t_col)
        L_ic   = self.ic_loss()
        L_tot  = L_data + lambda_phys * L_phys + lambda_ic * L_ic
        return L_tot, L_data, L_phys, L_ic


# ============================================================================
# 4. TRAINER  (single stage — analytical start means we don't need curriculum)
# ============================================================================

class InversePINNTrainer:
    def __init__(self, pinn, t_end, t_obs, P_obs,
                 N_col=2000, lambda_phys=1.0, lambda_ic=50.0):
        self.pinn        = pinn
        self.lambda_phys = lambda_phys
        self.lambda_ic   = lambda_ic

        self.t_obs = torch.tensor(t_obs.reshape(-1, 1), dtype=torch.float32)
        self.P_obs = torch.tensor(P_obs.reshape(-1, 1), dtype=torch.float32)
        t_col      = np.linspace(0, t_end, N_col).reshape(-1, 1)
        self.t_col = torch.tensor(t_col, dtype=torch.float32)

    def _log(self, epoch, L_tot, L_data, L_phys, L_ic):
        C_val = self.pinn.C.item()
        self.pinn.history["total"].append(L_tot)
        self.pinn.history["data"].append(L_data)
        self.pinn.history["physics"].append(L_phys)
        self.pinn.history["ic"].append(L_ic)
        self.pinn.history["C_inferred"].append(C_val)
        if epoch % 1000 == 0:
            print(f"    [{epoch:5d}]  L={L_tot:.3e}  "
                  f"L_data={L_data:.3e}  L_phys={L_phys:.3e}  "
                  f"C={C_val:.2f} mL/cmH2O")

    def train_adam(self, epochs=4000, lr_net=1e-3, lr_C=1e-3):
        print(f"  Adam ({epochs} epochs) ...")
        opt = optim.Adam([
            {"params": self.pinn.net.parameters(), "lr": lr_net},
            {"params": [self.pinn.C_raw],          "lr": lr_C},
        ])
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9997)

        for epoch in range(epochs):
            opt.zero_grad()
            L_tot, L_data, L_phys, L_ic = self.pinn.total_loss(
                self.t_col, self.t_obs, self.P_obs,
                self.lambda_phys, self.lambda_ic
            )
            L_tot.backward()
            opt.step(); sched.step()
            self._log(epoch, L_tot.item(), L_data.item(),
                      L_phys.item(), L_ic.item())

    def train_lbfgs(self, max_iter=300):
        print(f"  L-BFGS (up to {max_iter} iters) ...")
        opt = optim.LBFGS(
            self.pinn.parameters(), max_iter=max_iter,
            tolerance_grad=1e-9, tolerance_change=1e-12,
            history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            L_tot, L_data, L_phys, L_ic = self.pinn.total_loss(
                self.t_col, self.t_obs, self.P_obs,
                self.lambda_phys, self.lambda_ic
            )
            L_tot.backward()
            self.pinn.history["total"].append(L_tot.item())
            self.pinn.history["data"].append(L_data.item())
            self.pinn.history["physics"].append(L_phys.item())
            self.pinn.history["ic"].append(L_ic.item())
            self.pinn.history["C_inferred"].append(self.pinn.C.item())
            return L_tot

        opt.step(closure)
        print(f"    Final loss: {self.pinn.history['total'][-1]:.3e}")
        print(f"    Final C:    {self.pinn.C.item():.3f} mL/cmH2O")

    def train(self):
        self.train_adam(epochs=4000)
        self.train_lbfgs(max_iter=300)


# ============================================================================
# 5. MAIN
# ============================================================================

def run_case(vent, C_true, R, noise_std, label):
    print(f"\n{'='*60}")
    print(f"Case: {label}")
    print(f"  C_true={C_true}  noise={noise_std} cmH2O")
    print(f"{'='*60}")

    # Ground truth
    lung   = Lung(C_true, R)
    solver = ODE_Solver(vent, lung)
    t_true, P_true = solver.solve(N_breaths=5, dt=0.002)
    t_end  = t_true[-1]

    # Noisy observations
    t_obs = t_true[::5]
    P_obs = P_true[::5] + np.random.normal(0, noise_std, len(t_true[::5]))

    P_vent_full = vent.pressure_array(t_true)

    # Step 1: analytical estimate
    C_analytical, tau_est = estimate_C_analytical(t_obs, P_obs, vent, R)
    print(f"    C_true = {C_true}  |  C_analytical = {C_analytical:.2f}  "
          f"|  error = {100*abs(C_analytical-C_true)/C_true:.1f}%")

    # Step 2: PINN refinement starting from analytical estimate
    print(f"\n  PINN refinement from C_init = {C_analytical:.2f} ...")
    pinn = InverseLungPINN(
        R=R, peep=vent.peep, pip=vent.pip,
        t_all=t_true, Pvent_all=P_vent_full,
        C_init=C_analytical,
        hidden_layers=4, neurons=64
    )

    trainer = InversePINNTrainer(
        pinn, t_end=t_end, t_obs=t_obs, P_obs=P_obs,
        N_col=2000, lambda_phys=1.0, lambda_ic=50.0
    )
    trainer.train()

    C_pinn      = pinn.C.item()
    err_analyt  = 100 * abs(C_analytical - C_true) / C_true
    err_pinn    = 100 * abs(C_pinn - C_true) / C_true

    print(f"\n  RESULT:")
    print(f"    C_true        = {C_true:.1f} mL/cmH2O")
    print(f"    C_analytical  = {C_analytical:.2f}  (error = {err_analyt:.1f}%)")
    print(f"    C_PINN        = {C_pinn:.2f}  (error = {err_pinn:.1f}%)")

    t_test = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        P_pred = pinn(t_test).numpy().flatten()

    return {
        "label": label, "C_true": C_true,
        "C_analytical": C_analytical, "err_analyt": err_analyt,
        "C_pinn": C_pinn, "err_pinn": err_pinn,
        "t_true": t_true, "P_true": P_true, "P_pred": P_pred,
        "t_obs": t_obs, "P_obs": P_obs,
        "history": pinn.history, "vent": vent
    }


if __name__ == "__main__":

    vent = Ventilator(pip=26, peep=13, rr=20, ie=1/3)
    R    = 0.01

    case1 = run_case(vent, C_true=18, R=R, noise_std=1.0,
                     label="Low compliance  (C=18, ARDS)")

    case2 = run_case(vent, C_true=34, R=R, noise_std=1.0,
                     label="Medium compliance  (C=34)")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        "Phase 3 — Inverse PINN: Inferring Pulmonary Compliance\n"
        "Analytical warm start + PINN refinement  (noise = 1 cmH2O)",
        fontsize=13, fontweight='bold'
    )

    for col, case in enumerate([case1, case2]):
        t_true  = case["t_true"]
        P_true  = case["P_true"]
        P_pred  = case["P_pred"]
        history = case["history"]

        # Row 1: waveform
        ax1 = fig.add_subplot(3, 2, col + 1)
        ax1.scatter(case["t_obs"], case["P_obs"], s=3, alpha=0.3,
                    color='steelblue', label='Noisy sensor data', zorder=1)
        ax1.plot(t_true, P_true, 'k-',  lw=1.5, label='True P_lung', zorder=3)
        ax1.plot(t_true, P_pred, 'r--', lw=1.8,
                 label=f'PINN  C={case["C_pinn"]:.1f} '
                       f'(true={case["C_true"]})', zorder=4)
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Pressure (cmH2O)')
        ax1.set_title(case["label"])
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        # Row 2: C convergence — show both analytical and PINN
        ax2 = fig.add_subplot(3, 2, col + 3)
        C_hist = history["C_inferred"]
        ax2.plot(C_hist, color='darkorange', lw=1.5, label='C (PINN)')
        ax2.axhline(case["C_true"],       color='k',       ls='--', lw=2.0,
                    label=f'True C = {case["C_true"]}')
        ax2.axhline(case["C_analytical"], color='seagreen', ls='--', lw=1.5,
                    label=f'Analytical = {case["C_analytical"]:.1f} '
                          f'({case["err_analyt"]:.1f}% err)')
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('C (mL/cmH2O)')
        ax2.set_title('Compliance Convergence')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.annotate(
            f'  PINN: {case["C_pinn"]:.1f}\n  ({case["err_pinn"]:.1f}% err)',
            xy=(len(C_hist)-1, case["C_pinn"]),
            fontsize=9, color='darkorange'
        )

        # Row 3: loss
        ax3 = fig.add_subplot(3, 2, col + 5)
        iters = range(len(history["total"]))
        ax3.semilogy(iters, history["total"],   color='steelblue', lw=1.2, label='Total')
        ax3.semilogy(iters, history["data"],    color='tomato',    lw=1.2, label='Data',    ls='--')
        ax3.semilogy(iters, history["physics"], color='seagreen',  lw=1.2, label='Physics', ls=':')
        ax3.axvline(4000, color='gray', ls='--', alpha=0.6, label='Adam->L-BFGS')
        ax3.set_xlabel('Iteration'); ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ventilator_phase3.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved -> ventilator_phase3.png")
