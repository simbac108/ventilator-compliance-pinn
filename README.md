# Inferring Pulmonary Compliance via Inverse PINN

An extension of [Bishawi et al., ICCS 2022](https://doi.org/10.1007/978-3-031-08757-8_13) that solves the inverse problem the original paper could not address.

---

## Background

During the COVID-19 pandemic, our team at Duke University built a ventilator splitting and resistor system (VSRS) to safely share a single ventilator between two patients with differing pulmonary mechanics. The clinical decision support tool required **270 million simulations** across a 7-dimensional parameter space, executed on 24,000 Azure HPC cores in 72 hours.

The published paper explicitly identified one limitation:

> *"It will be important to redetermine the optimal resistor and ventilator settings as the patients' conditions can change rapidly with time."*

In practice, measuring pulmonary compliance requires temporarily occluding airflow — interrupting ventilation. This project addresses that limitation.

---

## What This Project Does

A **Physics-Informed Neural Network (PINN)** that infers pulmonary compliance *C* continuously from observed pressure waveforms — no ventilation interruption required.

```
Observed noisy P_lung(t)  +  known ventilator settings
                ↓
          Inverse PINN
                ↓
     Inferred compliance C
     (1–7% error at ±1 cmH₂O sensor noise)
```

---

## Physics

The lung is modelled as a single-compartment RC circuit (consistent with the original paper):

```
dP_lung/dt = (P_vent(t) - P_lung(t)) / (R × C)
```

- **Forward problem** (Phase 2): given known C and R, learn P_lung(t)
- **Inverse problem** (Phase 3): given observed P_lung(t), infer unknown C

---

## Results

| Patient | C true | C inferred | Error | Sensor noise |
|---|---|---|---|---|
| Low compliance (ARDS) | 18 mL/cmH₂O | ~18.9 | ~7% | ±1 cmH₂O |
| Medium compliance | 34 mL/cmH₂O | ~34.4 | ~1% | ±1 cmH₂O |

ICU pressure transducers typically have ±1–2 cmH₂O accuracy. The PINN operates within the physical noise floor of the sensors themselves.

---

## Key Technical Contributions

**Reformulated physics residual** — the standard PINN formulation allows a degenerate minimum where C → ∞ trivially minimises the loss. Rearranging the ODE so C appears in the numerator blocks this:

```python
# Degenerate (C→∞ makes this vanish):
residual = dP̃/dt̃ - (t_end/R·C) · (P̃_vent - P̃_lung)

# Stable (large C increases residual):
residual = (R·C/t_end) · dP̃/dt̃ - (P̃_vent - P̃_lung)
```

**Analytical warm start** — compliance is pre-estimated from a nonlinear least-squares fit of the RC step response to the inspiratory waveform. This gives the optimiser a physically meaningful starting point.

**Normalisation** — time and pressure are rescaled to [0, 1] before training. Without this, the ODE residual is O(70 cmH₂O/s) vs data residual O(1 cmH₂O), causing training instability.

---

## Project Structure

```
ventilator_phase1.py   — ODE simulator: reproduces paper waveforms
                         Classes: Ventilator, Lung, ODE_Solver
                         Validates against paper (tidal volume ratio correct)

ventilator_phase2.py   — Forward PINN: learns P_lung(t) from physics alone
                         Mean absolute error < 0.1 cmH₂O vs. ODE solver
                         Validates PINN framework and normalisation

ventilator_phase3.py   — Inverse PINN: infers unknown compliance C
                         Analytical warm start + PINN refinement
                         Closes the limitation stated in Section 4 of paper
```

---

## Installation

```bash
pip install torch numpy matplotlib scipy
```

No GPU required. All three phases run on CPU in under 10 minutes.

## Usage

```bash
python ventilator_phase1.py   # ODE simulation — waveforms and tidal volume
python ventilator_phase2.py   # forward PINN training — ~5 min
python ventilator_phase3.py   # inverse compliance inference — ~8 min
```

Each script saves a plot (`ventilator_phaseN.png`) on completion.

---

## Documentation

Full documentation covering all theory and code line-by-line, written for readers with no prior experience in differential equations or machine learning: [`ventilator_pinn_docs.pdf`](./ventilator_pinn_docs.pdf)

---

## Reference

Bishawi M., Kaplan M., **Chidyagwai S.**, et al.
*Patient- and Ventilator-Specific Modeling to Drive the Use and Development of 3D Printed Devices for Rapid Ventilator Splitting During the COVID-19 Pandemic.*
ICCS 2022. DOI: [10.1007/978-3-031-08757-8_13](https://doi.org/10.1007/978-3-031-08757-8_13)
# ventilator-compliance-pinn
# ventilator-compliance-pinn
