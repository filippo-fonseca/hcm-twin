"""D6: a single self-contained HTML page, no server and no build step.

The page has to redraw a pressure-volume loop on every slider movement, and a steady-state
solve takes 37 ms of Python. Shipping the solver to the browser is not an option and
neither is a round trip. So the page carries an *emulator*: the same polynomial surrogate
the identifiability analysis uses, fitted here over the slider ranges and emitted as three
arrays of numbers that a dozen lines of JavaScript evaluate in microseconds.

Two things make that honest rather than a shortcut.

The emulator's held-out accuracy is measured and **printed on the page**. A reader can see
how far to trust the curve they are dragging.

The drug step is computed exactly in JavaScript rather than emulated. Dose enters the
model only through ``phi_eff = phi * (1 - E_max C / (EC50 + C))`` with ``C = dose / CL``,
which is a closed-form expression; feeding the emulator ``phi_eff`` instead of ``(phi,
dose)`` removes a saturating nonlinearity from the fit and makes the dose slider exact.

The page is the artifact that gets attention in a lab meeting. The tie-breaker table is
what survives scrutiny afterwards. Both are deliverables and this one is not the more
important of the two.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import defaults as d
from ..drug import effective_phi, steady_state_concentration_ng_per_ml
from ..model import simulate_cohort
from ..observables import observe_arrays
from ..parameters import ModelConstants
from ..analysis.surrogate import Surrogate, latin_hypercube

logger = logging.getLogger(__name__)

LOOP_POINTS: int = 96
"""How many points of the pressure-volume loop the emulator reproduces."""

SLIDERS: tuple[dict[str, object], ...] = (
    {"key": "phi", "label": "Myosin availability", "hint": "fraction of unattached heads that are unparked; HCM raises it",
     "low": 0.28, "high": 0.62, "step": 0.005, "value": 0.55, "units": ""},
    {"key": "a_pas", "label": "Passive stiffness", "hint": "scale of the exponential passive fiber stress",
     "low": 0.40, "high": 5.50, "step": 0.05, "value": 4.00, "units": "kPa"},
    {"key": "wall", "label": "Wall volume", "hint": "measured: LV mass divided by myocardial density",
     "low": 100.0, "high": 300.0, "step": 1.0, "value": 245.0, "units": "mL"},
    {"key": "cavity", "label": "Unloaded cavity volume", "hint": "measured: cavity size at zero fiber strain",
     "low": 45.0, "high": 80.0, "step": 0.5, "value": 60.0, "units": "mL"},
    {"key": "dose", "label": "Maintained dose", "hint": "myosin inhibitor, mg per day at steady state",
     "low": 0.0, "high": 15.0, "step": 0.25, "value": 0.0, "units": "mg/day"},
    {"key": "clearance", "label": "Drug clearance", "hint": "hidden: CYP2C19 poor metabolisers sit near the bottom of this range",
     "low": 0.13, "high": 1.10, "step": 0.01, "value": 0.52, "units": "L/h"},
    {"key": "hr", "label": "Heart rate", "hint": "raising it shortens filling time",
     "low": 52.0, "high": 100.0, "step": 1.0, "value": 65.0, "units": "bpm"},
    {"key": "volume", "label": "Stressed blood volume", "hint": "preload; lowering it is the Valsalva maneuver",
     "low": 340.0, "high": 450.0, "step": 1.0, "value": 394.0, "units": "mL"},
)

# The emulator's own inputs. Note phi_eff, not phi and dose: the drug step is exact.
EMULATOR_INPUTS: tuple[tuple[str, float, float], ...] = (
    ("phi_eff", 0.11, 0.63),
    ("a_pas_kpa", 0.40, 5.50),
    ("wall_volume_ml", 100.0, 300.0),
    ("ref_cavity_volume_ml", 45.0, 80.0),
    ("heart_rate_bpm", 52.0, 100.0),
    ("total_blood_volume_ml", 340.0, 450.0),
)

READOUTS: tuple[tuple[str, str, str, int], ...] = (
    ("ejection_fraction", "Ejection fraction", "", 3),
    ("stroke_volume_ml", "Stroke volume", "mL", 1),
    ("edv_ml", "End-diastolic volume", "mL", 1),
    ("esv_ml", "End-systolic volume", "mL", 1),
    ("end_diastolic_pressure_mmhg", "Filling pressure", "mmHg", 1),
    ("peak_lvot_gradient_mmhg", "Peak outflow gradient", "mmHg", 1),
    ("wall_thickness_cm", "Wall thickness", "cm", 2),
    ("e_over_e_prime", "E/e' surrogate", "", 1),
    ("cardiac_output_l_per_min", "Cardiac output", "L/min", 2),
    ("atp_cost_per_stroke_work", "ATP per unit work", "AU/J", 0),
)


def build_emulator(
    n_design: int = 2600,
    seed: int = 20260816,
    constants: ModelConstants | None = None,
) -> tuple[Surrogate, list[str], dict[str, float]]:
    """Fit the emulator the page ships with, over the slider ranges.

    One cohort solve for the whole design. Returns the surrogate, its output names, and
    its held-out accuracy, which the page displays.
    """
    constants = constants or ModelConstants()
    lows = np.array([low for _, low, _ in EMULATOR_INPUTS])
    highs = np.array([high for _, _, high in EMULATOR_INPUTS])
    design = latin_hypercube(lows, highs, n_design, seed=seed)
    logger.info("emulator design: %d points over %d inputs", n_design, len(EMULATOR_INPUTS))

    phi_eff, a_pas, wall, cavity, heart_rate, volume = design.T
    # phi_eff is the emulator's input, so the cohort is run with the drug switched off and
    # phi set directly to the effective value.
    summary, _, _, converged, beats = simulate_cohort(
        wall_volume_ml=wall,
        ref_cavity_volume_ml=cavity,
        phi_baseline=phi_eff,
        a_pas_kpa=a_pas,
        b_pas=np.full(n_design, d.B_PAS),
        ca50_ref_um=np.full(n_design, d.CA50_REF_UM),
        clearance_l_per_h=np.full(n_design, d.DRUG_CL_L_PER_H),
        heart_rate_bpm=heart_rate,
        total_blood_volume_ml=volume,
        systemic_resistance=np.full(n_design, d.R_SYS_MMHG_S_PER_ML),
        dose_mg_per_day=0.0,
        constants=constants,
    )
    if not converged:
        logger.warning("emulator design did not fully converge after %d beats", beats)

    loops = _loop_traces(design, constants)
    fields = observe_arrays(
        summary, wall, np.full(n_design, d.BSA_M2), heart_rate
    )
    scalar_names = [name for name, _, _, _ in READOUTS]
    scalars = np.column_stack([fields[name] for name in scalar_names])

    outputs = np.hstack([loops, scalars])
    names = (
        [f"volume_{i}" for i in range(LOOP_POINTS)]
        + [f"pressure_{i}" for i in range(LOOP_POINTS)]
        + scalar_names
    )
    surrogate = Surrogate(names=tuple(names), degree=3).fit(np.log(design), outputs, seed=seed)
    accuracy = surrogate.error_report()
    logger.info("emulator held-out accuracy: %s", accuracy)
    return surrogate, names, accuracy


def _loop_traces(design: np.ndarray, constants: ModelConstants) -> np.ndarray:
    """Resampled pressure-volume loops for every design point.

    The cohort solver does not record traces (storing them for thousands of patients would
    cost gigabytes), so the loops are collected in chunks with the scalar path. This is the
    slow part of building the page and it runs once.
    """
    from ..model import simulate
    from ..parameters import HiddenMaterial, Loading, MeasuredGeometry

    index = np.linspace(0, constants.steps_per_beat - 1, LOOP_POINTS).astype(int)
    volumes = np.empty((len(design), LOOP_POINTS))
    pressures = np.empty((len(design), LOOP_POINTS))
    for row, (phi_eff, a_pas, wall, cavity, heart_rate, volume) in enumerate(design):
        result = simulate(
            MeasuredGeometry(wall_volume_ml=wall, ref_cavity_volume_ml=cavity),
            HiddenMaterial(
                phi_baseline=phi_eff, a_pas_kpa=a_pas, b_pas=d.B_PAS,
                ca50_ref_um=d.CA50_REF_UM, clearance_l_per_h=d.DRUG_CL_L_PER_H,
            ),
            Loading(heart_rate, volume, d.R_SYS_MMHG_S_PER_ML),
            0.0,
            constants=constants,
            record_trace=True,
        )
        trace = result.trace
        assert trace is not None
        volumes[row] = trace.cavity_volume_ml[index]
        pressures[row] = trace.lv_pressure_mmhg[index]
        if row % 400 == 0:
            logger.info("  loop traces: %d/%d", row, len(design))
    return np.hstack([volumes, pressures])


def _emulator_payload(surrogate: Surrogate, names: list[str], accuracy: dict[str, float]) -> str:
    def compact(array: np.ndarray) -> list:  # type: ignore[type-arg]
        return [round(float(v), 6) for v in np.asarray(array).ravel()]

    payload = {
        "inputs": [name for name, _, _ in EMULATOR_INPUTS],
        "mean": compact(surrogate._mean),  # noqa: SLF001
        "scale": compact(surrogate._scale),  # noqa: SLF001
        "powers": [[int(p) for p in row] for row in surrogate._powers],  # noqa: SLF001
        "coef": compact(surrogate._coef),  # noqa: SLF001
        "intercept": compact(surrogate._intercept),  # noqa: SLF001
        "nOutputs": len(names),
        "loopPoints": LOOP_POINTS,
        "names": names,
        "accuracy": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in accuracy.items()},
        "drug": {
            "eMax": d.DRUG_E_MAX,
            "ec50": d.DRUG_EC50_NG_PER_ML,
            "ngPerMg": 1.0e6,
            "hoursPerDay": 24.0,
            "mlPerL": 1.0e3,
        },
        "efThreshold": d.EF_INTERRUPTION_THRESHOLD,
        "obstructionThreshold": d.LVOT_OBSTRUCTIVE_THRESHOLD_MMHG,
        "sliders": [dict(s) for s in SLIDERS],
        "readouts": [
            {"key": k, "label": label, "units": units, "digits": digits}
            for k, label, units, digits in READOUTS
        ],
    }
    return json.dumps(payload, separators=(",", ":"))


def build(
    output: Path,
    labelled: pd.DataFrame | None = None,
    n_design: int = 2600,
    seed: int = 20260816,
) -> Path:
    """Write the self-contained explorer."""
    surrogate, names, accuracy = build_emulator(n_design=n_design, seed=seed)
    payload = _emulator_payload(surrogate, names, accuracy)

    cohort_note = ""
    if labelled is not None and "trial_eligible" in labelled:
        eligible = labelled[labelled["trial_eligible"]]
        if len(eligible):
            cohort_note = (
                f"Virtual cohort behind the analysis: {len(labelled):,} sampled, "
                f"{len(eligible):,} trial-eligible, "
                f"{100 * eligible['over_responder'].mean():.1f}% crossed the "
                "ejection-fraction floor at or below the mid dose."
            )

    html = _TEMPLATE.replace("__PAYLOAD__", payload).replace("__COHORT_NOTE__", cohort_note)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("explorer written to %s (%.0f kB)", output, output.stat().st_size / 1024)
    return output


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HCM digital twin: explorer</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb;
    --panel: #f5f4f1;
    --line: #e6e5e1;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --ink-3: #8a8880;
    --a: #2a78d6;
    --b: #eb6834;
    --good: #008300;
    --bad: #e34948;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19; --panel: #232322; --line: #383835;
      --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8a8880;
      --a: #3987e5; --b: #d95926; --good: #4caf50; --bad: #e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #232322; --line: #383835;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8a8880;
    --a: #3987e5; --b: #d95926; --good: #4caf50; --bad: #e66767;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface); color: var(--ink);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 28px 20px 56px; }
  h1 { font-size: 21px; margin: 0 0 6px; letter-spacing: -0.01em; }
  .lede { color: var(--ink-2); max-width: 68ch; margin: 0 0 4px; }
  .meta { color: var(--ink-3); font-size: 12px; margin: 10px 0 22px; max-width: 78ch; }
  .toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 18px; }
  button {
    font: inherit; font-size: 13px; padding: 7px 13px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer;
  }
  button:hover { border-color: var(--ink-3); }
  button[aria-pressed="true"] { background: var(--a); border-color: var(--a); color: #fff; }
  .grid { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 22px; align-items: start; }
  @media (max-width: 940px) { .grid { grid-template-columns: minmax(0, 1fr); } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em;
             color: var(--ink-3); margin: 0 0 12px; font-weight: 600; }
  svg { width: 100%; height: auto; display: block; overflow: visible; }
  .axis { stroke: var(--line); stroke-width: 1; }
  .grid-line { stroke: var(--line); stroke-width: 0.7; }
  .tick { fill: var(--ink-3); font-size: 10px; }
  .axis-label { fill: var(--ink-2); font-size: 11px; }
  .loop { fill: none; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .panels.single { grid-template-columns: 1fr; }
  .sl { margin-bottom: 13px; }
  .sl label { display: flex; justify-content: space-between; gap: 8px; font-size: 12px;
              color: var(--ink-2); margin-bottom: 3px; }
  .sl .val { font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 600; }
  .sl .hint { display: block; font-size: 11px; color: var(--ink-3); margin-top: 2px; line-height: 1.35; }
  input[type=range] { width: 100%; accent-color: var(--a); }
  .panel-b input[type=range] { accent-color: var(--b); }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); font-size: 12.5px; }
  th { color: var(--ink-3); font-weight: 600; font-size: 11px; text-transform: uppercase;
       letter-spacing: 0.05em; }
  td.num { text-align: right; }
  td.a { color: var(--a); font-weight: 600; }
  td.b { color: var(--b); font-weight: 600; }
  .flag { font-size: 11px; padding: 1px 6px; border-radius: 5px; font-weight: 600; }
  .flag.bad { background: var(--bad); color: #fff; }
  .flag.ok { background: transparent; color: var(--ink-3); border: 1px solid var(--line); }
  footer { margin-top: 30px; color: var(--ink-3); font-size: 11.5px; max-width: 80ch; }
  footer strong { color: var(--ink-2); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Hypertrophic cardiomyopathy digital twin</h1>
  <p class="lede">
    Two patients can share an ejection fraction and diverge completely under the same dose.
    Turn the sliders and watch which properties are visible in the loop and which are not.
  </p>
  <p class="meta" id="meta"></p>

  <div class="toolbar">
    <button id="compare" aria-pressed="false">Compare two patients</button>
    <button data-preset="healthy">Healthy</button>
    <button data-preset="hcm">HCM, untreated</button>
    <button data-preset="twins">The two twins</button>
    <button id="theme">Toggle theme</button>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Pressure-volume loop</h2>
      <svg id="chart" viewBox="0 0 640 440" role="img"
           aria-label="Left-ventricular pressure against volume"></svg>
      <div style="margin-top:12px">
        <table>
          <thead><tr><th>Measurement</th><th class="num" id="hdrA"></th>
          <th class="num" id="hdrB"></th></tr></thead>
          <tbody id="readouts"></tbody>
        </table>
      </div>
    </div>

    <div class="panels single" id="panels">
      <div class="card panel-a" data-panel="a">
        <h2><span class="swatch" style="background:var(--a)"></span>Patient A</h2>
        <div class="controls"></div>
      </div>
      <div class="card panel-b" data-panel="b" hidden>
        <h2><span class="swatch" style="background:var(--b)"></span>Patient B</h2>
        <div class="controls"></div>
      </div>
    </div>
  </div>

  <footer id="footer"></footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(() => {
  const P = JSON.parse(document.getElementById('payload').textContent);
  const nTerms = P.powers.length, nOut = P.nOutputs, L = P.loopPoints;

  // --- the emulator: standardise, build monomials, dot with the coefficients ----------
  function predict(x) {
    const z = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) z[i] = (Math.log(x[i]) - P.mean[i]) / P.scale[i];
    const out = new Float64Array(nOut);
    for (let k = 0; k < nOut; k++) out[k] = P.intercept[k];
    for (let t = 0; t < nTerms; t++) {
      let m = 1;
      const pw = P.powers[t];
      for (let i = 0; i < pw.length; i++) {
        const e = pw[i];
        if (e === 1) m *= z[i];
        else if (e === 2) m *= z[i] * z[i];
        else if (e === 3) m *= z[i] * z[i] * z[i];
        else if (e !== 0) m *= Math.pow(z[i], e);
      }
      if (m === 0) continue;
      const base = t * nOut;
      for (let k = 0; k < nOut; k++) out[k] += m * P.coef[base + k];
    }
    return out;
  }

  // The drug step is exact, not emulated: dose enters only through phi_eff.
  function effectivePhi(phi, dose, clearance) {
    const c = dose * P.drug.ngPerMg / (clearance * P.drug.hoursPerDay * P.drug.mlPerL);
    return phi * (1 - P.drug.eMax * c / (P.drug.ec50 + c));
  }

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const bounds = {};
  P.inputs.forEach((name, i) => { bounds[name] = i; });

  function evaluate(s) {
    const phiEff = clamp(effectivePhi(s.phi, s.dose, s.clearance), 0.111, 0.629);
    const raw = predict([phiEff, s.a_pas, s.wall, s.cavity, s.hr, s.volume]);
    const volume = Array.from(raw.slice(0, L));
    const pressure = Array.from(raw.slice(L, 2 * L));
    const scalars = {};
    P.readouts.forEach((r, i) => { scalars[r.key] = raw[2 * L + i]; });
    return { volume, pressure, scalars, phiEff, concentration:
      s.dose * P.drug.ngPerMg / (s.clearance * P.drug.hoursPerDay * P.drug.mlPerL) };
  }

  // --- state ------------------------------------------------------------------------
  const defaults = {};
  P.sliders.forEach(s => { defaults[s.key] = s.value; });
  const PRESETS = {
    healthy: { a: { phi: 0.35, a_pas: 0.90, wall: 140, cavity: 60, dose: 0, clearance: 0.52,
                    hr: 65, volume: 394 } },
    hcm:     { a: { ...defaults } },
    twins:   { a: { phi: 0.55, a_pas: 4.00, wall: 245, cavity: 60, dose: 0, clearance: 0.52,
                    hr: 65, volume: 394 },
               b: { phi: 0.55, a_pas: 4.00, wall: 245, cavity: 60, dose: 0, clearance: 0.18,
                    hr: 65, volume: 394 }, compare: true },
  };
  const state = { a: { ...defaults }, b: { ...defaults, clearance: 0.18 }, compare: false };

  // --- controls ---------------------------------------------------------------------
  function buildControls(panel) {
    const which = panel.dataset.panel;
    const host = panel.querySelector('.controls');
    host.innerHTML = '';
    P.sliders.forEach(s => {
      const wrap = document.createElement('div');
      wrap.className = 'sl';
      wrap.innerHTML =
        `<label for="${which}-${s.key}"><span>${s.label}</span>` +
        `<span class="val" id="v-${which}-${s.key}"></span></label>` +
        `<input type="range" id="${which}-${s.key}" min="${s.low}" max="${s.high}" ` +
        `step="${s.step}" value="${state[which][s.key]}">` +
        `<span class="hint">${s.hint}</span>`;
      host.appendChild(wrap);
      wrap.querySelector('input').addEventListener('input', ev => {
        state[which][s.key] = parseFloat(ev.target.value);
        render();
      });
    });
  }

  function syncControls() {
    ['a', 'b'].forEach(which => {
      P.sliders.forEach(s => {
        const input = document.getElementById(`${which}-${s.key}`);
        if (!input) return;
        input.value = state[which][s.key];
        const digits = s.step < 0.01 ? 3 : (s.step < 1 ? 2 : 0);
        document.getElementById(`v-${which}-${s.key}`).textContent =
          state[which][s.key].toFixed(digits) + (s.units ? ' ' + s.units : '');
      });
    });
  }

  // --- chart ------------------------------------------------------------------------
  const SVG = 'http://www.w3.org/2000/svg';
  function el(name, attrs, text) {
    const node = document.createElementNS(SVG, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function drawChart(series) {
    const svg = document.getElementById('chart');
    svg.innerHTML = '';
    const W = 640, H = 440, m = { l: 58, r: 18, t: 14, b: 46 };
    let vmax = 10, pmax = 10;
    series.forEach(s => {
      vmax = Math.max(vmax, ...s.volume);
      pmax = Math.max(pmax, ...s.pressure);
    });
    vmax = Math.ceil(vmax / 20) * 20 + 10;
    pmax = Math.ceil(pmax / 20) * 20 + 10;
    const X = v => m.l + (v / vmax) * (W - m.l - m.r);
    const Y = p => H - m.b - (p / pmax) * (H - m.t - m.b);

    for (let v = 0; v <= vmax; v += Math.max(20, Math.round(vmax / 8 / 20) * 20)) {
      svg.appendChild(el('line', { x1: X(v), y1: m.t, x2: X(v), y2: H - m.b, class: 'grid-line' }));
      svg.appendChild(el('text', { x: X(v), y: H - m.b + 16, class: 'tick',
                                   'text-anchor': 'middle' }, v));
    }
    for (let p = 0; p <= pmax; p += Math.max(20, Math.round(pmax / 8 / 20) * 20)) {
      svg.appendChild(el('line', { x1: m.l, y1: Y(p), x2: W - m.r, y2: Y(p), class: 'grid-line' }));
      svg.appendChild(el('text', { x: m.l - 8, y: Y(p) + 3.5, class: 'tick',
                                   'text-anchor': 'end' }, p));
    }
    svg.appendChild(el('line', { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b, class: 'axis' }));
    svg.appendChild(el('line', { x1: m.l, y1: m.t, x2: m.l, y2: H - m.b, class: 'axis' }));
    svg.appendChild(el('text', { x: (m.l + W - m.r) / 2, y: H - 8, class: 'axis-label',
                                 'text-anchor': 'middle' }, 'Left-ventricular volume (mL)'));
    svg.appendChild(el('text', { x: 14, y: (m.t + H - m.b) / 2, class: 'axis-label',
                                 'text-anchor': 'middle',
                                 transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` },
                       'Left-ventricular pressure (mmHg)'));

    series.forEach(s => {
      const pts = s.volume.map((v, i) => `${X(v).toFixed(1)},${Y(s.pressure[i]).toFixed(1)}`);
      pts.push(pts[0]);
      svg.appendChild(el('polyline', { points: pts.join(' '), class: 'loop', stroke: s.colour }));
      const iTop = s.pressure.indexOf(Math.max(...s.pressure));
      svg.appendChild(el('text', {
        x: X(s.volume[iTop]) + 8, y: Y(s.pressure[iTop]) - 6,
        fill: s.colour, 'font-size': 12, 'font-weight': 600,
      }, s.label));
    });
  }

  // --- readouts ---------------------------------------------------------------------
  function fmt(value, digits) {
    if (!isFinite(value)) return '—';
    return value.toFixed(digits);
  }

  function drawReadouts(a, b) {
    document.getElementById('hdrA').textContent = 'Patient A';
    document.getElementById('hdrB').textContent = state.compare ? 'Patient B' : '';
    const body = document.getElementById('readouts');
    body.innerHTML = '';
    P.readouts.forEach(r => {
      const tr = document.createElement('tr');
      let flag = '';
      if (r.key === 'ejection_fraction') {
        const bad = a.scalars[r.key] < P.efThreshold ||
                    (state.compare && b.scalars[r.key] < P.efThreshold);
        flag = bad ? ' <span class="flag bad">below 0.50</span>' : '';
      }
      tr.innerHTML =
        `<td>${r.label}${r.units ? ' <span style="color:var(--ink-3)">(' + r.units + ')</span>' : ''}${flag}</td>` +
        `<td class="num a">${fmt(a.scalars[r.key], r.digits)}</td>` +
        `<td class="num b">${state.compare ? fmt(b.scalars[r.key], r.digits) : ''}</td>`;
      body.appendChild(tr);
    });
    const extra = document.createElement('tr');
    extra.innerHTML =
      `<td>Steady-state concentration <span style="color:var(--ink-3)">(ng/mL)</span></td>` +
      `<td class="num a">${fmt(a.concentration, 0)}</td>` +
      `<td class="num b">${state.compare ? fmt(b.concentration, 0) : ''}</td>`;
    body.appendChild(extra);
    const phi = document.createElement('tr');
    phi.innerHTML =
      `<td>Effective myosin availability <span style="color:var(--ink-3)">(hidden)</span></td>` +
      `<td class="num a">${fmt(a.phiEff, 3)}</td>` +
      `<td class="num b">${state.compare ? fmt(b.phiEff, 3) : ''}</td>`;
    body.appendChild(phi);
  }

  function render() {
    const a = evaluate(state.a);
    const b = evaluate(state.b);
    const series = [{ ...a, colour: getComputedStyle(document.documentElement)
                        .getPropertyValue('--a').trim(), label: state.compare ? 'A' : '' }];
    if (state.compare) {
      series.push({ ...b, colour: getComputedStyle(document.documentElement)
                      .getPropertyValue('--b').trim(), label: 'B' });
    }
    drawChart(series);
    drawReadouts(a, b);
    syncControls();
  }

  // --- wiring -------------------------------------------------------------------------
  document.querySelectorAll('[data-panel]').forEach(buildControls);
  document.getElementById('compare').addEventListener('click', ev => {
    state.compare = !state.compare;
    ev.target.setAttribute('aria-pressed', String(state.compare));
    document.querySelector('[data-panel="b"]').hidden = !state.compare;
    document.getElementById('panels').classList.toggle('single', !state.compare);
    render();
  });
  document.querySelectorAll('[data-preset]').forEach(button => {
    button.addEventListener('click', () => {
      const preset = PRESETS[button.dataset.preset];
      Object.assign(state.a, preset.a);
      if (preset.b) Object.assign(state.b, preset.b);
      state.compare = !!preset.compare;
      document.getElementById('compare').setAttribute('aria-pressed', String(state.compare));
      document.querySelector('[data-panel="b"]').hidden = !state.compare;
      document.getElementById('panels').classList.toggle('single', !state.compare);
      render();
    });
  });
  document.getElementById('theme').addEventListener('click', () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
    render();
  });

  document.getElementById('meta').textContent = '__COHORT_NOTE__';
  document.getElementById('footer').innerHTML =
    '<strong>How this page works.</strong> Dragging a slider does not run the model. The ' +
    'page carries a polynomial emulator fitted to ' + P.names.length + ' outputs of the real ' +
    'simulator over these slider ranges, and evaluates it in the browser. Worst held-out ' +
    'coefficient of determination across all emulated outputs: <strong>' +
    P.accuracy.min_r2 + '</strong> (median ' + P.accuracy.median_r2 + ', worst output ' +
    P.accuracy.worst_output + '). The drug step is computed exactly rather than emulated. ' +
    '<br><br><strong>Try this.</strong> Press "The two twins". Both patients have identical ' +
    'geometry, identical tissue and identical loops. The only difference is drug clearance, ' +
    'which no measurement on this page can see. Now raise the dose on both and watch one of ' +
    'them cross the floor. That is the problem this project exists to study.' +
    '<br><br><strong>Limits.</strong> Prescribed calcium, a zero-dimensional chamber, ' +
    'steady-state pharmacokinetics, no fibrosis heterogeneity, no fiber disarray, no growth ' +
    'over time. Absolute values are illustrative; directional and relative behaviour is what ' +
    'this model is for.';

  render();
})();
</script>
</body>
</html>
"""
