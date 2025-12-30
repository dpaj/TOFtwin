#!/usr/bin/env python3
"""PyChop oracle: output ΔE(ω) for a given instrument configuration.

This script is meant to be called from Julia (TOFtwin) without requiring
any JSON parsing. It prints two whitespace-separated columns:

    etrans_meV   dE_fwhm_meV

where dE_fwhm_meV is the **FWHM** energy-transfer resolution (meV).

Internally we use PyChop to map configuration -> timing widths and distances,
then evaluate a standard explicit propagation formula (matching the
`cncs_pychop_compare.py` reference script).

Requirements (Python environment):
  - numpy
  - PyChop (https://github.com/mducle/pychop)

Example:
  python pychop_oracle_dE.py --instrument CNCS --Ei 12.0 --variant "High Flux" --freq 180,180 --etrans-min -1 --etrans-max 12 --npts 401
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np


FWHM_PER_SIGMA = 2.3548200450309493


def import_pychop_instrument():
    # Try a couple common import paths.
    try:
        import PyChop.Instruments as I
        return I
    except Exception:
        pass
    # Some installs may use lowercase package name
    try:
        import pychop.Instruments as I  # type: ignore
        return I
    except Exception:
        pass
    raise ImportError(
        "Could not import PyChop.Instruments. Ensure PyChop is installed in the Python environment used by this script."
    )


def energy_resolution_explicit_sigma_meV(
    Ei_meV: float,
    etrans_meV: float,
    delta_tp_s: float,
    delta_tc_s: float,
    delta_td_s: float,
    L1_m: float,
    L2_m: float,
    L3_m: float,
) -> float:
    """Return σ_E (NOT FWHM) in meV.

    This matches the explicit model used in `cncs_pychop_compare.py`.
    """

    # constants
    mn = 1.67492749804e-27
    meV = 1e-3 * 1.602176634e-19

    Ei = Ei_meV * meV
    dE = etrans_meV * meV
    Ef = Ei - dE
    if Ef <= 0:
        return float("nan")

    vi = math.sqrt(2.0 * Ei / mn)
    vf = math.sqrt(2.0 * Ef / mn)

    # time widths at sample / det from incident+chopper+detector contributions
    # (this is exactly as in the reference script)
    # Note: In the script's notation:
    #   L1 = x0 - xm
    #   L2 = x1
    #   L3 = x2
    # where x2 is sample->detector.

    term1 = (vi ** 3 / L1_m) ** 2 * (delta_tp_s ** 2)
    term2 = (vi ** 3 * L2_m / (L1_m ** 2)) ** 2 * (delta_tc_s ** 2)
    term3 = (vf ** 3 / L3_m) ** 2 * (delta_td_s ** 2)

    sigma_E_J = mn * math.sqrt(term1 + term2 + term3)
    return sigma_E_J / meV


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--instrument", required=True, help="Instrument name, e.g. CNCS or SEQUOIA")
    p.add_argument("--Ei", type=float, required=True, help="Incident energy Ei in meV")
    p.add_argument("--variant", default="", help="Instrument variant string (PyChop)")
    p.add_argument("--freq", default="", help="Comma-separated chopper frequency list, e.g. '180,180'")
    p.add_argument("--tc-index", type=int, default=0, help="Index into dtc2 list for chopper term")
    p.add_argument("--use-tc-rss", action="store_true", help="Use RSS of dtc2[tc_index] instead of first component")
    p.add_argument("--delta-td-us", type=float, default=0.0, help="Extra detector time σ (microseconds)")
    p.add_argument("--etrans-min", type=float, required=True)
    p.add_argument("--etrans-max", type=float, required=True)
    p.add_argument("--npts", type=int, default=401)

    args = p.parse_args(argv)

    I = import_pychop_instrument()
    inst = I.Instrument(args.instrument)

    if args.variant:
        inst.setVariant(args.variant)

    if args.freq:
        freq = [float(x) for x in args.freq.split(",") if x.strip()]
        if freq:
            inst.setFrequency(freq)

    Ei = float(args.Ei)

    # Pull PyChop timing model pieces
    dtm2 = float(inst.moderator.getWidthSquared(Ei))
    dtc2 = inst.chopper_system.getWidthSquared(Ei)

    # Distances (PyChop chopper system)
    x0, xa, x1, x2, xm = inst.chopper_system.getDistances()

    # Convert to the explicit formula distances
    L1 = float(x0 - xm)
    L2 = float(x1)
    L3 = float(x2)

    # Convert dtm2/dtc2 into σ terms
    delta_tp = math.sqrt(dtm2) * (1.0 - float(xm) / float(x0))

    tci = int(args.tc_index)
    if tci < 0 or tci >= len(dtc2):
        raise ValueError(f"tc-index out of range: {tci} (len(dtc2)={len(dtc2)})")

    if args.use_tc_rss:
        delta_tc = math.sqrt(sum(float(x) for x in dtc2[tci]))
    else:
        delta_tc = math.sqrt(float(dtc2[tci][0]))

    delta_td = float(args.delta_td_us) * 1e-6

    # Build ω grid
    wmin = float(args.etrans_min)
    wmax = float(args.etrans_max)
    npts = int(args.npts)
    if npts < 2:
        npts = 2
    ws = np.linspace(wmin, wmax, npts)

    # Compute σ_E and report FWHM
    dE_fwhm = np.empty_like(ws)
    for i, w in enumerate(ws):
        sigma = energy_resolution_explicit_sigma_meV(Ei, float(w), delta_tp, delta_tc, delta_td, L1, L2, L3)
        dE_fwhm[i] = float(sigma) * FWHM_PER_SIGMA

    # stdout: data table
    print("# etrans_meV dE_fwhm_meV")
    for w, de in zip(ws, dE_fwhm):
        if math.isfinite(de):
            print(f"{w:.8g} {de:.8g}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        # allow piping into head, etc.
        raise
