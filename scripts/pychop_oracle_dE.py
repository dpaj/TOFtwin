#!/usr/bin/env python3
"""PyChop oracle: output ΔE(ω) for a given instrument configuration.

This script is meant to be called from Julia (TOFtwin). It prints two
whitespace-separated columns:

    etrans_meV   dE_fwhm_meV

where dE_fwhm_meV is PyChop's ΔE(etrans) in meV.

Two modes:

1) CLI mode (human-friendly):
   python pychop_oracle_dE.py --instrument CNCS --Ei 12.0 --variant "High Flux" \
       --freq 180,180 --etrans-min -1 --etrans-max 12 --npts 401

2) JSON mode (TOFtwin):
   python pychop_oracle_dE.py --json cfg.json --out out.txt

JSON schema:
  instrument:  "CNCS" | "SEQUOIA" | ...
  variant:     string (optional)
  Ei:          float (meV)
  etrans_min:  float (meV)
  etrans_max:  float (meV)
  npts:        int
  tc_index:    int
  use_tc_rss:  bool
  delta_td_us: float (μs)
  freq_hz:     list[float]

Implementation notes:
- We *prefer* PyChop2.calculate(...) for ΔE(ω) (same as cncs_pychop_compare.py).
- If PyChop2 isn't importable, we fall back to the explicit timing-propagation
  formula from cncs_pychop_compare.py using moderator/chopper timing variances.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Iterable, List, Optional, Sequence, Union

import numpy as np

# ----------------------------
# Physical constants (SI)
# ----------------------------
m_n = 1.67492749804e-27  # neutron mass [kg]
meV_to_J = 1.602176634e-22  # 1 meV in Joules
J_to_meV = 1.0 / meV_to_J


# ----------------------------
# PyChop import helpers
# ----------------------------
def import_pychop2():
    """Try a few plausible module paths for PyChop2; return symbol or None."""
    candidates = [
        ("PyChop", "PyChop2"),
        ("PyChop.PyChop", "PyChop2"),
        ("PyChop.PyChop2", "PyChop2"),
        ("pychop", "PyChop2"),
        ("pychop.PyChop", "PyChop2"),
        ("pychop.PyChop2", "PyChop2"),
    ]
    for modname, sym in candidates:
        try:
            mod = __import__(modname, fromlist=[sym])
            obj = getattr(mod, sym, None)
            if obj is not None:
                return obj
        except Exception:
            pass
    return None



def compute_pychop2_dE_meV(
    inst_name: str,
    variant: Optional[str],
    freq: Optional[Union[float, Sequence[float]]],
    Ei_meV: float,
    etrans_meV: np.ndarray,
) -> Optional[np.ndarray]:
    """Return PyChop2 ΔE(etrans) in meV, or None if unavailable."""
    PyChop2 = import_pychop2()
    if PyChop2 is None:
        return None

    chtyp = variant if (variant and variant.strip()) else None

    # Prefer the documented static calculate interface.
    if hasattr(PyChop2, "calculate"):
        for inst_try in (inst_name.lower(), inst_name):
            try:
                # Mantid/PyChop typically returns (res, flux)
                res, _flux = PyChop2.calculate(inst=inst_try, chtyp=chtyp, freq=freq, ei=Ei_meV, etrans=etrans_meV)
                return np.asarray(res, dtype=float)
            except Exception:
                pass

    # Fall back to an OO interface if present.
    try:
        obj = PyChop2(inst_name.lower())
    except Exception:
        try:
            obj = PyChop2(inst_name)
        except Exception:
            return None

    # Apply settings if possible.
    if chtyp is not None:
        for m in ("setChopper", "setChopperType"):
            if hasattr(obj, m):
                try:
                    getattr(obj, m)(chtyp)
                    break
                except Exception:
                    pass
    if freq is not None:
        for m in ("setFrequency",):
            if hasattr(obj, m):
                try:
                    getattr(obj, m)(freq)
                    break
                except Exception:
                    pass
    if hasattr(obj, "setEi"):
        try:
            obj.setEi(Ei_meV)
        except Exception:
            pass
    if hasattr(obj, "getResolution"):
        try:
            return np.asarray(obj.getResolution(etrans_meV), dtype=float)
        except Exception:
            pass

    return None



def import_pychop_instrument_class():
    """Return Instrument class from PyChop, trying common import paths."""
    try:
        from PyChop.Instruments import Instrument  # type: ignore
        return Instrument
    except Exception:
        from pychop.Instruments import Instrument  # type: ignore
        return Instrument


# ----------------------------
# Compatibility setters for Instrument APIs
# ----------------------------
def _call_method_compat(obj: Any, names: Iterable[str], *args: Any, **kwargs: Any) -> bool:
    """Try calling the first existing method in `names` on obj.
    Returns True if a call succeeded. For TypeError, tries next name.
    """
    for name in names:
        if not hasattr(obj, name):
            continue
        fn = getattr(obj, name)
        if not callable(fn):
            continue
        try:
            fn(*args, **kwargs)
            return True
        except TypeError:
            continue
    return False



def _set_variant_compat(inst: Any, variant: str) -> None:
    if not variant:
        return
    if _call_method_compat(inst, ["setVariant", "set_variant", "setvariant", "setConfiguration", "set_configuration"], variant):
        return
    if hasattr(inst, "chopper_system") and _call_method_compat(
        inst.chopper_system, ["setVariant", "set_variant", "setvariant", "setConfiguration", "set_configuration"], variant
    ):
        return
    for attr in ("variant", "configuration", "chtyp", "chopper"):
        if hasattr(inst, attr):
            try:
                setattr(inst, attr, variant)
                return
            except Exception:
                pass
    raise AttributeError("Could not set variant on PyChop Instrument (unknown API).")



def _set_frequency_compat(inst: Any, freq_list: Sequence[float]) -> None:
    if not freq_list:
        return

    # Some APIs want a scalar for single-frequency.
    freq_arg: Any = float(freq_list[0]) if len(freq_list) == 1 else list(map(float, freq_list))

    if _call_method_compat(inst, ["setFrequency", "set_frequency", "setFreq", "setfreq", "setfrequency"], freq_arg):
        return
    if hasattr(inst, "chopper_system") and _call_method_compat(
        inst.chopper_system, ["setFrequency", "set_frequency", "setFreq", "setfreq", "setfrequency"], freq_arg
    ):
        return
    # last resort attribute
    for attr in ("frequency", "freq", "freq_Hz"):
        try:
            setattr(inst, attr, freq_arg)
            return
        except Exception:
            pass
    raise AttributeError("Could not set frequency on PyChop Instrument (unknown API).")



def _parse_freq(freq_str: str) -> List[float]:
    if not freq_str:
        return []
    out: List[float] = []
    for tok in str(freq_str).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


# ----------------------------
# Explicit ΔE model (same as cncs_pychop_compare.py)
# ----------------------------
def energy_resolution_explicit_meV(
    Ei_meV: float,
    E_meV: Union[float, np.ndarray],
    L1_m: float,
    L2_m: float,
    L3_m: float,
    delta_tp_s: float,
    delta_tc_s: float,
    delta_td_s: float,
) -> Union[float, np.ndarray]:
    """Timing-propagation ΔE(Etransfer) in meV (matches cncs_pychop_compare.py)."""
    E = np.asarray(E_meV, dtype=float)
    Ef = Ei_meV - E
    out = np.full_like(E, np.nan, dtype=float)
    ok = Ef > 0

    Ei_J = Ei_meV * meV_to_J
    Ef_J = Ef[ok] * meV_to_J

    vi_cubed = (2.0 * Ei_J / m_n) ** (1.5)
    vf_cubed = (2.0 * Ef_J / m_n) ** (1.5)

    term_tp = (vi_cubed / L1_m + vf_cubed * L2_m / (L1_m * L3_m)) ** 2 * (delta_tp_s**2)
    term_tc = (vi_cubed / L1_m + vf_cubed * (L1_m + L2_m) / (L1_m * L3_m)) ** 2 * (delta_tc_s**2)
    term_td = (vf_cubed / L3_m) ** 2 * (delta_td_s**2)

    delta_E_J = m_n * np.sqrt(term_tp + term_tc + term_td)
    out[ok] = delta_E_J * J_to_meV
    return out if isinstance(E_meV, np.ndarray) else float(out.item())



def _get_width2_list(x: Any) -> np.ndarray:
    """Normalize PyChop getWidthSquared return types to 1D float array."""
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, (float, int, np.floating, np.integer)):
        return np.array([float(x)], dtype=float)
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float).ravel()
    try:
        return np.asarray(list(x), dtype=float).ravel()
    except Exception:
        return np.array([float(x)], dtype=float)


# ----------------------------
# CLI + main
# ----------------------------
def _load_json_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def _get_required(cfg: dict, key: str) -> Any:
    if key not in cfg:
        raise ValueError(f"JSON missing required key: {key}")
    return cfg[key]



def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser()

    # JSON mode (TOFtwin)
    p.add_argument("--json", type=str, default=None, help="Read inputs from JSON file (TOFtwin mode)")
    p.add_argument("--out", type=str, default=None, help="Write output table to file (default: stdout)")

    # CLI mode (human)
    p.add_argument("--instrument", type=str, default=None, help="Instrument name (e.g. CNCS)")
    p.add_argument("--Ei", type=float, default=None, help="Incident energy (meV)")
    p.add_argument("--variant", type=str, default="", help="Variant string (PyChop)")
    p.add_argument("--freq", type=str, default="", help="Comma-separated chopper frequencies (Hz)")
    p.add_argument("--tc-index", type=int, default=0, help="Index into dtc2 list for chopper term")
    p.add_argument("--use-tc-rss", action="store_true", help="Use RSS of dtc2 components")
    p.add_argument("--delta-td-us", type=float, default=0.0, help="Extra detector timing uncertainty (μs)")
    p.add_argument("--etrans-min", type=float, default=None, help="Min energy transfer ω (meV)")
    p.add_argument("--etrans-max", type=float, default=None, help="Max energy transfer ω (meV)")
    p.add_argument("--npts", type=int, default=401, help="Number of ω points")
    p.add_argument("--no-pychop2", action="store_true", help="Disable PyChop2 and force explicit fallback")

    args = p.parse_args(argv)

    if args.json:
        cfg = _load_json_cfg(args.json)
        instrument = str(_get_required(cfg, "instrument"))
        variant = str(cfg.get("variant", "") or "")
        Ei = float(_get_required(cfg, "Ei"))
        etrans_min = float(_get_required(cfg, "etrans_min"))
        etrans_max = float(_get_required(cfg, "etrans_max"))
        npts = int(cfg.get("npts", 401))
        tc_index = int(cfg.get("tc_index", 0))
        use_tc_rss = bool(cfg.get("use_tc_rss", False))
        delta_td_us = float(cfg.get("delta_td_us", 0.0))
        freq_list = [float(x) for x in cfg.get("freq_hz", [])]
    else:
        if args.instrument is None or args.Ei is None or args.etrans_min is None or args.etrans_max is None:
            p.error("the following arguments are required: --instrument, --Ei, --etrans-min, --etrans-max")
            return 2
        instrument = str(args.instrument)
        variant = str(args.variant or "")
        Ei = float(args.Ei)
        etrans_min = float(args.etrans_min)
        etrans_max = float(args.etrans_max)
        npts = int(args.npts)
        tc_index = int(args.tc_index)
        use_tc_rss = bool(args.use_tc_rss)
        delta_td_us = float(args.delta_td_us)
        freq_list = _parse_freq(args.freq)

    # Build ω grid
    npts = max(2, int(npts))
    ws = np.linspace(float(etrans_min), float(etrans_max), npts)

    # 1) Prefer PyChop2 ΔE
    freq_for_pychop2: Optional[Union[float, Sequence[float]]] = None
    if freq_list:
        freq_for_pychop2 = float(freq_list[0]) if len(freq_list) == 1 else freq_list

    dE: Optional[np.ndarray]
    if not args.no_pychop2:
        dE = compute_pychop2_dE_meV(instrument, variant, freq_for_pychop2, Ei, ws)
    else:
        dE = None

    # 2) Fallback: explicit formula using timing variances from Instrument
    if dE is None:
        Instrument = import_pychop_instrument_class()
        inst = Instrument(instrument)

        if variant.strip():
            _set_variant_compat(inst, variant)
        if freq_list:
            _set_frequency_compat(inst, freq_list)

        dtm2 = float(inst.moderator.getWidthSquared(Ei))
        dtc2 = _get_width2_list(inst.chopper_system.getWidthSquared(Ei))

        x0, xa, x1, x2, xm = inst.chopper_system.getDistances()
        x0 = float(x0)
        xm = float(xm)

        L1 = float(x0 - xm)
        L2 = float(x1)
        L3 = float(x2)

        delta_tp = math.sqrt(dtm2) * (1.0 - (xm / x0))

        if dtc2.size == 0:
            raise RuntimeError("PyChop chopper_system.getWidthSquared(Ei) returned empty dtc2.")

        if use_tc_rss:
            delta_tc = math.sqrt(float(np.sum(dtc2)))
        else:
            if tc_index < 0 or tc_index >= dtc2.size:
                raise ValueError(f"tc-index out of range: {tc_index} (len(dtc2)={dtc2.size})")
            delta_tc = math.sqrt(float(dtc2[tc_index]))

        delta_td = float(delta_td_us) * 1e-6

        dE = np.asarray(energy_resolution_explicit_meV(Ei, ws, L1, L2, L3, delta_tp, delta_tc, delta_td), dtype=float)

    # Output
    lines = ["# etrans_meV dE_fwhm_meV"]
    for w, de in zip(ws, dE):
        if math.isfinite(float(de)):
            lines.append(f"{float(w):.8g} {float(de):.8g}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    else:
        sys.stdout.write("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
