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
  variant:     string (optional)   # aka "package" in newer Mantid PyChop2
  Ei:          float (meV)
  etrans_min:  float (meV)
  etrans_max:  float (meV)
  npts:        int
  tc_index:    int                 # used only by explicit fallback model
  use_tc_rss:  bool                # used only by explicit fallback model
  delta_td_us: float (μs)          # used only by explicit fallback model
  freq_hz:     list[float]         # passed to PyChop2 as frequency/freq

Implementation notes:
- We prefer Mantid's PyChop2.calculate(...) for ΔE(ω).
- Mantid has used different kwarg names across versions:
    older:  chtyp, freq
    newer:  package, frequency
  We try *both* spellings.
- If PyChop2 is unavailable or errors, we fall back to an explicit
  timing-propagation model using Instrument.getWidthSquared(...) terms.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    # Prefer Mantid's interface module first (as per Mantid docs).
    candidates = [
        ("mantidqtinterfaces.PyChop", "PyChop2"),
        ("mantidqtinterfaces.PyChop.PyChop2", "PyChop2"),
        ("mantidqtinterfaces.PyChop.PyChop", "PyChop2"),
        # Older / alternative installs
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


def _pychop2_origin(PyChop2: Any) -> Tuple[str, str]:
    """Return (module, file) for a PyChop2 symbol."""
    mod = getattr(PyChop2, "__module__", "") or ""
    try:
        path = inspect.getfile(PyChop2)
    except Exception:
        path = ""
    return mod, path


def compute_pychop2_dE_meV(
    inst_name: str,
    variant: Optional[str],
    freq: Optional[Union[float, Sequence[float]]],
    Ei_meV: float,
    etrans_meV: np.ndarray,
    *,
    debug: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Return (ΔE_FWHM array in meV, meta dict). If unavailable, returns (None, meta)."""
    meta: Dict[str, Any] = {"backend": None, "pychop2_module": None, "pychop2_file": None, "pychop2_call": None}
    PyChop2 = import_pychop2()
    if PyChop2 is None:
        return None, meta

    mod, path = _pychop2_origin(PyChop2)
    meta["pychop2_module"] = mod
    meta["pychop2_file"] = path

    package = variant if (variant and variant.strip()) else None

    # Helper to normalize return value: sometimes (res, flux), sometimes just res.
    def _extract_res(x: Any) -> Any:
        if isinstance(x, tuple) and len(x) >= 1:
            return x[0]
        return x

    errors: List[str] = []

    # Prefer the documented static calculate interface if present.
    if hasattr(PyChop2, "calculate"):
        for inst_try in (inst_name.lower(), inst_name):
            # Try multiple keyword spellings across Mantid versions.
            call_variants: List[Tuple[str, Dict[str, Any]]] = []

            base = {"inst": inst_try, "ei": Ei_meV, "etrans": etrans_meV}

            # Newer Mantid docs: package, frequency
            if package is not None:
                call_variants.append(("kw:package+frequency", {**base, "package": package, "frequency": freq}))
                call_variants.append(("kw:package+freq", {**base, "package": package, "freq": freq}))
            # Older spellings: chtyp, freq
            if package is not None:
                call_variants.append(("kw:chtyp+frequency", {**base, "chtyp": package, "frequency": freq}))
                call_variants.append(("kw:chtyp+freq", {**base, "chtyp": package, "freq": freq}))
            # Allow calls with no package/variant (defaults)
            call_variants.append(("kw:default+frequency", {**base, "frequency": freq}))
            call_variants.append(("kw:default+freq", {**base, "freq": freq}))

            # Drop None freq keys to avoid signatures that don't accept it.
            cleaned_variants: List[Tuple[str, Dict[str, Any]]] = []
            for name, kwargs in call_variants:
                kw2 = dict(kwargs)
                if kw2.get("frequency", "X") is None:
                    kw2.pop("frequency", None)
                if kw2.get("freq", "X") is None:
                    kw2.pop("freq", None)
                cleaned_variants.append((name, kw2))

            for call_name, kwargs in cleaned_variants:
                try:
                    out = PyChop2.calculate(**kwargs)
                    res = np.asarray(_extract_res(out), dtype=float)
                    meta["backend"] = "pychop2.calculate"
                    meta["pychop2_call"] = call_name
                    meta["inst_arg"] = inst_try
                    if debug:
                        meta["pychop2_kwargs"] = {k: ("<array>" if k == "etrans" else v) for k, v in kwargs.items()}
                        meta["pychop2_errors"] = errors
                    return res, meta
                except TypeError as e:
                    # signature mismatch (unexpected kw)
                    if debug:
                        errors.append(f"{inst_try} {call_name} TypeError: {e}")
                    continue
                except Exception as e:
                    if debug:
                        errors.append(f"{inst_try} {call_name} Exception: {e}")
                    continue

            # Positional fallbacks (very old signature): calculate(inst, package, freq, ei, etrans)
            if package is not None and freq is not None:
                try:
                    out = PyChop2.calculate(inst_try, package, freq, Ei_meV, etrans_meV)
                    res = np.asarray(_extract_res(out), dtype=float)
                    meta["backend"] = "pychop2.calculate"
                    meta["pychop2_call"] = "positional:inst,package,freq,ei,etrans"
                    meta["inst_arg"] = inst_try
                    if debug:
                        meta["pychop2_errors"] = errors
                    return res, meta
                except Exception as e:
                    if debug:
                        errors.append(f"{inst_try} positional Exception: {e}")

    # OO interface
    obj = None
    for inst_try in (inst_name.lower(), inst_name):
        try:
            obj = PyChop2(inst_try)
            meta["inst_arg"] = inst_try
            break
        except Exception as e:
            if debug:
                errors.append(f"OO ctor {inst_try} Exception: {e}")

    if obj is None:
        if debug:
            meta["pychop2_errors"] = errors
        return None, meta

    # Apply settings if possible.
    if package is not None:
        for m in ("setChopper", "setChopperType", "setChopperPackage", "setPackage"):
            if hasattr(obj, m):
                try:
                    getattr(obj, m)(package)
                    meta["pychop2_call"] = f"oo:{m}"
                    break
                except Exception as e:
                    if debug:
                        errors.append(f"OO {m}({package!r}) Exception: {e}")

    if freq is not None:
        for m in ("setFrequency", "setFrequencies"):
            if hasattr(obj, m):
                try:
                    getattr(obj, m)(freq)
                    meta["pychop2_freq_method"] = m
                    break
                except Exception as e:
                    if debug:
                        errors.append(f"OO {m}({freq!r}) Exception: {e}")

    if hasattr(obj, "setEi"):
        try:
            obj.setEi(Ei_meV)
        except Exception as e:
            if debug:
                errors.append(f"OO setEi({Ei_meV}) Exception: {e}")

    if hasattr(obj, "getResolution"):
        try:
            res = np.asarray(obj.getResolution(etrans_meV), dtype=float)
            meta["backend"] = "pychop2.oo"
            if debug:
                meta["pychop2_errors"] = errors
            return res, meta
        except Exception as e:
            if debug:
                errors.append(f"OO getResolution Exception: {e}")

    if debug:
        meta["pychop2_errors"] = errors
    return None, meta


def import_pychop_instrument_class():
    """Return Instrument class from PyChop, trying common import paths."""
    for modname in (
        "mantidqtinterfaces.PyChop.Instruments",
        "PyChop.Instruments",
        "pychop.Instruments",
    ):
        try:
            mod = __import__(modname, fromlist=["Instrument"])
            Instrument = getattr(mod, "Instrument", None)
            if Instrument is not None:
                return Instrument
        except Exception:
            pass
    raise ImportError("Could not import PyChop Instrument class.")


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

    # Many PyChop APIs call this "package" / "chopper package" or "chopper type".
    if _call_method_compat(
        inst,
        [
            "setVariant",
            "set_variant",
            "setvariant",
            "setConfiguration",
            "set_configuration",
            "setChopper",
            "setChopperType",
            "setChopperPackage",
            "setPackage",
            "set_package",
        ],
        variant,
    ):
        return

    if hasattr(inst, "chopper_system") and _call_method_compat(
        inst.chopper_system,
        [
            "setVariant",
            "set_variant",
            "setvariant",
            "setConfiguration",
            "set_configuration",
            "setChopper",
            "setChopperType",
            "setChopperPackage",
            "setPackage",
            "set_package",
        ],
        variant,
    ):
        return

    # Last resort: attribute assignment (may not trigger reconfiguration on some versions).
    for attr in ("variant", "configuration", "chtyp", "chopper", "package"):
        if hasattr(inst, attr):
            try:
                setattr(inst, attr, variant)
                return
            except Exception:
                pass

    raise AttributeError("Could not set variant/package on PyChop Instrument (unknown API).")


def _set_frequency_compat(inst: Any, freq_list: Sequence[float]) -> None:
    if not freq_list:
        return

    # Some APIs want a scalar for single-frequency.
    freq_arg: Any = float(freq_list[0]) if len(freq_list) == 1 else list(map(float, freq_list))

    if _call_method_compat(inst, ["setFrequency", "set_frequency", "setFreq", "setfreq", "setfrequency", "setFrequencies"], freq_arg):
        return
    if hasattr(inst, "chopper_system") and _call_method_compat(
        inst.chopper_system, ["setFrequency", "set_frequency", "setFreq", "setfreq", "setfrequency", "setFrequencies"], freq_arg
    ):
        return
    # last resort attribute
    for attr in ("frequency", "freq", "freq_Hz", "frequency_hz"):
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
# Distance mapping helper (CNCS/SEQUOIA-safe)
# ----------------------------
def choose_x_start(x0_m: float, xm_m: float, *, xm_margin_m: float = 0.5) -> Tuple[float, bool]:
    """Choose the 'incident start' distance used in the simplified timing propagation.

    PyChop instrument definitions often provide:
      - x0: moderator -> energy-defining chopper
      - xm: moderator -> upstream pulse-defining element (optional)

    CNCS-like "two-stage" definitions typically have xm < x0 by many meters.
    Some instruments/YAMLs (e.g. SEQUOIA in many PyChop versions) have xm == x0,
    meaning there is no distinct upstream element in the model; in that case we
    fall back to moderator start (x_start = 0).

    Returns:
      (x_start_m, use_xm_as_start)
    """
    use_xm_as_start = (xm_m > 0.0) and (x0_m > xm_m) and ((x0_m - xm_m) > xm_margin_m)
    return (xm_m if use_xm_as_start else 0.0), use_xm_as_start

# ----------------------------
# Explicit ΔE model
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
    p.add_argument("--variant", type=str, default="", help="Variant / package string (PyChop)")
    p.add_argument("--freq", type=str, default="", help="Comma-separated chopper frequencies (Hz)")
    p.add_argument("--tc-index", type=int, default=0, help="Index into dtc2 list for explicit model chopper term")
    p.add_argument("--use-tc-rss", action="store_true", help="Use RSS of dtc2 components (explicit model)")
    p.add_argument("--delta-td-us", type=float, default=0.0, help="Extra detector timing uncertainty (μs) (explicit model)")
    p.add_argument("--etrans-min", type=float, default=None, help="Min energy transfer ω (meV)")
    p.add_argument("--etrans-max", type=float, default=None, help="Max energy transfer ω (meV)")
    p.add_argument("--npts", type=int, default=401, help="Number of ω points")
    p.add_argument("--no-pychop2", action="store_true", help="Disable PyChop2 and force explicit fallback")
    p.add_argument("--debug", action="store_true", help="Print extra diagnostic header lines")

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
        debug = bool(cfg.get("debug", False)) or bool(args.debug)
        no_pychop2 = bool(cfg.get("no_pychop2", False)) or bool(args.no_pychop2)
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
        debug = bool(args.debug)
        no_pychop2 = bool(args.no_pychop2)

    # Build ω grid
    npts = max(2, int(npts))
    ws = np.linspace(float(etrans_min), float(etrans_max), npts)

    # Make a freq argument for PyChop2: scalar or list
    freq_for_pychop2: Optional[Union[float, Sequence[float]]] = None
    if freq_list:
        freq_for_pychop2 = float(freq_list[0]) if len(freq_list) == 1 else list(map(float, freq_list))

    # 1) Try PyChop2 ΔE
    dE: Optional[np.ndarray] = None
    meta: Dict[str, Any] = {}
    if not no_pychop2:
        dE, meta = compute_pychop2_dE_meV(instrument, variant, freq_for_pychop2, Ei, ws, debug=debug)

    backend_used = meta.get("backend") if meta else None

    # 2) Fallback: explicit formula using timing variances from Instrument
    explicit_diag: Dict[str, Any] = {}
    if dE is None:
        Instrument = import_pychop_instrument_class()

        # Some versions accept variant/package as ctor arg; try that first.
        inst = None
        if variant.strip():
            try:
                inst = Instrument(instrument, variant)
            except Exception:
                inst = None
        if inst is None:
            inst = Instrument(instrument)
            if variant.strip():
                _set_variant_compat(inst, variant)

        if freq_list:
            _set_frequency_compat(inst, freq_list)

        dtm2 = float(inst.moderator.getWidthSquared(Ei))
        dtc2 = _get_width2_list(inst.chopper_system.getWidthSquared(Ei))

        x0, xa, x1, x2, xm = inst.chopper_system.getDistances()
        x0 = float(x0); xa = float(xa); x1 = float(x1); x2 = float(x2); xm = float(xm)

        if x0 == 0.0:
            raise ValueError("x0 returned by PyChop is 0; cannot compute delta_tp scaling.")

        # Choose incident start (CNCS-like two-stage uses xm; SEQUOIA often falls back to moderator start)
        x_start, use_xm_as_start = choose_x_start(x0, xm, xm_margin_m=0.5)

        # Map to L1/L2/L3 used by the explicit timing-propagation formulas
        L1 = float(x0 - x_start)   # start -> energy-defining chopper
        L2 = float(x1)             # chopper -> sample
        L3 = float(x2)             # sample -> detector

        if not (L1 > 0.0 and L2 > 0.0 and L3 > 0.0):
            raise ValueError(
                f"Non-positive distance(s) for explicit model: L1={L1}, L2={L2}, L3={L3}. "
                f"(raw: x0={x0}, xm={xm}, x1={x1}, x2={x2}; x_start={x_start})"
            )

        # Moderator pulse contribution. In single-stage case (x_start=0): delta_tp = sqrt(dtm2).
        delta_tp = math.sqrt(dtm2) * (1.0 - (x_start / x0))

        if dtc2.size == 0:
            raise RuntimeError("PyChop chopper_system.getWidthSquared(Ei) returned empty dtc2.")

        if use_tc_rss:
            delta_tc = math.sqrt(float(np.nansum(dtc2)))
        else:
            if tc_index < 0 or tc_index >= dtc2.size:
                raise ValueError(f"tc-index out of range: {tc_index} (len(dtc2)={dtc2.size})")
            val = float(dtc2[tc_index])
            if not np.isfinite(val):
                finite = [float(v) for v in dtc2 if np.isfinite(v)]
                val = finite[0] if finite else float("nan")
            delta_tc = math.sqrt(val)

        delta_td = float(delta_td_us) * 1e-6

        dE = np.asarray(energy_resolution_explicit_meV(Ei, ws, L1, L2, L3, delta_tp, delta_tc, delta_td), dtype=float)

        backend_used = "explicit"
        explicit_diag = {
            "dtm2_s2": dtm2,
            "dtc2_s2": dtc2.tolist(),
            "distances_m": {"x0": float(x0), "xm": float(xm), "x_start": float(x_start), "use_xm_as_start": bool(use_xm_as_start), "x1": float(x1), "x2": float(x2), "xa": float(xa)},
            "L_m": {"L1": float(L1), "L2": float(L2), "L3": float(L3)},
            "delta_tp_us": float(delta_tp * 1e6),
            "delta_tc_us": float(delta_tc * 1e6),
            "delta_td_us": float(delta_td_us),
        }

    # Header (always include minimal provenance; more if debug)
    hdr: List[str] = []
    hdr.append("# pychop_oracle_dE")
    hdr.append(f"# instrument={instrument} Ei_meV={Ei:g} variant={variant!r} freq_hz={[float(x) for x in freq_list]}")
    hdr.append(f"# tc_index={tc_index} use_tc_rss={bool(use_tc_rss)} delta_td_us={delta_td_us:g}")
    hdr.append(f"# backend={backend_used or 'unknown'}")

    if meta and meta.get("pychop2_module"):
        hdr.append(f"# pychop2_module={meta.get('pychop2_module','')} pychop2_file={meta.get('pychop2_file','')}")
        if meta.get("pychop2_call"):
            hdr.append(f"# pychop2_call={meta.get('pychop2_call')}")
        if debug and meta.get("pychop2_kwargs"):
            hdr.append(f"# pychop2_kwargs={meta.get('pychop2_kwargs')}")
        if debug and meta.get("pychop2_errors"):
            for i, e in enumerate(meta["pychop2_errors"][:8]):
                hdr.append(f"# pychop2_error[{i}]={e}")

    if explicit_diag:
        hdr.append(f"# dtm2_s2={explicit_diag['dtm2_s2']:.8g} dtc2_s2={explicit_diag['dtc2_s2']}")
        d = explicit_diag["distances_m"]
        hdr.append(f"# distances_m x0={d['x0']:.6g} xm={d['xm']:.6g} x_start={d.get('x_start', float('nan')):.6g} x1={d['x1']:.6g} x2={d['x2']:.6g} xa={d['xa']:.6g}")
        L = explicit_diag["L_m"]
        hdr.append(f"# L_m L1={L['L1']:.6g} L2={L['L2']:.6g} L3={L['L3']:.6g}")
        hdr.append(f"# deltas_us tp={explicit_diag['delta_tp_us']:.6g} tc={explicit_diag['delta_tc_us']:.6g} td={explicit_diag['delta_td_us']:.6g}")

    # Data lines
    lines = hdr + ["# etrans_meV dE_fwhm_meV"]
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
