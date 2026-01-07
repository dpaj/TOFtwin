#!/usr/bin/env python3
"""
sequoia_pychop_compare.py

SEQUOIA-focused sibling of cncs_pychop_compare.py.

Purpose
-------
Compare:
  (A) PyChop's internal elastic/inelastic energy resolution ΔE(etrans)
  (B) Your explicit timing-propagation formulas for ΔE and ΔQ

This script is designed to be dependency-light and easy to call from Julia via
PythonCall.jl (same philosophy as the CNCS version).

Chopper variant/opening options (SEQUOIA)
----------------------------------------
For SEQUOIA, the most important user-facing knob is the *Fermi package* (PyChop
calls it the chopper "variant" / opening / type, depending on the API).

By default, this script will try to discover variant names from a local
'sequoia.yaml' (PyChop-style instrument YAML). If that file is next to this
script, you can list the variants via:

  python sequoia_pychop_compare.py --list

You may also point at a YAML explicitly with:

  python sequoia_pychop_compare.py --yaml /path/to/sequoia.yaml --list

Typical variants found in the provided SEQUOIA YAML include (names are YAML keys):
  - Fine
  - Sloppy
  - SEQ-100-2.0-AST
  - SEQ-700-3.5-AST
  - ARCS-100-1.5-AST
  - ARCS-700-1.5-AST
  - ARCS-700-0.5-AST
  - ARCS-100-1.5-SMI
  - ARCS-700-1.5-SMI

Notes
-----
* When computing dE_pychop_meV, we prefer Instrument.getResolution() on the
  configured Instrument instance. This avoids the common gotcha where
  PyChop2.calculate(...) silently ignores the chopper variant/opening.
* If Instrument.getResolution() is unavailable, we fall back to PyChop2 OO usage.

Examples
--------
# List available variants (from PyChop instrument object and/or sequoia.yaml)
python sequoia_pychop_compare.py --list

# Compare ΔE(E) at Ei=100 meV using a specific Fermi package at 300 Hz
python sequoia_pychop_compare.py --Ei 100 --variant Fine --freq 300

# Compare and emit JSON for ingestion by Julia
python sequoia_pychop_compare.py --Ei 100 --variant Fine --freq 300 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np


# ----------------------------
# Small parsing helper
# ----------------------------
def parse_float_list(tokens: Optional[Sequence[str]]) -> Optional[list[float]]:
    """Parse CLI tokens into a list of floats.

    Accepts:
      --freq 300 60
      --freq 300,60
      --freq 300, 60
      --freq 300;60
    """
    if tokens is None:
        return None
    out: list[float] = []
    for tok in tokens:
        if tok is None:
            continue
        # Split on commas/semicolons; argparse already splits on whitespace.
        for part in str(tok).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
    return out if out else None



# ----------------------------
# Distance mapping helper
# ----------------------------
def choose_x_start(x0_m: float, xm_m: float, *, xm_margin_m: float = 0.5) -> tuple[float, bool]:
    """Choose the 'incident start' distance used in the simplified timing propagation.

    Many PyChop instrument definitions provide:
      - x0: moderator -> energy-defining chopper
      - xm: moderator -> upstream pulse-defining element (optional)

    CNCS-like "two-stage" definitions typically have xm < x0 by many meters.
    Some instruments (or some YAMLs) have xm == x0, meaning there is no distinct
    upstream element in the model; in that case we fall back to moderator start.

    Returns:
      (x_start_m, use_xm_as_start)
    """
    use_xm_as_start = (xm_m > 0.0) and (x0_m > xm_m) and ((x0_m - xm_m) > xm_margin_m)
    return (xm_m if use_xm_as_start else 0.0), use_xm_as_start
# ----------------------------
# SEQUOIA variant discovery (YAML)
# ----------------------------
def _yaml_safe_load(text: str) -> Optional[dict]:
    """Try to parse YAML using PyYAML if present. Returns dict or None."""
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        obj = yaml.safe_load(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def variants_from_sequoia_yaml(yaml_path: Union[str, Path]) -> list[str]:
    """Extract chopper package keys from a PyChop-style SEQUOIA YAML file.

    Strategy:
      1) Try PyYAML (if installed).
      2) Fallback to a simple regex/indentation parser for 'packages:' blocks.

    Returns a sorted list of unique variant keys.
    """
    path = Path(yaml_path)
    if not path.exists():
        return []

    txt = path.read_text(encoding="utf-8", errors="replace")

    # 1) PyYAML route
    obj = _yaml_safe_load(txt)
    if obj is not None:
        try:
            choppers = obj.get("chopper_system", {}).get("choppers", [])
            keys: set[str] = set()
            for ch in choppers:
                pkgs = (ch or {}).get("packages", {})
                if isinstance(pkgs, dict):
                    keys.update([str(k) for k in pkgs.keys()])
            return sorted(keys)
        except Exception:
            # Fall through to text parsing if structure is unexpected.
            pass

    # 2) Minimal text parser: find 'packages:' then capture subsequent keys indented more.
    # This is intentionally conservative; if it fails, it returns [] rather than lying.
    keys: set[str] = set()
    in_packages = False
    packages_indent: Optional[int] = None

    for line in txt.splitlines():
        raw = line.rstrip("\n")
        # detect packages:
        if raw.strip().startswith("packages:"):
            in_packages = True
            packages_indent = len(raw) - len(raw.lstrip(" "))
            continue

        if not in_packages or packages_indent is None:
            continue

        # stop when indentation returns to <= packages_indent
        indent = len(raw) - len(raw.lstrip(" "))
        if raw.strip() == "":
            continue
        if indent <= packages_indent:
            in_packages = False
            packages_indent = None
            continue

        # keys at the first indentation level under packages:
        # e.g. "        Fine:" (8 spaces then key:)
        # We accept keys containing letters, digits, underscore, dot, dash.
        stripped = raw.strip()
        if stripped.endswith(":") and ":" not in stripped[:-1]:
            key = stripped[:-1].strip()
            if key and all(c.isalnum() or c in "_.-" for c in key):
                keys.add(key)

    return sorted(keys)


def default_sequoia_yaml_path() -> Optional[Path]:
    """Return a good default guess for sequoia.yaml location."""
    here = Path(__file__).resolve().parent
    cand = here / "sequoia.yaml"
    return cand if cand.exists() else None

def derive_L123_from_instrument(inst: Any, x0: float, xa: float, x1: float, x2: float, xm: float) -> Tuple[float, float, float]:
    """Derive (L1, L2, L3) used by the TOFtwin timing-propagation formulas.

    Goals:
      * Avoid instrument-specific assumptions that can yield L1=0 (as you saw on SEQUOIA).
      * Prefer physically meaningful distances:
          L1 = moderator -> (final) chopper used for energy definition
          L2 = chopper -> sample
          L3 = sample -> detector

    We try, in order:
      - L2 from x1 (getDistances), else from inst.chopper_system.chop_sam if present
      - L3 from x2 (getDistances), else from inst.chopper_system.sam_det if present
      - L1 from max(chopper.distance) if accessible, else from (xm - L2) if positive, else from (x0 - xm) if positive

    Raises ValueError if we cannot obtain strictly-positive L1/L2/L3.
    """
    # L2 and L3: usually reliable from getDistances()
    L2 = float(x1) if (x1 is not None and float(x1) > 0) else float(getattr(inst.chopper_system, "chop_sam", 0.0) or 0.0)
    L3 = float(x2) if (x2 is not None and float(x2) > 0) else float(getattr(inst.chopper_system, "sam_det", 0.0) or 0.0)

    # L1: prefer explicit chopper distances (distance from moderator to each chopper)
    L1_candidates = []
    try:
        choppers = getattr(inst.chopper_system, "choppers", None)
        if choppers:
            dists = []
            for ch in choppers:
                d = getattr(ch, "distance", None)
                if d is not None and float(d) > 0:
                    dists.append(float(d))
            if dists:
                L1_candidates.append(max(dists))
    except Exception:
        pass

    # Alternative: if xm is moderator->sample, then L1 = xm - L2 (moderator->chopper)
    try:
        if xm is not None and L2 > 0 and float(xm) > L2:
            L1_candidates.append(float(xm) - L2)
    except Exception:
        pass

    # Legacy mapping used in CNCS script
    try:
        if x0 is not None and xm is not None and float(x0) > float(xm):
            L1_candidates.append(float(x0) - float(xm))
    except Exception:
        pass

    L1 = max(L1_candidates) if L1_candidates else 0.0

    # Final validation
    if not (L1 > 0 and L2 > 0 and L3 > 0):
        raise ValueError(
            f"Could not derive positive L1/L2/L3. "
            f"Got (L1, L2, L3)=({L1}, {L2}, {L3}) from getDistances x0={x0}, xa={xa}, x1={x1}, x2={x2}, xm={xm}."
        )

    return L1, L2, L3


def derive_x0_for_delta_tp(x0: float, x2: float, xm: float, L1: float, L2: float, L3: float) -> float:
    """Choose an effective x0 used in delta_tp = sqrt(dtm2) * (1 - xm/x0).

    On some instruments, PyChop's getDistances() can return x0 == xm (or x0 == 0),
    which would incorrectly force delta_tp to 0 and can break downstream math.

    Heuristic:
      - If x0 is sane and > xm, keep it.
      - Else, if xm and x2 are positive, use xm + x2 (moderator->sample + sample->det).
      - Else, fall back to total L1+L2+L3.
    """
    try:
        if x0 is not None and float(x0) > 0 and xm is not None and float(x0) > float(xm):
            return float(x0)
    except Exception:
        pass
    try:
        if xm is not None and float(xm) > 0 and x2 is not None and float(x2) > 0:
            return float(xm) + float(x2)
    except Exception:
        pass
    tot = float(L1 + L2 + L3)
    return tot if tot > 0 else 1.0  # last-resort avoid division by zero



# ----------------------------
# Physical constants (SI)
# ----------------------------
m_n = 1.67492749804e-27          # neutron mass [kg]
hbar = 1.054571817e-34           # [J*s]
h = 6.62607015e-34               # [J*s]
meV_to_J = 1.602176634e-22       # 1 meV in Joules
J_to_meV = 1.0 / meV_to_J
Ainv_to_minv = 1.0e10            # Å^-1 -> m^-1 multiply by 1e10
minv_to_Ainv = 1.0e-10           # m^-1 -> Å^-1 multiply by 1e-10


# ----------------------------
# Your kinematics helpers
# ----------------------------
def energy_to_velocity(E_meV: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """E [meV] -> v [m/s]."""
    E_J = np.asarray(E_meV, dtype=float) * meV_to_J
    return np.sqrt(2.0 * E_J / m_n)


def calculate_two_theta_rad(Ei_meV: float, E_meV: float, Q_Ainv: float, clip: bool = True) -> float:
    """2θ [rad] from Ei, Etransfer, |Q| [Å^-1]."""
    Ef_meV = Ei_meV - E_meV
    if Ef_meV <= 0:
        return float("nan")
    v_i = float(energy_to_velocity(Ei_meV))
    v_f = float(energy_to_velocity(Ef_meV))
    Q_minv = Q_Ainv * Ainv_to_minv
    cos_2theta = (v_i**2 + v_f**2 - (Q_minv**2 * hbar**2 / m_n**2)) / (2.0 * v_i * v_f)
    if clip:
        cos_2theta = float(np.clip(cos_2theta, -1.0, 1.0))
    return math.acos(cos_2theta)


# ----------------------------
# Your explicit ΔE model
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
    """Your explicit ΔE(E) formula (timing propagation). Returns ΔE in meV."""
    E = np.asarray(E_meV, dtype=float)
    Ef = Ei_meV - E
    out = np.full_like(E, np.nan, dtype=float)
    ok = Ef > 0

    Ei_J = Ei_meV * meV_to_J
    Ef_J = Ef[ok] * meV_to_J

    vi_cubed = (2.0 * Ei_J / m_n) ** (1.5)
    vf_cubed = (2.0 * Ef_J / m_n) ** (1.5)

    term_tp = (vi_cubed / L1_m + vf_cubed * L2_m / (L1_m * L3_m)) ** 2 * (delta_tp_s ** 2)
    term_tc = (vi_cubed / L1_m + vf_cubed * (L1_m + L2_m) / (L1_m * L3_m)) ** 2 * (delta_tc_s ** 2)
    term_td = (vf_cubed / L3_m) ** 2 * (delta_td_s ** 2)

    delta_E_J = m_n * np.sqrt(term_tp + term_tc + term_td)
    out[ok] = delta_E_J * J_to_meV
    return out if isinstance(E_meV, np.ndarray) else float(out.item())


# ----------------------------
# Your explicit ΔQ model
# ----------------------------
def dQx_Ainv(
    Ei_meV: float,
    E_meV: float,
    Q_Ainv: float,
    L1_m: float,
    L2_m: float,
    L3_m: float,
    delta_tp_s: float,
    delta_tc_s: float,
    delta_td_s: float,
    delta_theta_rad: float,
    clip: bool = True,
) -> float:
    Ef = Ei_meV - E_meV
    if Ef <= 0:
        return float("nan")
    v_i = float(energy_to_velocity(Ei_meV))
    v_f = float(energy_to_velocity(Ef))
    two_theta = calculate_two_theta_rad(Ei_meV, E_meV, Q_Ainv, clip=clip)
    if not np.isfinite(two_theta):
        return float("nan")

    cos_2theta = math.cos(two_theta)
    sin_2theta = math.sin(two_theta)

    term1 = (v_i**2 + v_f**2 * (L2_m / L3_m) * cos_2theta) ** 2 * (delta_tp_s**2) / (L1_m**2)
    term2 = (v_i**2 + v_f**2 * ((L1_m + L2_m) / L3_m) * cos_2theta) ** 2 * (delta_tc_s**2) / (L1_m**2)
    term3 = (v_f**2 / L3_m * cos_2theta) ** 2 * (delta_td_s**2)
    term4 = (v_f**2 * (sin_2theta**2)) * (delta_theta_rad**2)

    dQx_minv = (m_n / hbar) * math.sqrt(term1 + term2 + term3 + term4)
    return dQx_minv * minv_to_Ainv


def dQy_Ainv(
    Ei_meV: float,
    E_meV: float,
    Q_Ainv: float,
    L1_m: float,
    L2_m: float,
    L3_m: float,
    delta_tp_s: float,
    delta_tc_s: float,
    delta_td_s: float,
    delta_theta_rad: float,
    clip: bool = True,
) -> float:
    Ef = Ei_meV - E_meV
    if Ef <= 0:
        return float("nan")
    v_i = float(energy_to_velocity(Ei_meV))
    v_f = float(energy_to_velocity(Ef))
    two_theta = calculate_two_theta_rad(Ei_meV, E_meV, Q_Ainv, clip=clip)
    if not np.isfinite(two_theta):
        return float("nan")

    cos_2theta = math.cos(two_theta)
    sin_2theta = math.sin(two_theta)

    term1 = (v_f**2 * (L2_m / (L1_m * L3_m)) * sin_2theta) ** 2 * (delta_tp_s**2)
    term2 = (v_f**2 * ((L1_m + L2_m) / (L1_m * L3_m)) * sin_2theta) ** 2 * (delta_tc_s**2)
    term3 = (v_f**2 / L3_m * sin_2theta) ** 2 * (delta_td_s**2)
    term4 = (v_f**2 * (cos_2theta**2)) * (delta_theta_rad**2)

    dQy_minv = (m_n / hbar) * math.sqrt(term1 + term2 + term3 + term4)
    return dQy_minv * minv_to_Ainv


def dQ_Ainv(
    Ei_meV: float,
    E_meV: float,
    Q_Ainv: float,
    L1_m: float,
    L2_m: float,
    L3_m: float,
    delta_tp_s: float,
    delta_tc_s: float,
    delta_td_s: float,
    delta_theta_rad: float,
    clip: bool = True,
) -> float:
    Ef = Ei_meV - E_meV
    if Ef <= 0 or Q_Ainv <= 0:
        return float("nan")

    v_i = float(energy_to_velocity(Ei_meV))
    v_f = float(energy_to_velocity(Ef))
    two_theta = calculate_two_theta_rad(Ei_meV, E_meV, Q_Ainv, clip=clip)
    if not np.isfinite(two_theta):
        return float("nan")

    dqx = dQx_Ainv(Ei_meV, E_meV, Q_Ainv, L1_m, L2_m, L3_m, delta_tp_s, delta_tc_s, delta_td_s, delta_theta_rad, clip=clip)
    dqy = dQy_Ainv(Ei_meV, E_meV, Q_Ainv, L1_m, L2_m, L3_m, delta_tp_s, delta_tc_s, delta_td_s, delta_theta_rad, clip=clip)

    cos_2theta = math.cos(two_theta)
    sin_2theta = math.sin(two_theta)

    Qx_Ainv = ((m_n / hbar) * (v_i - v_f * cos_2theta)) * minv_to_Ainv
    Qy_Ainv = ((m_n / hbar) * (-v_f * sin_2theta)) * minv_to_Ainv

    return (1.0 / Q_Ainv) * math.sqrt((Qx_Ainv**2) * (dqx**2) + (Qy_Ainv**2) * (dqy**2))


# ----------------------------
# PyChop import / API adapters
# ----------------------------
def import_instrument_class():
    # Standalone repo uses "PyChop" package dir; some envs might expose "pychop".
    try:
        from PyChop.Instruments import Instrument  # type: ignore
        return Instrument
    except Exception:
        from pychop.Instruments import Instrument  # type: ignore
        return Instrument


def import_pychop2():
    """Try a few plausible locations. We don't assume the exact module layout."""
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


def safe_call_set_chopper(inst: Any, variant: str) -> None:
    """Try a few calling conventions for setting 'variant/opening'."""
    if variant is None:
        return

    for kwargs in ({"variant": variant}, {"chtyp": variant}, {}):
        try:
            if kwargs:
                inst.setChopper(**kwargs)
            else:
                inst.setChopper(variant)
            return
        except Exception:
            pass

    for attr in ("variant", "chtyp", "chopper"):
        try:
            setattr(inst, attr, variant)
            return
        except Exception:
            pass

    raise RuntimeError(f"Could not set chopper variant/opening to '{variant}' (unknown API).")


def safe_call_set_frequency(inst: Any, freq: Union[float, Sequence[float]]) -> None:
    """Try a few calling conventions for setting frequency."""
    if freq is None:
        return
    try:
        inst.setFrequency(freq=freq)
        return
    except Exception:
        pass
    try:
        inst.setFrequency(freq)
        return
    except Exception:
        # Some APIs want multiple positional args
        try:
            if isinstance(freq, (list, tuple, np.ndarray)) and not isinstance(freq, (str, bytes)):
                inst.setFrequency(*freq)
                return
        except Exception:
            pass
        pass
    try:
        setattr(inst, "frequency", freq)
        return
    except Exception:
        pass
    raise RuntimeError(f"Could not set frequency to '{freq}' (unknown API).")


def get_width2_list(x: Any) -> np.ndarray:
    """Normalize PyChop getWidthSquared return types to a 1D numpy array."""
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


def compute_instrument_dE_meV(inst: Any, Ei_meV: float, etrans_meV: np.ndarray) -> Optional[np.ndarray]:
    """Compute dE(etrans) using the configured Instrument object if it provides getResolution."""
    if hasattr(inst, "setEi"):
        try:
            inst.setEi(Ei_meV)
        except Exception:
            pass

    for m in ("getResolution", "getResolutions", "resolution"):
        if not hasattr(inst, m):
            continue
        meth = getattr(inst, m)
        for args in ((etrans_meV,), (Ei_meV, etrans_meV)):
            try:
                out = meth(*args)
                return np.asarray(out, dtype=float).ravel()
            except Exception:
                pass
    return None


def compute_pychop_dE_meV(
    inst_name: str,
    variant: Optional[str],
    freq: Optional[Union[float, Sequence[float]]],
    Ei_meV: float,
    etrans_meV: np.ndarray,
    prefer_oo: bool = True,
) -> Optional[np.ndarray]:
    """Compute PyChop ΔE(etrans) in meV using PyChop2 if available."""
    PyChop2 = import_pychop2()
    if PyChop2 is None:
        return None

    if not prefer_oo:
        if hasattr(PyChop2, "calculate"):
            try:
                res, _flux = PyChop2.calculate(inst=inst_name.lower(), chtyp=variant, freq=freq, ei=Ei_meV, etrans=etrans_meV)
                return np.asarray(res, dtype=float)
            except Exception:
                try:
                    res, _flux = PyChop2.calculate(inst=inst_name, chtyp=variant, freq=freq, ei=Ei_meV, etrans=etrans_meV)
                    return np.asarray(res, dtype=float)
                except Exception:
                    pass

    try:
        obj = PyChop2(inst_name.lower())
    except Exception:
        obj = PyChop2(inst_name)

    if variant is not None:
        for m in ("setChopper", "setChopperType"):
            if hasattr(obj, m):
                try:
                    getattr(obj, m)(variant)
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


# ----------------------------
# CLI + main
# ----------------------------
def parse_args():
    default_yaml = default_sequoia_yaml_path()
    default_yaml_str = str(default_yaml) if default_yaml is not None else None
    yaml_variants = variants_from_sequoia_yaml(default_yaml) if default_yaml is not None else []

    epilog_lines = []
    if default_yaml_str is not None:
        epilog_lines.append(f"Default YAML: {default_yaml_str}")
        if yaml_variants:
            epilog_lines.append("Variants from default YAML:")
            epilog_lines.append("  " + ", ".join(yaml_variants))
    epilog = "\n".join(epilog_lines) if epilog_lines else None

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog,
    )

    p.add_argument("--inst", type=str, default="SEQUOIA", help="Instrument name for PyChop (default: SEQUOIA).")
    p.add_argument("--yaml", type=str, default=default_yaml_str,
                   help="Optional path to a PyChop YAML for variant listing (used by --list and for help text).")


    p.add_argument("--Ei", type=float, default=100.0, help="Incident energy [meV].")

    # Etrans grid
    p.add_argument("--E", type=float, default=10.0, help="Single energy transfer [meV] for ΔQ calcs.")
    p.add_argument("--etrans-min", type=float, default=None, help="etrans grid min [meV] (default -Ei).")
    p.add_argument("--etrans-max", type=float, default=None, help="etrans grid max [meV] (default +Ei).")
    p.add_argument("--etrans-n", type=int, default=201, help="Number of points in etrans grid.")

    p.add_argument("--Q", type=float, default=2.0, help="|Q| [Å^-1] for ΔQ calcs.")
    p.add_argument("--delta-theta-deg", type=float, default=1.0, help="Angular uncertainty [deg] for ΔQ calcs.")
    p.add_argument("--no-clip", action="store_true", help="Disable clipping cos(2θ) into [-1,1].")

    # SEQUOIA settings
    p.add_argument("--variant", type=str, default=None,
                   help="Chopper opening/variant name (for SEQUOIA this is typically the Fermi package key, e.g. 'Fine' or 'Sloppy').")
    p.add_argument("--freq", type=str, nargs="*", default=None,
                   help="Chopper frequency/frequencies [Hz]. Accepts e.g. --freq 300 or --freq 300,60.")
    p.add_argument("--dE-source", type=str, default="auto", choices=("auto","instrument","pychop2"),
                   help="How to compute dE_pychop: try Instrument.getResolution, PyChop2, or auto (instrument then PyChop2).")

    # timing extraction / scaling
    p.add_argument("--tc-index", type=int, default=0,
                   help="Which element of chopper_system.getWidthSquared(Ei) to treat as δt_c^2.")
    p.add_argument("--use-tc-rss", action="store_true",
                   help="Use δt_c = sqrt(sum(dtc2)) instead of picking tc-index.")
    p.add_argument("--delta-td-us", type=float, default=0.0,
                   help="Extra detector/sample time uncertainty δt_d [microseconds]." )

    # output
    p.add_argument("--json", type=str, default=None, help="Write a JSON blob of inputs/outputs to this path.")
    p.add_argument("--list", action="store_true",
                   help="List available variants/openings (from Instrument object and/or YAML) and exit.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Import PyChop lazily enough that `--list` can still work from YAML
    # even if PyChop isn't installed in the current Python environment.
    inst = None
    pychop_err: Optional[BaseException] = None
    try:
        Instrument = import_instrument_class()
        inst = Instrument(args.inst)
    except Exception as e:
        pychop_err = e

    # List variants if requested

    if args.list:
        print(f"Instrument = {args.inst}")

        # 1) From PyChop instrument object, if available
        names = None
        if inst is None:
            print("PyChop Instrument unavailable (import/init failed):", repr(pychop_err))
        else:
            for m in ("getChopperNames", "getChopperName", "getChopperTypes"):
                if hasattr(inst, m):
                    try:
                        names = getattr(inst, m)()
                        break
                    except Exception:
                        pass
        print("Available variants from Instrument =", names)

        # 2) From YAML (if provided)
        if args.yaml is not None:
            ypath = Path(args.yaml)
            if ypath.exists():
                v = variants_from_sequoia_yaml(ypath)
                print(f"Available variants from YAML ({ypath}) =", v)
            else:
                print(f"YAML path does not exist: {ypath}")
        else:
            print("No YAML path provided (use --yaml /path/to/sequoia.yaml)")
        return 0

    # Apply settings
    if args.variant is not None:
        safe_call_set_chopper(inst, args.variant)

    # Parse frequency tokens into float(s)
    freq_val: Optional[Union[float, Sequence[float]]] = None
    freq_list = parse_float_list(args.freq)
    if freq_list is not None and len(freq_list) > 0:
        freq_val = freq_list[0] if len(freq_list) == 1 else freq_list
        safe_call_set_frequency(inst, freq_val)

    Ei = float(args.Ei)

    # Distances: expect (x0, xa, x1, x2, xm)
    x0, xa, x1, x2, xm = inst.chopper_system.getDistances()
    x0 = float(x0); xa = float(xa); x1 = float(x1); x2 = float(x2); xm = float(xm)

    # ------------------------------------------------------------------
    # Mapping refresher (why CNCS used L1=x0-xm and delta_tp*(1-x_start/x0))
    #
    # In the "two-stage" case (common on CNCS-like definitions), `xm` is the
    # distance from the moderator to an *upstream pulse-defining element*
    # (e.g. a T0 / bandwidth / pulse-shaping chopper), and `x0` is the
    # distance to the *energy-defining* chopper (e.g. Fermi).
    #
    # If you treat that upstream element at `xm` as the effective "start" for
    # incident TOF, then:
    #   L1 = (moderator->energy chopper) - (moderator->pulse chopper) = x0 - xm
    # and the moderator pulse-time variance contributes reduced by the
    # time-focusing factor (1 - xm/x0) in your simplified propagation model.
    #
    # In the "single-stage" case (SEQUOIA as defined in the PyChop YAML),
    # PyChop returns xm == x0 (there is no separate upstream pulse chopper in
    # the model). In that case you should fall back to using the moderator as
    # the start:
    #   x_start = 0,  L1 = x0,  delta_tp = sqrt(dtm2).
    # ------------------------------------------------------------------

    # Choose incident start (two-stage definitions use xm; single-stage falls back to moderator start)
    x_start, use_xm_as_start = choose_x_start(x0, xm, xm_margin_m=0.5)

    # Map to L1/L2/L3 used by the explicit timing-propagation formulas
    L1 = x0 - x_start   # start -> energy-defining chopper
    L2 = x1             # chopper -> sample
    L3 = x2             # sample -> detector

    # Validate (avoid divisions by 0 in explicit formulas)
    if not (L1 > 0.0 and L2 > 0.0 and L3 > 0.0):
        raise ValueError(f"Non-positive derived distance(s): L1={L1}, L2={L2}, L3={L3} (raw: x0={x0}, x1={x1}, x2={x2}, xm={xm})")

    # Early debug print (before any math that could divide by 0)
    print("---- Distances (raw from PyChop) ----")
    print(f"x0={x0:.6g}  xa={xa:.6g}  x1={x1:.6g}  x2={x2:.6g}  xm={xm:.6g}  [m]")
    if use_xm_as_start:
        print(f"Using upstream start at xm={xm:.6g} m  => L1=x0-xm={L1:.6g} m")
    else:
        print("Using moderator as start (single-stage) => L1=x0")
    print(f"Mapped:  L1={L1:.6g}  L2={L2:.6g}  L3={L3:.6g}  [m]  (L2=x1, L3=x2)")
    print("Note: this is the mapping used in the explicit formulas below.")
    print("")

    # Pull timing variances from PyChop
    dtm2 = float(inst.moderator.getWidthSquared(Ei))              # moderator width^2
    dtc2 = get_width2_list(inst.chopper_system.getWidthSquared(Ei))  # chopper width^2 list-like


    if x0 == 0.0:
        raise ValueError('x0 returned by PyChop is 0; cannot compute delta_tp scaling.')
    # Your scaling / picks
    delta_tp = math.sqrt(dtm2) * (1.0 - (x_start / x0))  # moderator contribution (reduced only in two-stage case)
    if args.use_tc_rss:
        delta_tc = math.sqrt(float(np.nansum(dtc2)))
    else:
        if len(dtc2) == 0:
            delta_tc = float("nan")
        else:
            idx = int(args.tc_index)
            if idx < 0 or idx >= len(dtc2):
                raise ValueError(f"--tc-index {idx} out of range for dtc2 length {len(dtc2)}")
            val = float(dtc2[idx])
            if not np.isfinite(val):
                # Fallback: pick the first finite entry (PyChop sometimes returns NaN for unused stages)
                finite = [float(v) for v in dtc2 if np.isfinite(v)]
                val = finite[0] if finite else float("nan")
            delta_tc = math.sqrt(val)

    delta_td = float(args.delta_td_us) * 1e-6  # user-specified [us] -> [s]
    delta_theta = float(args.delta_theta_deg) * math.pi / 180.0
    clip = not args.no_clip

    # Etrans grid for ΔE comparison
    emin = -Ei if args.etrans_min is None else float(args.etrans_min)
    emax = Ei if args.etrans_max is None else float(args.etrans_max)
    npts = int(args.etrans_n)
    etrans = np.linspace(emin, emax, npts)

    # Compute explicit ΔE(etrans)
    dE_exp = energy_resolution_explicit_meV(Ei, etrans, L1, L2, L3, delta_tp, delta_tc, delta_td)

    # Compute PyChop ΔE(etrans) if possible
    dE_pychop_source = None
    dE_pychop = None
    if args.dE_source in ("auto", "instrument"):
        dE_pychop = compute_instrument_dE_meV(inst, Ei, etrans)
        if dE_pychop is not None:
            dE_pychop_source = "Instrument.getResolution"
    if dE_pychop is None and args.dE_source in ("auto", "pychop2"):
        dE_pychop = compute_pychop_dE_meV(args.inst, args.variant, freq_val, Ei, etrans, prefer_oo=True)
        if dE_pychop is not None:
            dE_pychop_source = "PyChop2 (OO)"

    # Single-point ΔQ comparison at (E, Q)
    E_single = float(args.E)
    Q_single = float(args.Q)
    dqx = dQx_Ainv(Ei, E_single, Q_single, L1, L2, L3, delta_tp, delta_tc, delta_td, delta_theta, clip=clip)
    dqy = dQy_Ainv(Ei, E_single, Q_single, L1, L2, L3, delta_tp, delta_tc, delta_td, delta_theta, clip=clip)
    dq = dQ_Ainv(Ei, E_single, Q_single, L1, L2, L3, delta_tp, delta_tc, delta_td, delta_theta, clip=clip)

    # Pretty prints
    def s2us(x_s: float) -> float:
        return x_s * 1e6

    def s2us2(x_s2: float) -> float:
        return x_s2 * 1e12

    print("---- PyChop settings ----")
    print(f"Instrument: {args.inst}")
    print(f"Ei [meV]:    {Ei}")
    print(f"variant:     {args.variant}")
    print(f"freq [Hz]:   {freq_list}")
    print(f"dE source:   {dE_pychop_source}")
    if args.yaml is not None:
        print(f"yaml:        {args.yaml}")
    print("")

    print("---- Raw timing variances from PyChop ----")
    print(f"dtm2 (moderator) = {dtm2:.6e} s^2  = {s2us2(dtm2):.6g} us^2")
    print(f"dtc2 (choppers)  = {dtc2} s^2  = {dtc2*1e12} us^2")
    print("")

    print("---- Timings used in your explicit formulas ----")
    print(f"delta_tp = sqrt(dtm2)*(1-x_start/x0) = {delta_tp:.6e} s = {s2us(delta_tp):.6g} us")
    if args.use_tc_rss:
        print(f"delta_tc = sqrt(sum(dtc2))      = {delta_tc:.6e} s = {s2us(delta_tc):.6g} us")
    else:
        print(f"delta_tc = sqrt(dtc2[{args.tc_index}])    = {delta_tc:.6e} s = {s2us(delta_tc):.6g} us")
    print(f"delta_td (user)                = {delta_td:.6e} s = {s2us(delta_td):.6g} us")
    print("")

    print("---- ΔQ at single point ----")
    print(f"E = {E_single} meV, Q = {Q_single} Å^-1, delta_theta = {args.delta_theta_deg} deg")
    print(f"dQx = {dqx:.6g} Å^-1, dQy = {dqy:.6g} Å^-1, dQ = {dq:.6g} Å^-1")
    print("")

    print("---- ΔE(etrans) comparison ----")
    if dE_pychop is None:
        print("PyChop resolution could not be computed (PyChop2 not importable / incompatible API).")
             
        print("Printing explicit ΔE only.")
        print("etrans_meV  dE_exp_meV")
        for e, de in zip(etrans[:: max(1, npts // 20)], dE_exp[:: max(1, npts // 20)]):
            print(f"{e:10.5g}  {de:10.5g}")
    else:
        print("etrans_meV  dE_pychop_meV  dE_exp_meV  ratio(exp/pychop)")
        step = max(1, npts // 25)
        for e, dep, dee in zip(etrans[::step], dE_pychop[::step], dE_exp[::step]):
            ratio = (dee / dep) if (np.isfinite(dee) and np.isfinite(dep) and dep != 0) else np.nan
            print(f"{e:10.5g}  {dep:13.6g}  {dee:10.6g}  {ratio:14.6g}")

        ok = np.isfinite(dE_pychop) & np.isfinite(dE_exp) & (dE_pychop > 0)
        if np.any(ok):
            frac = (dE_exp[ok] - dE_pychop[ok]) / dE_pychop[ok]
            rms = float(np.sqrt(np.mean(frac**2)))
            print("\nRMS fractional difference over valid points: {:.6g}".format(rms))

    # Optional JSON output (nice for Julia ingestion)
    if args.json is not None:
        blob = {
            "inst": args.inst,
            "Ei_meV": Ei,
            "variant": args.variant,
            "freq_Hz": args.freq,
            "yaml": args.yaml,
            "distances_m": {"x0": x0, "xa": xa, "x1": x1, "x2": x2, "xm": xm, "x_start": x_start, "use_xm_as_start": bool(use_xm_as_start), "L1": L1, "L2": L2, "L3": L3},
            "timing_variances_s2": {"dtm2": dtm2, "dtc2": dtc2.tolist()},
            "timings_used_s": {
                "delta_tp": delta_tp,
                "delta_tc": delta_tc,
                "delta_td": delta_td,
                "delta_theta_rad": delta_theta,
                "clip_cos2theta": clip,
                "tc_index": args.tc_index,
                "use_tc_rss": bool(args.use_tc_rss),
            },
            "etrans_meV": etrans.tolist(),
            "dE_explicit_meV": dE_exp.tolist(),
            "dE_pychop_meV": (None if dE_pychop is None else np.asarray(dE_pychop, dtype=float).tolist()),
            "single_point": {
                "E_meV": E_single,
                "Q_Ainv": Q_single,
                "dQx_Ainv": dqx,
                "dQy_Ainv": dqy,
                "dQ_Ainv": dq,
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        print(f"\nWrote JSON to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
