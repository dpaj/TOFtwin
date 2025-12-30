#!/usr/bin/env python3
"""
cncs_pychop_compare.py

Compare:
  (A) PyChop's internal elastic/inelastic energy resolution ΔE(etrans)
  (B) Your explicit timing-propagation formulas for ΔE and ΔQ

Designed to be dependency-light and easy to call from Julia via PythonCall.jl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np


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


def velocity_to_wavelength_A(v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """v [m/s] -> wavelength [Å]."""
    v = np.asarray(v, dtype=float)
    lam_m = h / (m_n * v)
    return lam_m * 1.0e10


def calculate_Q_Ainv(Ei_meV: float, E_meV: float, two_theta_rad: float) -> float:
    """|Q| [Å^-1] from Ei, Etransfer, 2θ."""
    Ef_meV = Ei_meV - E_meV
    if Ef_meV <= 0:
        return float("nan")
    v_i = float(energy_to_velocity(Ei_meV))
    v_f = float(energy_to_velocity(Ef_meV))
    Q_minv = (m_n / hbar) * math.sqrt(v_i**2 + v_f**2 - 2.0 * v_i * v_f * math.cos(two_theta_rad))
    return Q_minv * minv_to_Ainv


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
    """
    Your explicit ΔE(E) formula (timing propagation).
    Returns ΔE in meV.
    """
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
    """
    Try a few plausible locations. We don't assume the exact module layout.
    Returns PyChop2 symbol or None.
    """
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
    """
    Try a few calling conventions for setting 'variant/opening'.
    """
    if variant is None:
        return

    # Most likely: inst.setChopper(variant=...) or inst.setChopper(...)
    for kwargs in ({"variant": variant}, {"chtyp": variant}, {}):
        try:
            if kwargs:
                inst.setChopper(**kwargs)
            else:
                inst.setChopper(variant)
            return
        except Exception:
            pass

    # Some implementations might store it directly.
    for attr in ("variant", "chtyp", "chopper"):
        try:
            setattr(inst, attr, variant)
            return
        except Exception:
            pass

    raise RuntimeError(f"Could not set chopper variant/opening to '{variant}' (unknown API).")


def safe_call_set_frequency(inst: Any, freq: Union[float, Sequence[float]]) -> None:
    """
    Try a few calling conventions for setting frequency.
    """
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
        pass
    # As a last resort, set attribute
    try:
        setattr(inst, "frequency", freq)
        return
    except Exception:
        pass
    raise RuntimeError(f"Could not set frequency to '{freq}' (unknown API).")


def get_width2_list(x: Any) -> np.ndarray:
    """
    Normalize PyChop getWidthSquared return types to a 1D numpy array.
    """
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, (float, int, np.floating, np.integer)):
        return np.array([float(x)], dtype=float)
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float).ravel()
    # sometimes a dict-like
    try:
        return np.asarray(list(x), dtype=float).ravel()
    except Exception:
        return np.array([float(x)], dtype=float)


def compute_pychop_dE_meV(
    inst_name: str,
    variant: Optional[str],
    freq: Optional[Union[float, Sequence[float]]],
    Ei_meV: float,
    etrans_meV: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute PyChop ΔE(etrans) in meV using PyChop2 if available.

    Tries:
      - PyChop2.calculate(inst=..., chtyp=..., freq=..., ei=..., etrans=...)
      - OO usage: obj = PyChop2(...); obj.setChopper(...); obj.setFrequency(...); obj.setEi(...); obj.getResolution(etrans)
    """
    PyChop2 = import_pychop2()
    if PyChop2 is None:
        return None

    # 1) Try the Mantid-documented static calculate() interface.
    if hasattr(PyChop2, "calculate"):
        try:
            res, flux = PyChop2.calculate(inst=inst_name.lower(), chtyp=variant, freq=freq, ei=Ei_meV, etrans=etrans_meV)
            return np.asarray(res, dtype=float)
        except Exception:
            try:
                res, flux = PyChop2.calculate(inst=inst_name, chtyp=variant, freq=freq, ei=Ei_meV, etrans=etrans_meV)
                return np.asarray(res, dtype=float)
            except Exception:
                pass

    # 2) Try OO interface.
    try:
        obj = PyChop2(inst_name.lower())
    except Exception:
        obj = PyChop2(inst_name)

    # Apply settings if those methods exist
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
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--inst", type=str, default="CNCS", help="Instrument name for PyChop (e.g. CNCS).")
    p.add_argument("--Ei", type=float, default=6.59, help="Incident energy [meV].")

    # Etrans grid
    p.add_argument("--E", type=float, default=3.0, help="Single energy transfer [meV] for ΔQ calcs.")
    p.add_argument("--etrans-min", type=float, default=None, help="etrans grid min [meV] (default -Ei).")
    p.add_argument("--etrans-max", type=float, default=None, help="etrans grid max [meV] (default +Ei).")
    p.add_argument("--etrans-n", type=int, default=201, help="Number of points in etrans grid.")

    p.add_argument("--Q", type=float, default=1.0, help="|Q| [Å^-1] for ΔQ calcs.")
    p.add_argument("--delta-theta-deg", type=float, default=1.5, help="Angular uncertainty [deg] for ΔQ calcs.")
    p.add_argument("--no-clip", action="store_true", help="Disable clipping cos(2θ) into [-1,1].")

    # CNCS settings
    p.add_argument("--variant", type=str, default=None, help="Chopper opening/variant name (PyChop 'variant').")
    p.add_argument("--freq", type=float, nargs="*", default=None, help="Chopper frequency/frequencies [Hz].")

    # timing extraction / scaling
    p.add_argument("--tc-index", type=int, default=0, help="Which element of chopper_system.getWidthSquared(Ei) to treat as δt_c^2.")
    p.add_argument("--use-tc-rss", action="store_true", help="Use δt_c = sqrt(sum(dtc2)) instead of picking tc-index.")
    p.add_argument("--delta-td-us", type=float, default=0.0, help="Extra detector/sample time uncertainty δt_d [microseconds].")

    # output
    p.add_argument("--json", type=str, default=None, help="Write a JSON blob of inputs/outputs to this path.")
    p.add_argument("--list", action="store_true", help="List available variants/openings and exit.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    Instrument = import_instrument_class()
    inst = Instrument(args.inst)

    # List variants if requested
    if args.list:
        # Many PyChop instrument objects expose getChopperNames()
        names = None
        for m in ("getChopperNames", "getChopperName", "getChopperTypes"):
            if hasattr(inst, m):
                try:
                    names = getattr(inst, m)()
                    break
                except Exception:
                    pass
        print(f"Instrument = {args.inst}")
        print("Available variants/openings =", names)
        return 0

    # Apply settings
    if args.variant is not None:
        safe_call_set_chopper(inst, args.variant)

    if args.freq is not None and len(args.freq) > 0:
        f: Union[float, Sequence[float]] = args.freq[0] if len(args.freq) == 1 else args.freq
        safe_call_set_frequency(inst, f)

    Ei = float(args.Ei)

    # Distances: expect (x0, xa, x1, x2, xm)
    x0, xa, x1, x2, xm = inst.chopper_system.getDistances()
    x0 = float(x0); xa = float(xa); x1 = float(x1); x2 = float(x2); xm = float(xm)

    # Map to your L1/L2/L3 (as in your script)
    L1 = x0 - xm
    L2 = x1
    L3 = x2

    # Pull timing variances from PyChop
    dtm2 = float(inst.moderator.getWidthSquared(Ei))  # moderator width^2
    dtc2 = get_width2_list(inst.chopper_system.getWidthSquared(Ei))  # chopper width^2 list-like

    # Your scaling / picks
    delta_tp = math.sqrt(dtm2) * (1.0 - (xm / x0))  # your propagation factor
    if args.use_tc_rss:
        delta_tc = math.sqrt(float(np.sum(dtc2)))
    else:
        if len(dtc2) == 0:
            delta_tc = float("nan")
        else:
            idx = int(args.tc_index)
            if idx < 0 or idx >= len(dtc2):
                raise ValueError(f"--tc-index {idx} out of range for dtc2 length {len(dtc2)}")
            delta_tc = math.sqrt(float(dtc2[idx]))

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
    f_for_pychop: Optional[Union[float, Sequence[float]]] = None
    if args.freq is not None and len(args.freq) > 0:
        f_for_pychop = args.freq[0] if len(args.freq) == 1 else args.freq

    dE_pychop = compute_pychop_dE_meV(args.inst, args.variant, f_for_pychop, Ei, etrans)

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
    print(f"freq [Hz]:   {args.freq}")
    print("")
    print("---- Distances ----")
    print(f"x0={x0:.6g}  xa={xa:.6g}  x1={x1:.6g}  x2={x2:.6g}  xm={xm:.6g}  [m]")
    print(f"L1={L1:.6g}  L2={L2:.6g}  L3={L3:.6g}  [m]  (your mapping: L1=x0-xm, L2=x1, L3=x2)")
    print("")
    print("---- Raw timing variances from PyChop ----")
    print(f"dtm2 (moderator) = {dtm2:.6e} s^2  = {s2us2(dtm2):.6g} us^2")
    print(f"dtc2 (choppers)  = {dtc2} s^2  = {dtc2*1e12} us^2")
    print("")
    print("---- Timings used in your explicit formulas ----")
    print(f"delta_tp = sqrt(dtm2)*(1-xm/x0) = {delta_tp:.6e} s = {s2us(delta_tp):.6g} us")
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
        print("PyChop2 resolution could not be computed (PyChop2 not importable / incompatible API).")
        print("Printing explicit ΔE only.")
        print("etrans_meV  dE_exp_meV")
        for e, de in zip(etrans[:: max(1, npts // 20)], dE_exp[:: max(1, npts // 20)]):
            print(f"{e:10.5g}  {de:10.5g}")
    else:
        # Print a decimated table
        print("etrans_meV  dE_pychop_meV  dE_exp_meV  ratio(exp/pychop)")
        step = max(1, npts // 25)
        for e, dep, dee in zip(etrans[::step], dE_pychop[::step], dE_exp[::step]):
            ratio = (dee / dep) if (np.isfinite(dee) and np.isfinite(dep) and dep != 0) else np.nan
            print(f"{e:10.5g}  {dep:13.6g}  {dee:10.6g}  {ratio:14.6g}")

        # Also report a simple RMS fractional difference over valid points
        ok = np.isfinite(dE_pychop) & np.isfinite(dE_exp) & (dE_pychop > 0)
        if np.any(ok):
            frac = (dE_exp[ok] - dE_pychop[ok]) / dE_pychop[ok]
            rms = float(np.sqrt(np.mean(frac**2)))
            print("")
            print(f"RMS fractional difference over valid points: {rms:.6g}")

    # Optional JSON output (nice for Julia ingestion)
    if args.json is not None:
        blob = {
            "inst": args.inst,
            "Ei_meV": Ei,
            "variant": args.variant,
            "freq_Hz": args.freq,
            "distances_m": {"x0": x0, "xa": xa, "x1": x1, "x2": x2, "xm": xm, "L1": L1, "L2": L2, "L3": L3},
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
