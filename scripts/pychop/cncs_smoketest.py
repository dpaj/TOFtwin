import numpy as np

# Robust import: standalone (PyChop) vs mantid-style (pychop)
try:
    from PyChop.Instruments import Instrument
except ImportError:
    from pychop.Instruments import Instrument  # fallback if you're in a Mantid env

# Optional: if you want to use the "PyChop2" interface described in Mantid docs
PyChop2 = None
for cand in [
    ("PyChop", "PyChop2"),
    ("PyChop.PyChop", "PyChop2"),
    ("PyChop.PyChop2", "PyChop2"),
]:
    try:
        mod = __import__(cand[0], fromlist=[cand[1]])
        PyChop2 = getattr(mod, cand[1])
        break
    except Exception:
        pass

cncs = Instrument("CNCS")

import argparse
import numpy as np

# Standalone pychop ships as package "PyChop" (folder name),
# but some environments expose it as "pychop". Try both.
try:
    from PyChop.Instruments import Instrument
except ImportError:
    from pychop.Instruments import Instrument


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--Ei", type=float, default=6.59)
    p.add_argument("--E", type=float, default=3.0)
    p.add_argument("--Q", type=float, default=1.0)

    # "Resolution opening" / slot package choice == PyChop "variant" (for CNCS)
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        help='Chopper variant / opening setting (e.g. "High Flux", "Intermediate", "High Resolution")',
    )

    # Chopper frequencies: PyChop accepts scalar or list; we’ll parse as list when provided.
    p.add_argument(
        "--freq",
        type=float,
        nargs="*",
        default=None,
        help="Chopper frequencies in Hz. Provide 1 value or multiple values.",
    )

    p.add_argument("--list", action="store_true", help="List available CNCS variants and exit.")
    return p.parse_args()


args = parse_args()

CNCS = Instrument("CNCS")  # CNCS is a built-in known instrument name in standalone PyChop :contentReference[oaicite:1]{index=1}

if args.list:
    print("Available CNCS variants/packages:", CNCS.getChopperNames())  # :contentReference[oaicite:2]{index=2}
    # Optional: show frequency label hints if present in the YAML
    if hasattr(CNCS.chopper_system, "frequency_names"):
        print("Frequency names:", CNCS.chopper_system.frequency_names)
    raise SystemExit(0)

# ---- Apply "resolution opening" (variant) ----
if args.variant is not None:
    # For non-Fermi instruments (CNCS), setChopper(...) maps to "variant" :contentReference[oaicite:3]{index=3}
    CNCS.setChopper(variant=args.variant)
    # (equivalently: CNCS.setChopper(args.variant))

# ---- Apply frequencies ----
if args.freq is not None and len(args.freq) > 0:
    # Accept either one frequency or a list
    freq = args.freq[0] if len(args.freq) == 1 else args.freq
    CNCS.setFrequency(freq=freq)  # :contentReference[oaicite:4]{index=4}

Ei = args.Ei
E  = args.E
Q  = args.Q

# (Optional sanity prints)
print("Ei =", Ei)
print("Variant =", CNCS.getChopper())
print("Frequency =", CNCS.getFrequency())


Ei = 6.59
x0, xa, x1, x2, xm = cncs.chopper_system.getDistances()
print("Distances (x0, xa, x1, x2, xm) =", (x0, xa, x1, x2, xm))

# These are the “timing variances” PyChop uses internally
print("Moderator width^2 (s^2) =", cncs.moderator.getWidthSquared(Ei))
print("Chopper system width^2 =", cncs.chopper_system.getWidthSquared(Ei))

# If PyChop2 is available, use the documented OO workflow to get a resolution curve
if PyChop2 is not None:
    # Mantid docs show: setChopper, setFrequency, setEi, then getResolution(etrans) :contentReference[oaicite:4]{index=4}
    pc = PyChop2("cncs")
    # NOTE: you’ll need a valid "package" + frequency for CNCS.
    # Once you know the valid package strings, set them here:
    # pc.setChopper("<package>")
    # pc.setFrequency(<Hz>)
    pc.setEi(Ei)

    etrans = np.linspace(-0, Ei, 201)
    try:
        res = pc.getResolution(etrans)
        print("PyChop2.getResolution():", res[:5], "...", res[-5:])
    except Exception as e:
        print("PyChop2 present but getResolution failed (likely missing chopper package/freq):", e)
else:
    print("PyChop2 not importable in this install; Instrument-level access still works.")
