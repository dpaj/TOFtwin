#!/usr/bin/env julia
# -----------------------------------------------------------------------------
# Generate Sunny powder-averaged S(|Q|,ω) tables for Ba3ZnRu2O9 (BZRO)
#
# This script is intended to produce JLD2 files usable by TOFtwin's powder workflow.
# It mirrors the structure of scripts like sunny_corh2o4_powder.jl, but uses the
# BZRO model (Ru1 subcrystal) and writes multiple tables (one per instrument/Ei).
#
# Axes below are the TOFtwin-suggested (|Q|, ω) grids for:
#   1) SEQUOIA Ei=160 meV
#   2) CNCS    Ei=12  meV
#   3) CNCS    Ei=3.32 meV
#
# You can override basic runtime knobs via CLI/env:
#   --nsamp 400        (powder directions per radius; default 400)
#   --fwhm  0.25       (intrinsic Lorentzian broadening [meV]; default 0.25)
#   --outdir out       (default "out")
#   --float32          (store S_Qω as Float32)
#   --only TAG         (compute only one table: seq160, cncs12, cncs3p32)
#
# -----------------------------------------------------------------------------

using Pkg

# Find project root (directory containing Project.toml), starting from this script.
function _find_project_root(start_dir::AbstractString)
    d = abspath(start_dir)
    while true
        if isfile(joinpath(d, "Project.toml"))
            return d
        end
        parent = dirname(d)
        parent == d && error("Could not find Project.toml above $start_dir")
        d = parent
    end
end

const _PROJ = _find_project_root(@__DIR__)
Pkg.activate(_PROJ)

using Sunny
using LinearAlgebra
using JLD2
using Logging

# ------------------------------ tiny CLI helpers ------------------------------
function _arg_value(flag::AbstractString, default::AbstractString)
    for i in 1:length(ARGS)-1
        if ARGS[i] == flag
            return ARGS[i+1]
        end
    end
    return default
end

function _has_flag(flag::AbstractString)
    any(==(flag), ARGS)
end

function _only_tag()
    for i in 1:length(ARGS)-1
        if ARGS[i] == "--only"
            return ARGS[i+1]
        end
    end
    return nothing
end

nsamp = parse(Int, get(ENV, "SUNNY_POWDER_NSAMP", _arg_value("--nsamp", "2000")))
fwhm  = parse(Float64, get(ENV, "SUNNY_POWDER_FWHM_MEV", _arg_value("--fwhm", "0.25")))
outdir = get(ENV, "SUNNY_POWDER_OUTDIR", _arg_value("--outdir", "out"))
as_f32 = _has_flag("--float32") || (lowercase(get(ENV, "SUNNY_POWDER_FLOAT32", "false")) in ("1","true","yes","y"))
only_tag = _only_tag()

mkpath(outdir)

# ------------------------------ BZRO model -----------------------------------
"""
Build the BZRO model system (Ru1 magnetic subcrystal) from CIF.

This is the same model as `original_model` in your model.jl, but written here
without relying on `datadir()` so the script is runnable standalone.

Parameters:
- dims: supercell dims for System (default (1,1,1))
- D, J1..J4: model parameters in meV
- mode: :SUN (default)
"""
function bzro_model(; dims=(1,1,1), D=-0.0665, J1=21.7, J2=5.65, J3=0.66, J4=0.085, mode=:SUN,
                    cif_path::AbstractString)

    xtal_bzro = Crystal(cif_path; symprec=0.001)
    mag_xtal  = subcrystal(xtal_bzro, "Ru1")
    moments = [1 => Moment(; s=3/2, g=2)]
    sys = System(mag_xtal, moments, mode; dims=dims)

    set_exchange!(sys, J1, Bond(1, 2, [0, 0, 0]))
    set_exchange!(sys, J2, Bond(2, 3, [0, 1, 0]))
    set_exchange!(sys, J3, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J4, Bond(1, 2, [1, 0, 0]))

    set_onsite_coupling!(sys, S -> D*S[3]^2, 1)
    return sys, mag_xtal
end

# Locate CIF (override with env var if desired)
default_cif = joinpath(_PROJ, "data", "BZRO_CollCode253813.cif")
cif_path = get(ENV, "BZRO_CIF", default_cif)
isfile(cif_path) || error("BZRO CIF not found at: $cif_path (set BZRO_CIF to override)")

@info "Using CIF" cif_path

sys, cryst = bzro_model(; cif_path)

# Ground state (needed for LSWT reference)
@info "Finding classical ground state (minimize_energy!) ..."
randomize_spins!(sys)
minimize_energy!(sys)

# Measurement / SWT
# Form factors: try a sensible Ru ion; if unavailable, fall back to unit form factor.
function _pick_first_formfactor(candidates::AbstractVector{<:AbstractString})
    for ion in candidates
        try
            return ion, FormFactor(ion)
        catch
            # try next candidate
        end
    end
    return nothing, nothing
end

function _make_measure(sys)
    # Sunny expects ion strings like "Fe2" (Fe²⁺). For Ru⁵⁺, try "Ru5" (no plus sign).
    candidates = ["Ru5", "Ru4", "Ru3", "Ru2", "Ru1"]
    ion, ff = _pick_first_formfactor(candidates)
    if ff === nothing
        @warn "No Ru form factor available; using ssf_perp without form factors" candidates
        return ssf_perp(sys)
    end
    @info "Using magnetic form factor" ion=ion
    return ssf_perp(sys; formfactors=[1 => ff])
end

measure = _make_measure(sys)
swt = SpinWaveTheory(sys; measure)

kernel = lorentzian(fwhm=fwhm)

# Best-effort Sunny version (avoid hard dependency on UUIDs).
function _sunny_version()
    try
        return string(Base.pkgversion(Sunny))
    catch
        return nothing
    end
end

# ------------------------------ Axes sets ------------------------------------
# Tags are used in output filenames and for --only filtering.

const TABLE_SPECS = [
    (
        tag = "seq160",
        desc = "SEQUOIA Ei=160 meV",
        radii = collect(range(0.0, 9.0; length=500)),
        energies = collect(range(0.0, 160.0; length=200)),
    ),
    (
        tag = "cncs12",
        desc = "CNCS Ei=12 meV",
        radii = collect(range(0.0, 4.5; length=500)),
        energies = collect(range(0.0, 12.0; length=200)),
    ),
    (
        tag = "cncs3p32",
        desc = "CNCS Ei=3.32 meV",
        radii = collect(range(0.0, 2.5; length=500)),
        energies = collect(range(0.0, 3.32; length=200)),
    ),
]

# ------------------------------ Utilities ------------------------------------
function intensities_matrix(res)
    # Many Sunny "intensities-like" objects act like arrays; try that first.
    try
        return Array(res)
    catch
    end
    # Fallback: try common field names.
    for nm in (:data, :I, :intensities, :vals, :S)
        if hasproperty(res, nm)
            return Array(getproperty(res, nm))
        end
    end
    error("Can't extract matrix from $(typeof(res)); fields=$(fieldnames(typeof(res)))")
end

function compute_table!(spec)
    tag = spec.tag
    desc = spec.desc
    radii = spec.radii
    energies = spec.energies

    @info "Computing table" tag desc nQ=length(radii) nW=length(energies) nsamp fwhm

    # Powder average: qs is a Vector of q-vectors for each sample on the sphere at |Q|=r
    res_pow = powder_average(cryst, radii, nsamp) do qs
        intensities(swt, qs; energies, kernel)
    end

    S_Qω = intensities_matrix(res_pow)
    if as_f32
        S_Qω = Float32.(S_Qω)
    end

    @info "Computed S_Qω" size=size(S_Qω) Q_range=(first(radii), last(radii)) W_range=(first(energies), last(energies))

    outfile = joinpath(outdir, "sunny_powder_bzro_$(tag).jld2")

    meta = (
        tag = tag,
        desc = desc,
        nsamp = nsamp,
        intrinsic_fwhm_meV = fwhm,
        float32 = as_f32,
        model = (
            name = "BZRO Ru1 (Ba3ZnRu2O9)",
            S = 3/2,
            g = 2.0,
            D = -0.665,
            J1 = 21.7,
            J2 = 5.65,
            J3 = 0.66,
            J4 = 0.085,
        ),
        cif = cif_path,
        sunny_version = _sunny_version(),
    )

    @save outfile radii energies S_Qω meta
    @info "Wrote" outfile
end

# ------------------------------ Main -----------------------------------------
for spec in TABLE_SPECS
    if only_tag !== nothing && spec.tag != only_tag
        continue
    end
    compute_table!(spec)
end

@info "Done."
