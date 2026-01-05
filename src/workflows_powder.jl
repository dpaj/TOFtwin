# ---------------------------------------------------------------------------
# Powder workflow helpers
#
# Provides:
#   - PowderCtx
#   - setup_powder_ctx(...)
#   - eval_powder(ctx, model; do_hist=true)
#
# Adds PyChop-driven timing curves (σt(t)) with optional "group-aware" (L2-aware)
# precomputed CDF work for GaussianTimingCDFResolution.
# ---------------------------------------------------------------------------

using Serialization
using LinearAlgebra

# -----------------------------
# Small helpers
# -----------------------------

# Robust bool env parsing (kept local to avoid depending on scripts)
_parse_bool(s::AbstractString) = lowercase(strip(s)) in ("1","true","t","yes","y","on")

# Linear interpolation (clamped) on sorted x-grid
function _lininterp(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, xq::Real)
    n = length(x)
    @assert length(y) == n
    xq <= x[1] && return float(y[1])
    xq >= x[end] && return float(y[end])
    i = searchsortedlast(x, xq)
    i = clamp(i, 1, n-1)
    x0 = float(x[i]); x1 = float(x[i+1])
    y0 = float(y[i]); y1 = float(y[i+1])
    t = (float(xq) - x0) / (x1 - x0)
    return (1-t)*y0 + t*y1
end

# Default python path for calling PyChop oracle.
# (Users can override with TOFTWIN_PYCHOP_PYTHON or PYTHON.)
function _default_pychop_python()
    return get(ENV, "TOFTWIN_PYCHOP_PYTHON",
        get(ENV, "PYTHON",
            Sys.iswindows() ? raw"C:\Users\vdp\AppData\Local\Microsoft\WindowsApps\python3.11.exe" : "python3"
        )
    )
end

# Default path to our oracle script (in scripts/)
_default_pychop_script() = normpath(joinpath(@__DIR__, "..", "scripts", "pychop_oracle_dE.py"))

# -----------------------------
# PyChop oracle: run python, parse curve
# -----------------------------
"""
_run_pychop_oracle_dE(; python, script, instrument, Ei_meV, variant, freq_hz, tc_index, use_tc_rss, delta_td_us, etrans_min, etrans_max, npts)

Runs `pychop_oracle_dE.py` and returns (etrans_meV::Vector{Float64}, dE_fwhm_meV::Vector{Float64}).

- dE is FWHM in energy transfer, as reported by the oracle (PyChop).
"""
function _run_pychop_oracle_dE(;
    python::AbstractString,
    script::AbstractString,
    instrument::Symbol,
    Ei_meV::Float64,
    variant::AbstractString,
    freq_hz::AbstractVector{<:Real},
    tc_index::Int,
    use_tc_rss::Bool,
    delta_td_us::Float64,
    etrans_min::Float64,
    etrans_max::Float64,
    npts::Int,
)
    # Write JSON to a temp file (avoid command-length problems)
    tmp = tempname() * ".json"
    open(tmp, "w") do io
        # Keep schema stable, simple
        print(io, "{")
        print(io, "\"instrument\":\"", String(instrument), "\",")
        print(io, "\"variant\":\"", variant, "\",")
        print(io, "\"Ei\":", Ei_meV, ",")
        print(io, "\"etrans_min\":", etrans_min, ",")
        print(io, "\"etrans_max\":", etrans_max, ",")
        print(io, "\"npts\":", npts, ",")
        print(io, "\"tc_index\":", tc_index, ",")
        print(io, "\"use_tc_rss\":", use_tc_rss ? "true" : "false", ",")
        print(io, "\"delta_td_us\":", delta_td_us, ",")
        print(io, "\"freq_hz\":[", join(string.(float.(freq_hz)), ","), "]")
        print(io, "}")
    end

    out = tempname() * ".txt"
    cmd = `$(python) $(script) --json $(tmp) --out $(out)`
    run(cmd)

    # Parse whitespace-separated columns: etrans  dE_FWHM
    et = Float64[]
    dE = Float64[]
    for ln in eachline(out)
        s = strip(ln)
        isempty(s) && continue
        startswith(s, "#") && continue
        parts = split(s)
        length(parts) < 2 && continue
        push!(et, parse(Float64, parts[1]))
        push!(dE, parse(Float64, parts[2]))
    end
    rm(tmp; force=true)
    rm(out; force=true)

    if isempty(et)
        error("PyChop oracle returned no points (instrument=$(instrument), Ei=$(Ei_meV)).")
    end
    return et, dE
end

# -----------------------------
# Group-aware σt(t) from PyChop ΔE(ω)
# -----------------------------
"""
sigma_t_work_from_pychop(inst, pixels, Ei_meV, tof_edges_s, res; ...)

Builds CDF smear work that is *group-aware* in L2:

- We get ΔE_FWHM(ω) from PyChop (oracle).
- For each (rounded) L2 bucket, we map TOF-bin centers -> ω(t;L2),
  compute σ_ω = ΔE_FWHM/2.355, then σt = σ_ω / |dω/dt|.
- We then precompute `TofSmearWork` for that σt(t) curve and reuse it
  for all pixels in the same L2 bucket.

Returns:
  (σt_ref_bins::Vector{Float64}, work_by_pixelid::Vector{Any}, meta::NamedTuple)

`work_by_pixelid[p.id]` can be passed as `cdf_work` into `predict_pixel_tof`.
"""
function sigma_t_work_from_pychop(
    inst::Instrument,
    pixels::Vector{DetectorPixel},
    Ei_meV::Float64,
    tof_edges_s::Vector{Float64},
    res::GaussianTimingCDFResolution;
    python::AbstractString=_default_pychop_python(),
    script::AbstractString=_default_pychop_script(),
    instrument::Symbol=Symbol(inst.name),
    variant::AbstractString="default",
    freq_hz::AbstractVector{<:Real}=Float64[],
    tc_index::Int=0,
    use_tc_rss::Bool=true,
    delta_td_us::Float64=0.0,
    npts::Int=401,
    etrans_min::Float64=0.0,
    etrans_max::Float64=min(Ei_meV-0.5, Ei_meV),
    sigma_q::Float64=0.0,
    l2_bucket_m::Float64=0.01,
    disk_cache::Bool=true,
    cache_dir::AbstractString=joinpath(@__DIR__, "..", "scripts", ".toftwin_cache"),
    cache_tag::AbstractString="",
)
    # Cache the ΔE curve (small) on disk keyed by config
    # NOTE: keep the on-disk filename portable (Windows disallows characters like '|').
    # Put all config into the hashed "parts" key, and keep the prefix simple.
    curve_path = _wf_cache_path(cache_dir, "pychop_dE";
        parts = (
            "tag=$(cache_tag)",
            "instrument=$(instrument)",
            "Ei=$(Ei_meV)",
            "variant=$(variant)",
            "freq=$(join(string.(float.(freq_hz)), ","))",
            "tci=$(tc_index)",
            "rss=$(use_tc_rss ? 1 : 0)",
            "dtd=$(delta_td_us)",
            "emin=$(etrans_min)",
            "emax=$(etrans_max)",
            "sigma_q=$(sigma_q)",
            "npts=$(npts)",
        )
    )

    etrans, dE_fwhm = _wf_load_or_compute(curve_path, () -> begin
        _run_pychop_oracle_dE(
            python=python,
            script=script,
            instrument=instrument,
            Ei_meV=Ei_meV,
            variant=variant,
            freq_hz=freq_hz,
            tc_index=tc_index,
            use_tc_rss=use_tc_rss,
            delta_td_us=delta_td_us,
            etrans_min=etrans_min,
            etrans_max=etrans_max,
            npts=npts,
        )
    end; disk_cache=disk_cache)

    etrans = Vector{Float64}(etrans)
    dE_fwhm = Vector{Float64}(dE_fwhm)

    n_tof = length(tof_edges_s) - 1
    t_cent = 0.5 .* (tof_edges_s[1:end-1] .+ tof_edges_s[2:end])

    # Representative pixel for returning σt_ref
    p_ref = pixels[clamp(Int(round(length(pixels)/2)), 1, length(pixels))]
    L2_ref = L2(inst, p_ref.id)

    function sigma_t_bins_for_L2(L2p::Float64)
        σt_bins = similar(t_cent)
        for i in eachindex(t_cent)
            t = t_cent[i]
            Ef = Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            if !(Ef > 0)
                σt_bins[i] = 0.0
                continue
            end
            ω = Ei_meV - Ef
            # Clamp to oracle curve domain
            ωc = clamp(ω, etrans[1], etrans[end])
            dE = _lininterp(etrans, dE_fwhm, ωc) # FWHM meV
            σω = dE / 2.354820045               # sigma meV
            dwdt = abs(dω_dt(inst.L1, L2p, Ei_meV, t))  # meV/s
            σt_bins[i] = (dwdt > 0) ? (σω / dwdt) : 0.0
        end
        return σt_bins
    end

    # Work objects are identical for pixels in same L2 bucket
    work_cache = Dict{Int,Any}()
    maxid = maximum(p.id for p in pixels)
    work_by_id = Vector{Any}(undef, maxid)
    fill!(work_by_id, nothing)

    bucket_key(L2p) = Int(round(L2p / l2_bucket_m))

    for p in pixels
        L2p = L2(inst, p.id)
        key = bucket_key(L2p)
        w = get!(work_cache, key) do
            σt_bins = sigma_t_bins_for_L2(L2p)
            # Precompute overlap work for these σt bins. We can use the public helper.
            resL = GaussianTimingCDFResolution(σt_bins; nsigma=res.nsigma)
            precompute_tof_smear_work(inst, pixels, Ei_meV, tof_edges_s, resL)
        end
        work_by_id[p.id] = w
    end

    σt_ref = sigma_t_bins_for_L2(L2_ref)

    meta = (
        etrans=etrans,
        etrans_meV = etrans,
        dE_fwhm=dE_fwhm,
        L2_ref=L2_ref,
        l2_bucket_m=l2_bucket_m,
        nbuckets=length(work_cache),
        python=String(python),
        script=String(script),
    )

    return σt_ref, work_by_id, meta
end

# --- sanity check: compare our timing-based smear to the "bare" PyChop ΔE(E)

function _lininterp1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, x0::Real)
    n = length(x)
    n == 0 && return NaN
    n == 1 && return float(y[1])
    # clamp to range
    if x0 <= x[1]
        return float(y[1])
    elseif x0 >= x[end]
        return float(y[end])
    end
    i = searchsortedlast(x, x0)
    i = clamp(i, 1, n-1)
    x1, x2 = float(x[i]), float(x[i+1])
    y1, y2 = float(y[i]), float(y[i+1])
    t = (float(x0) - x1) / (x2 - x1)
    return (1-t)*y1 + t*y2
end

function _fwhm_monotonic_x(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n == 0 && return NaN
    # ignore negative values
    ymax = maximum(y)
    ymax <= 0 && return NaN
    half = 0.5 * ymax
    imax = argmax(y)

    # left crossing
    xL = NaN
    for i in imax:-1:2
        y1, y2 = y[i-1], y[i]
        if y1 < half && y2 >= half
            t = (half - y1) / (y2 - y1)
            xL = float(x[i-1]) + t*(float(x[i]) - float(x[i-1]))
            break
        end
    end

    # right crossing
    xR = NaN
    for i in imax:1:(n-1)
        y1, y2 = y[i], y[i+1]
        if y1 >= half && y2 < half
            t = (half - y1) / (y2 - y1)
            xR = float(x[i]) + t*(float(x[i+1]) - float(x[i]))
            break
        end
    end

    return (isfinite(xL) && isfinite(xR)) ? (xR - xL) : NaN
end

"""Compare TOFtwin's timing-smear *energy* FWHM against PyChop's ΔE(E) at a few Etransfer points.

This is meant as a *sanity check* that our use of PyChop-derived timing variances is internally consistent.
"""
function pychop_check_dE(
    inst::Instrument,
    pixels::AbstractVector,
    Ei_meV::Real,
    tof_edges_s::AbstractVector{<:Real},
    resolution::GaussianTimingCDFResolution,
    cdf_work,
    meta;
    etrans_test_meV::AbstractVector{<:Real}=Float64[],
    npix_test::Int=5,
)

    # Normalize oracle output field names (older versions used :etrans/:dE_fwhm).
    if !hasproperty(meta, :etrans_meV) && hasproperty(meta, :etrans)
        meta = (; meta..., etrans_meV = meta.etrans)
    end
    if !hasproperty(meta, :dE_fwhm_meV) && hasproperty(meta, :dE_fwhm)
        meta = (; meta..., dE_fwhm_meV = meta.dE_fwhm)
    end

    # choose a few representative pixels by L2 (min / mid / max)
    L2s = [L2(inst, p.id) for p in pixels]
    perm = sortperm(L2s)
    picks = unique(clamp.(round.(Int, LinRange(1, length(pixels), npix_test)), 1, length(pixels)))
    pix_test = [pixels[perm[i]] for i in picks]

    # choose etrans points
    if isempty(etrans_test_meV)
        emax = min(float(Ei_meV) - 1e-3, maximum(meta.etrans_meV))
        etrans_test_meV = Float64[0.0, 0.5, 1.0, 3.0, 6.0, 0.9*emax]
    end
    etrans_test_meV = [float(x) for x in etrans_test_meV if x >= minimum(meta.etrans_meV) && x <= min(float(Ei_meV)-1e-3, maximum(meta.etrans_meV))]

    results = NamedTuple[]

    ntof = length(tof_edges_s) - 1
    # Precompute ω edges for each pixel lazily (depends on L2).

    for p in pix_test
        L2p = L2(inst, p.id)
        # ω edges corresponding to tof_edges
        ω_edges = similar(tof_edges_s, Float64)
        for i in eachindex(tof_edges_s)
            t = float(tof_edges_s[i])
            Ef = Ef_from_tof(inst.L1, L2p, float(Ei_meV), t)
            ω_edges[i] = float(Ei_meV) - Ef
        end
        ω_centers = (ω_edges[1:end-1] .+ ω_edges[2:end]) ./ 2
        dω = ω_edges[2:end] .- ω_edges[1:end-1]

        for ω0 in etrans_test_meV
            Ef0 = float(Ei_meV) - ω0
            Ef0 <= 0 && continue
            t0 = tof_from_EiEf(inst.L1, L2p, float(Ei_meV), Ef0)
            # bin index for impulse
            i0 = searchsortedlast(tof_edges_s, t0)
            i0 = clamp(i0, 1, ntof)

            C = zeros(Float64, 1, ntof)
            C[1, i0] = 1.0
            apply_tof_resolution!(C, inst, [p], float(Ei_meV), collect(float.(tof_edges_s)), resolution; work=cdf_work)
            yω = vec(C) ./ dω
            dE_toftwin = _fwhm_monotonic_x(ω_centers, yω)
            dE_pychop = _lininterp1(meta.etrans_meV, meta.dE_fwhm_meV, ω0)
            push!(results, (
                pixel_id = p.id,
                L2_m = L2p,
                etrans_meV = ω0,
                dE_pychop_FWHM_meV = dE_pychop,
                dE_toftwin_FWHM_meV = dE_toftwin,
                rel_err = (dE_toftwin - dE_pychop) / dE_pychop,
            ))
        end
    end

    if !isempty(results)
        rels = [abs(r.rel_err) for r in results if isfinite(r.rel_err)]
        maxrel = isempty(rels) ? NaN : maximum(rels)
        @info "PyChop ΔE check" n=length(results) max_abs_rel_err=maxrel
    end

    return results
end

"""Compare TOFtwin's σt(t) (from PyChop) against PyChop's ΔE(E) using the Jacobian dω/dt.

This avoids calling apply_tof_resolution! (which can be crash-prone on Windows when looped).
"""
function pychop_check_dE_2(
    inst::Instrument,
    pixels::AbstractVector,
    Ei_meV::Real,
    tof_edges_s::AbstractVector{<:Real},
    resolution::GaussianTimingCDFResolution,
    cdf_work,
    meta;
    etrans_test_meV::AbstractVector{<:Real}=Float64[],
    npix_test::Int=3,
    dt_s::Float64=1e-6,   # finite-difference step for dω/dt
)

    # Normalize oracle output field names (older versions used :etrans/:dE_fwhm).
    if !hasproperty(meta, :etrans_meV) && hasproperty(meta, :etrans)
        meta = (; meta..., etrans_meV = meta.etrans)
    end
    if !hasproperty(meta, :dE_fwhm_meV) && hasproperty(meta, :dE_fwhm)
        meta = (; meta..., dE_fwhm_meV = meta.dE_fwhm)
    end

    Ei = float(Ei_meV)

    # choose a few representative pixels by L2 (min / mid / max)
    L2s = [L2(inst, p.id) for p in pixels]
    perm = sortperm(L2s)
    picks = unique(clamp.(round.(Int, LinRange(1, length(pixels), npix_test)), 1, length(pixels)))
    pix_test = [pixels[perm[i]] for i in picks]

    # choose etrans points
    if isempty(etrans_test_meV)
        emax = min(Ei - 1e-3, maximum(meta.etrans_meV))
        etrans_test_meV = Float64[0.0, 0.5, 1.0, 3.0, 6.0, 0.9*emax]
    end
    etrans_test_meV = [float(x) for x in etrans_test_meV
                       if x >= minimum(meta.etrans_meV) && x <= min(Ei-1e-3, maximum(meta.etrans_meV))]

    # tiny helper: ω(t) and dω/dt
    ω_of_t(L2p, t) = Ei - Ef_from_tof(inst.L1, L2p, Ei, t)
    function dωdt(L2p, t0)
        tlo = max(t0 - dt_s, eps(Float64))
        thi = t0 + dt_s
        return (ω_of_t(L2p, thi) - ω_of_t(L2p, tlo)) / (thi - tlo)
    end

    results = NamedTuple[]

    for p in pix_test
        L2p = L2(inst, p.id)

        for ω0 in etrans_test_meV
            Ef0 = Ei - ω0
            Ef0 <= 0 && continue

            t0 = tof_from_EiEf(inst.L1, L2p, Ei, Ef0)

            # σt at this (pixel, Ei, t0). This supports scalar/vector/function σt.
            σt0 = TOFtwin._σt(resolution, inst, p, Ei, t0)  # seconds

            # Convert to ΔE_FWHM via Jacobian
            dE_toftwin = 2.354820045 * σt0 * abs(dωdt(L2p, t0))

            dE_pychop = _lininterp1(meta.etrans_meV, meta.dE_fwhm_meV, ω0)

            push!(results, (
                pixel_id = p.id,
                L2_m = L2p,
                etrans_meV = ω0,
                dE_pychop_FWHM_meV = dE_pychop,
                dE_toftwin_FWHM_meV = dE_toftwin,
                rel_err = (dE_toftwin - dE_pychop) / dE_pychop,
            ))
        end
    end

    if !isempty(results)
        rels = [abs(r.rel_err) for r in results if isfinite(r.rel_err)]
        maxrel = isempty(rels) ? NaN : maximum(rels)
        @info "PyChop ΔE check (Jacobian)" n=length(results) max_abs_rel_err=maxrel
    end

    return results
end


# -----------------------------
# Context + evaluation
# -----------------------------

struct PowderCtx
    instr::Symbol
    idf_path::String
    inst::Instrument
    pixels::Vector{DetectorPixel}

    Ei_meV::Float64
    tof_edges_s::Vector{Float64}
    Q_edges::Vector{Float64}
    ω_edges::Vector{Float64}

    resolution::AbstractResolutionModel
    cdf_work::Any              # nothing | TofSmearWork | Vector{TofSmearWork} (by pixel id)

    Cv::Any                    # cached vanadium (pixel×TOF) response (flat kernel)
    rmap::Any                  # cached reduction map

    disk_cache::Bool
    cache_dir::String
end

"""
setup_powder_ctx(; ...)

Builds a `PowderCtx` that contains:
- instrument + (optionally grouped/masked) pixels
- TOF edges, Q/ω edges
- resolution model
- (optional) precomputed CDF smear work
- (optional) cached Cv + reduce-map

Key knobs:
- res_mode = "none" | "gh" | "cdf"
- σt_source = "env" | "pychop"
"""
function setup_powder_ctx(;
    instr::Symbol,
    idf_path::String,

    # kernel bounds (for axes)
    kern_Qmin::Float64,
    kern_Qmax::Float64,
    kern_wmin::Float64,
    kern_wmax::Float64,

    rebuild_geom::Bool=false,
    ψstride::Int=1,
    ηstride::Int=1,

    grouping::AbstractString="none",
    grouping_file::Union{Nothing,String}=nothing,
    mask_btp::AbstractString="",
    mask_mode::Symbol=:drop,
    angle_step::Float64=1.0,

    Ei::Float64=12.0,
    nQbins::Int=220,
    nωbins::Int=240,

    # resolution
    res_mode::AbstractString="cdf",
    σt_us::Float64=100.0,
    σt_source::AbstractString="env",
    gh_order::Int=3,
    nsigma::Float64=6.0,

    # CDF / striping control
    ntof_env::AbstractString="auto",
    ntof_alpha::Float64=0.5,
    ntof_beta::Float64=1/3,
    ntof_min::Int=200,
    ntof_max::Int=2000,

    # caching
    disk_cache::Bool=true,
    cache_dir::AbstractString=joinpath(@__DIR__, "..", "scripts", ".toftwin_cache"),
    cache_ver::AbstractString="v1",
    cache_Cv::Bool=true,
    cache_rmap::Bool=true,

    # PyChop knobs
    pychop_python::AbstractString=_default_pychop_python(),
    pychop_script::AbstractString=_default_pychop_script(),
    pychop_variant::AbstractString="default",
    pychop_freq_hz::AbstractVector{<:Real}=Float64[],
    pychop_tc_index::Int=0,
    pychop_use_tc_rss::Bool=true,
    pychop_delta_td_us::Float64=0.0,
    pychop_npts::Int=401,
    pychop_l2_bucket_m::Float64=0.01,
    pychop_sigma_q::Float64=0.0,
    pychop_check::Bool=false,
    pychop_check_etrans_meV::AbstractVector{<:Real}=Float64[],
)
    # -------------------------
    # Load instrument
    # -------------------------
    out = if isdefined(TOFtwin.MantidIDF, :load_mantid_idf_diskcached)
        TOFtwin.MantidIDF.load_mantid_idf_diskcached(idf_path; rebuild=rebuild_geom)
    else
        TOFtwin.MantidIDF.load_mantid_idf(idf_path)
    end
    inst = out.inst
    bank = (hasproperty(out, :bank) && out.bank !== nothing) ? out.bank : DetectorBank(inst.name, out.pixels)

    # Pixel selection stride (fast, before grouping)
    pix0 = sample_pixels(bank, AngularDecimate(ψstride, ηstride))

    # Apply grouping/masking if requested (can also be "none")
    pix, meta = apply_grouping_masking(pix0;
        instrument=instr,
        grouping=grouping,
        grouping_file=grouping_file,
        mask_btp=mask_btp,
        mask_mode=mask_mode,
        outdir=joinpath(@__DIR__, "..", "scripts", "out"),
        angle_step=angle_step,
        return_meta=true,
    )

    # -------------------------
    # Axes
    # -------------------------
    Ei_meV = Ei
    Q_edges = collect(range(kern_Qmin, kern_Qmax; length=nQbins+1))
    ωlo = min(-2.0, kern_wmin)
    ωhi = max(Ei_meV, kern_wmax)
    ω_edges = collect(range(ωlo, ωhi; length=nωbins+1))

    # TOF window based on geometry
    L2min = minimum(L2(inst, p.id) for p in pix)
    L2max = maximum(L2(inst, p.id) for p in pix)
    #Ef_min, Ef_max = 1.0, Ei_meV
    Ef_min, Ef_max = 0.1*Ei_meV, Ei_meV
    tmin = tof_from_EiEf(inst.L1, L2min, Ei_meV, Ef_max)
    tmax = tof_from_EiEf(inst.L1, L2max, Ei_meV, Ef_min)

    # Pick NTOF
    ntof = 0
    ntof_s = lowercase(String(ntof_env))
    if ntof_s in ("auto","suggest","0")
        if lowercase(String(res_mode)) == "cdf" && (σt_us > 0)
            σt_s = σt_us * 1e-6
            ntof, dt, maxd, dωbin = suggest_ntof(inst, pix, Ei_meV, tmin, tmax, ω_edges, σt_s;
                α=ntof_alpha, β=ntof_beta, ntof_min=ntof_min, ntof_max=ntof_max)
        else
            ntof = 500
        end
    else
        ntof = parse(Int, ntof_s)
    end
    tof_edges_s = collect(range(tmin, tmax; length=ntof+1))

    # -------------------------
    # Resolution model + precomputed work (CDF)
    # -------------------------
    res_mode_s = lowercase(String(res_mode))
    σt_source_s = lowercase(String(σt_source))

    resolution =
        (res_mode_s == "none" || σt_us <= 0) ? NoResolution() :
        (res_mode_s == "gh")  ? GaussianTimingResolution(σt_us * 1e-6; order=gh_order) :
        (res_mode_s == "cdf") ? GaussianTimingCDFResolution(σt_us * 1e-6; nsigma=nsigma) :
        throw(ArgumentError("res_mode must be none|gh|cdf (got '$res_mode')"))

    cdf_work = nothing
    # If CDF mode: precompute (either uniform σt or group-aware σt(t) from PyChop)
    if resolution isa GaussianTimingCDFResolution
        if σt_source_s == "env"
            # scalar σt -> uniform work
            cdf_work = precompute_tof_smear_work(inst, pix, Ei_meV, tof_edges_s, resolution)
        elseif σt_source_s == "pychop"
            σt_ref, work_by_id, meta_pychop = sigma_t_work_from_pychop(
                inst, pix, Ei_meV, tof_edges_s, resolution;
                python=pychop_python,
                script=pychop_script,
                instrument=instr,
                variant=pychop_variant,
                freq_hz=pychop_freq_hz,
                tc_index=pychop_tc_index,
                use_tc_rss=pychop_use_tc_rss,
                delta_td_us=pychop_delta_td_us,
                npts=pychop_npts,
                etrans_min=0.0,
                etrans_max=min(Ei_meV-0.5, Ei_meV),
                sigma_q=pychop_sigma_q,
                l2_bucket_m=pychop_l2_bucket_m,
                disk_cache=disk_cache,
                cache_dir=cache_dir,
                cache_tag="powder|" * cache_ver,
            )
            cdf_work = work_by_id

            if pychop_check
                try
                    pychop_check_dE(
                        inst, pix, Ei_meV, tof_edges_s, resolution, cdf_work, meta_pychop;
                        etrans_test_meV=collect(float.(pychop_check_etrans_meV))
                    )
                catch err
                    @warn "PyChop dE sanity check failed" exception=(err, catch_backtrace())
                end
            end
        else
            throw(ArgumentError("σt_source must be env|pychop (got '$σt_source')"))
        end
    end

    # -------------------------
    # Cached Cv + reduction map
    # -------------------------
    Cv = nothing
    if disk_cache && cache_Cv
        # Keep cache filenames portable on Windows (no '|').
        Cv_path = _wf_cache_path(cache_dir, "Cv_flat_" * String(instr) * "_" * cache_ver;
            parts=("Ei=$(Ei_meV)", "ntof=$(ntof)", "res=$(typeof(resolution))")
        )
        Cv = _wf_load_or_compute(Cv_path, () -> begin
            predict_pixel_tof(inst;
                pixels=pix, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
                model=(Q,ω)->1.0, resolution=resolution, cdf_work=cdf_work
            )
        end; disk_cache=true)
    else
        Cv = predict_pixel_tof(inst;
            pixels=pix, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            model=(Q,ω)->1.0, resolution=resolution, cdf_work=cdf_work
        )
    end

    rmap = nothing
    if cache_rmap
        # rmap is instrument/Ei/tof-dependent only, not model-dependent
        rmap = precompute_pixel_tof_Qω_map(inst;
            pixels=pix, Ei_meV=Ei_meV, tof_edges_s=tof_edges_s,
            cache=disk_cache, cache_dir=cache_dir, cache_tag="powder|" * cache_ver
        )
    end

    return PowderCtx(
        instr, idf_path, inst, pix,
        Ei_meV, tof_edges_s, Q_edges, ω_edges,
        resolution, cdf_work,
        Cv, rmap,
        disk_cache, String(cache_dir)
    )
end

"""
eval_powder(ctx, model; do_hist=true, eps=1e-12)

Per-model evaluation:
- Cs = predict_pixel_tof(...)
- Cnorm = normalize_by_vanadium(Cs, ctx.Cv)
- Optionally reduce to (|Q|,ω) maps using ctx.rmap
"""
function eval_powder(ctx::PowderCtx, model; do_hist::Bool=true, eps::Float64=1e-12)
    Cs = predict_pixel_tof(ctx.inst;
        pixels=ctx.pixels,
        Ei_meV=ctx.Ei_meV,
        tof_edges_s=ctx.tof_edges_s,
        model=model,
        resolution=ctx.resolution,
        cdf_work=ctx.cdf_work,
    )

    Cnorm = normalize_by_vanadium(Cs, ctx.Cv; eps=eps)

    if !do_hist
        return (Cs=Cs, Cv=ctx.Cv, Cnorm=Cnorm)
    end

    Hraw = reduce_pixel_tof_to_Qω_powder(ctx.inst;
        pixels=ctx.pixels, Ei_meV=ctx.Ei_meV, tof_edges_s=ctx.tof_edges_s,
        C=Cs, Q_edges=ctx.Q_edges, ω_edges=ctx.ω_edges,
        map=ctx.rmap
    )

    Hsum = reduce_pixel_tof_to_Qω_powder(ctx.inst;
        pixels=ctx.pixels, Ei_meV=ctx.Ei_meV, tof_edges_s=ctx.tof_edges_s,
        C=Cnorm, Q_edges=ctx.Q_edges, ω_edges=ctx.ω_edges,
        map=ctx.rmap
    )

    W = zeros(size(Cnorm))
    W[ctx.Cv .> 0.0] .= 1.0
    Hwt = reduce_pixel_tof_to_Qω_powder(ctx.inst;
        pixels=ctx.pixels, Ei_meV=ctx.Ei_meV, tof_edges_s=ctx.tof_edges_s,
        C=W, Q_edges=ctx.Q_edges, ω_edges=ctx.ω_edges,
        map=ctx.rmap
    )

    Hmean = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Hraw=Hraw, Hsum=Hsum, Hwt=Hwt, Hmean=Hmean, Cs=Cs, Cv=ctx.Cv, Cnorm=Cnorm)
end
