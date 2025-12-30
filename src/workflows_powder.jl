
# -----------------------------------------------------------------------------
# workflows_powder.jl
#
# Powder "setup ctx" vs "evaluate model" workflow, Makie-free.
# Designed for:
#   - Do expensive setup once (instrument/pixels, axes, TOF, resolution, CDF-work,
#     Cv(flat), reduce-map)
#   - Then iterate many models: only Cs + reduce
#
# Uses grouping/masking via apply_grouping_masking (Mantid-style analogs).
# -----------------------------------------------------------------------------

# -----------------------------
# Reduce-map for fast powder reduction
# -----------------------------
struct ReduceMapPowder
    iQ::Matrix{Int32}
    iW::Matrix{Int32}
    fQ::Matrix{Float32}
    fW::Matrix{Float32}
    valid::BitMatrix
end

# -----------------------------
# PyChop effective σt(t) helper (stdlib-only)
# -----------------------------

const _FWHM_PER_SIGMA = 2.3548200450309493

@inline function _median_real(x::AbstractVector{<:Real})
    n = length(x)
    n == 0 && throw(ArgumentError("median of empty vector"))
    v = sort!(collect(Float64.(x)))
    if isodd(n)
        return v[(n+1) >>> 1]
    else
        i = n >>> 1
        return 0.5 * (v[i] + v[i+1])
    end
end

"""Linear interpolation y(x) at query xq with clamped endpoints."""
@inline function _interp1_clamped(x::Vector{Float64}, y::Vector{Float64}, xq::Float64)
    n = length(x)
    n == 0 && return NaN
    xq <= x[1]  && return y[1]
    xq >= x[end] && return y[end]
    i = searchsortedlast(x, xq)
    i < 1 && return y[1]
    i >= n && return y[end]
    x0 = x[i]; x1 = x[i+1]
    y0 = y[i]; y1 = y[i+1]
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
end

function _run_pychop_oracle_dE(; python::AbstractString,
    script::AbstractString,
    instrument::Symbol,
    Ei_meV::Real,
    variant::AbstractString="",
    freq_hz::AbstractVector{<:Real}=Float64[],
    tc_index::Int=0,
    use_tc_rss::Bool=true,
    delta_td_us::Real=0.0,
    etrans_min_meV::Real=0.0,
    etrans_max_meV::Real=0.0,
    npts::Int=401)

    args = String[
        String(python),
        String(script),
        "--instrument", String(instrument),
        "--Ei", string(Float64(Ei_meV)),
        "--etrans-min", string(Float64(etrans_min_meV)),
        "--etrans-max", string(Float64(etrans_max_meV)),
        "--npts", string(Int(npts)),
        "--tc-index", string(Int(tc_index)),
        "--delta-td-us", string(Float64(delta_td_us)),
        "--tc-mode", (use_tc_rss ? "rss" : "first"),
    ]
    if !isempty(variant)
        append!(args, ["--variant", String(variant)])
    end
    if !isempty(freq_hz)
        fstr = join(string.(Float64.(freq_hz)), ",")
        append!(args, ["--freq", fstr])
    end

    cmd = Cmd(args)
    out = try
        read(cmd, String)
    catch err
        throw(ErrorException("PyChop oracle failed. Command was: $(repr(cmd))\nError: $(err)"))
    end

    xs = Float64[]
    ys = Float64[]
    for line in split(out, '\n')
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        cols = split(s)
        length(cols) < 2 && continue
        push!(xs, parse(Float64, cols[1]))
        push!(ys, parse(Float64, cols[2]))
    end
    isempty(xs) && throw(ErrorException("PyChop oracle returned no data. Output:\n" * out))
    return xs, ys
end

"""
    _pychop_effective_sigma_t_bins(inst, pixels, Ei_meV, tof_edges_s; ...)

Compute an *effective* TOF-domain σt(t) per TOF bin such that the implied
energy-width roughly matches a PyChop ΔE(ω) curve:

    σ_ω(t) ≈ (ΔE_FWHM(ω(t)) / 2.355)
    σt(t)  =  σ_ω(t) / |dω/dt|

This is a pragmatic bridge: it preserves PyChop's configuration → resolution mapping,
but expresses it as a TOF-domain convolution width that we can feed to the CDF kernel.
"""
function _pychop_effective_sigma_t_bins(inst::Instrument,
        pixels::Vector{DetectorPixel},
        Ei_meV::Float64,
        tof_edges_s::Vector{Float64};
        python::AbstractString="python",
        script::AbstractString=joinpath(@__DIR__, "..", "scripts", "pychop_oracle_dE.py"),
        instrument::Symbol=:CNCS,
        variant::AbstractString="",
        freq_hz::AbstractVector{<:Real}=Float64[],
        tc_index::Int=0,
        use_tc_rss::Bool=true,
        delta_td_us::Float64=0.0,
        npts::Int=401)

    # Representative L2 for mapping ω(t) and dω/dt.
    L2s = [L2(inst, p.id) for p in pixels]
    L2ref = _median_real(L2s)

    # ω range we need (from TOF bin centers)
    nT = length(tof_edges_s) - 1
    ω_cent = Vector{Float64}(undef, nT)
    @inbounds for it in 1:nT
        t = 0.5 * (tof_edges_s[it] + tof_edges_s[it+1])
        Ef = try
            Ef_from_tof(inst.L1, L2ref, Ei_meV, t)
        catch
            ω_cent[it] = NaN
            continue
        end
        ω_cent[it] = Ei_meV - Ef
    end

    ωf = filter(isfinite, ω_cent)
    isempty(ωf) && throw(ArgumentError("No finite ω values found from TOF window; can't build PyChop σt(t)"))
    ωmin = minimum(ωf)
    ωmax = maximum(ωf)

    # Ask the oracle for a smooth ΔE_FWHM(ω) curve over that domain.
    xs, dE_fwhm = _run_pychop_oracle_dE(
        python=python,
        script=script,
        instrument=instrument,
        Ei_meV=Ei_meV,
        variant=variant,
        freq_hz=freq_hz,
        tc_index=tc_index,
        use_tc_rss=use_tc_rss,
        delta_td_us=delta_td_us,
        etrans_min_meV=ωmin,
        etrans_max_meV=ωmax,
        npts=npts,
    )

    # Convert ΔE_FWHM(ω) to σ_ω, then to σt via local derivative.
    σt_bins = zeros(Float64, nT)
    @inbounds for it in 1:nT
        ω = ω_cent[it]
        if !isfinite(ω)
            σt_bins[it] = 0.0
            continue
        end
        dE = _interp1_clamped(xs, dE_fwhm, ω)               # meV (FWHM)
        σω = dE / _FWHM_PER_SIGMA                           # meV (σ)
        t = 0.5 * (tof_edges_s[it] + tof_edges_s[it+1])
        dωdt = abs(dω_dt(inst.L1, L2ref, Ei_meV, t))         # meV/s
        σt_bins[it] = (dωdt > 0) ? (σω / dωdt) : 0.0         # s
    end
    return σt_bins
end

"""
    build_reduce_map_powder(inst, pixels, Ei_meV, tof_edges_s, Q_edges, ω_edges)

Precompute mapping from (pixel, tof-bin-center) -> (|Q|, ω) bin indices plus
bilinear weights. This makes repeated reductions much faster.
"""
function build_reduce_map_powder(inst::Instrument, pixels::Vector{DetectorPixel},
        Ei_meV::Float64, tof_edges_s::Vector{Float64},
        Q_edges::Vector{Float64}, ω_edges::Vector{Float64})

    nP = length(pixels)
    nT = length(tof_edges_s) - 1
    nQ = length(Q_edges) - 1
    nW = length(ω_edges) - 1

    iQ = fill(Int32(0), nP, nT)
    iW = fill(Int32(0), nP, nT)
    fQ = fill(Float32(0), nP, nT)
    fW = fill(Float32(0), nP, nT)
    valid = falses(nP, nT)

    @inbounds for ip in 1:nP
        p = pixels[ip]
        L2p = L2(inst, p.id)
        for it in 1:nT
            t = 0.5 * (tof_edges_s[it] + tof_edges_s[it+1])

            Ef = try
                Ef_from_tof(inst.L1, L2p, Ei_meV, t)
            catch
                continue
            end
            (Ef <= 0 || Ef > Ei_meV) && continue

            Qvec, ω = Qω_from_pixel(p.r_L, Ei_meV, Ef; r_samp_L=inst.r_samp_L)
            Qmag = norm(Qvec)

            iq = searchsortedlast(Q_edges, Qmag)
            iw = searchsortedlast(ω_edges, ω)
            if iq < 1 || iq >= nQ || iw < 1 || iw >= nW
                continue
            end

            dq = Q_edges[iq+1] - Q_edges[iq]
            dw = ω_edges[iw+1] - ω_edges[iw]
            (dq <= 0 || dw <= 0) && continue

            fq = Float32((Qmag - Q_edges[iq]) / dq)
            fw = Float32((ω    - ω_edges[iw]) / dw)
            fq = min(max(fq, 0.0f0), 1.0f0)
            fw = min(max(fw, 0.0f0), 1.0f0)

            iQ[ip,it] = Int32(iq)
            iW[ip,it] = Int32(iw)
            fQ[ip,it] = fq
            fW[ip,it] = fw
            valid[ip,it] = true
        end
    end

    return ReduceMapPowder(iQ, iW, fQ, fW, valid)
end

"""
    reduce_three!(Hraw, Hsum, Hwt, pixels, rmap, Cs, Cv; eps=1e-12)

One-pass powder reduction with bilinear weights:
- Hraw gets raw SUM (uses Cs)
- Hsum gets SUM of vanadium-normalized values (Cs/Cv)
- Hwt gets SUM of weights
Then MEAN can be computed as Hsum./(Hwt+eps).
"""
function reduce_three!(Hraw::Hist2D, Hsum::Hist2D, Hwt::Hist2D,
        pixels::Vector{DetectorPixel}, rmap::ReduceMapPowder,
        Cs::AbstractMatrix, Cv::AbstractMatrix; eps=1e-12)

    nP = length(pixels)
    nT = size(rmap.valid, 2)

    @inbounds for ip in 1:nP
        pid = pixels[ip].id
        for it in 1:nT
            rmap.valid[ip,it] || continue
            cv = Cv[pid, it]
            cv <= 0.0 && continue

            cs = Cs[pid, it]
            iq = Int(rmap.iQ[ip,it])
            iw = Int(rmap.iW[ip,it])
            fq = Float64(rmap.fQ[ip,it])
            fw = Float64(rmap.fW[ip,it])

            w00 = (1 - fq) * (1 - fw)
            w10 = fq       * (1 - fw)
            w01 = (1 - fq) * fw
            w11 = fq       * fw

            Hraw.counts[iq,   iw]   += w00 * cs
            Hraw.counts[iq+1, iw]   += w10 * cs
            Hraw.counts[iq,   iw+1] += w01 * cs
            Hraw.counts[iq+1, iw+1] += w11 * cs

            v = cs / (cv + eps)
            Hsum.counts[iq,   iw]   += w00 * v
            Hsum.counts[iq+1, iw]   += w10 * v
            Hsum.counts[iq,   iw+1] += w01 * v
            Hsum.counts[iq+1, iw+1] += w11 * v

            Hwt.counts[iq,   iw]   += w00
            Hwt.counts[iq+1, iw]   += w10
            Hwt.counts[iq,   iw+1] += w01
            Hwt.counts[iq+1, iw+1] += w11
        end
    end
    return nothing
end

# -----------------------------
# Context
# -----------------------------
"""
    PowderCtx

Holds all model-independent setup products for powder forward prediction:
instrument, selected pixels (after grouping/masking/decimation), axes, TOF edges,
resolution object + optional CDF work, Cv(flat), and a precomputed reduce-map.
"""
struct PowderCtx{R,W}
    instr::Symbol
    idf_path::String
    inst::Instrument
    pixels::Vector{DetectorPixel}
    Ei::Float64
    tof_edges::Vector{Float64}
    Q_edges::Vector{Float64}
    ω_edges::Vector{Float64}
    resolution::R
    cdf_work::W
    Cv::Matrix{Float64}
    rmap::ReduceMapPowder
end

"""
    setup_powder_ctx(; ...)

Build a powder forward-prediction context.
This is the "setup" part to run once and reuse for many models.

You pass kernel domain bounds (Qmin/Qmax/wmin/wmax) rather than a kernel object.
This keeps the workflow code independent from any particular kernel source.
"""
function setup_powder_ctx(;
    instr::Symbol,
    idf_path::AbstractString,
    kern_Qmin::Float64,
    kern_Qmax::Float64,
    kern_wmin::Float64,
    kern_wmax::Float64,
    rebuild_geom::Bool=false,
    ψstride::Int=1,
    ηstride::Int=1,
    grouping::AbstractString="",
    grouping_file::Union{Nothing,AbstractString}=nothing,
    mask_btp::AbstractString="",
    mask_mode::Symbol=:drop,
    angle_step::Real=0.5,
    Ei::Float64,
    nQbins::Int=220,
    nωbins::Int=240,
    res_mode::AbstractString="cdf",
    # Manual timing width (constant σt) in microseconds
    σt_us::Float64=100.0,
    # Timing width source:
    #   "manual"  -> use σt_us above
    #   "pychop"  -> derive an effective σt(t) curve from a PyChop ΔE(ω) curve
    #                and map it onto TOF-bin centers for a representative L2.
    σt_source::AbstractString="manual",
    # PyChop oracle settings (used when σt_source="pychop")
    pychop_python::AbstractString="python",
    pychop_script::AbstractString=joinpath(@__DIR__, "..", "scripts", "pychop_oracle_dE.py"),
    pychop_variant::AbstractString="",
    pychop_freq_hz::AbstractVector{<:Real}=Float64[],
    pychop_tc_index::Int=0,
    pychop_use_tc_rss::Bool=true,
    pychop_delta_td_us::Float64=0.0,
    pychop_npts::Int=401,
    gh_order::Int=3,
    nsigma::Float64=6.0,
    ntof_env::AbstractString="auto",
    ntof_alpha::Float64=0.5,
    ntof_beta::Float64=1/3,
    ntof_min::Int=200,
    ntof_max::Int=2000,
    # caching (Serialization-based)
    disk_cache::Bool=true,
    cache_dir::AbstractString=joinpath(@__DIR__, "..", "scripts", ".toftwin_cache"),
    cache_ver::AbstractString="ctx_v1",
    cache_Cv::Bool=false,
    cache_rmap::Bool=false)

    # --- instrument ---
    out = if isdefined(MantidIDF, :load_mantid_idf_diskcached)
        MantidIDF.load_mantid_idf_diskcached(String(idf_path); rebuild=rebuild_geom)
    else
        @warn "load_mantid_idf_diskcached not found; falling back to load_mantid_idf (slower)."
        MantidIDF.load_mantid_idf(String(idf_path))
    end

    inst = out.inst
    bank0 = (hasproperty(out, :bank) && out.bank !== nothing) ? out.bank : DetectorBank(inst.name, out.pixels)

    # --- grouping/masking ---
    pixels_gm = if isdefined(@__MODULE__, :apply_grouping_masking)
        apply_grouping_masking(bank0.pixels;
            instrument=instr,
            grouping=grouping,
            grouping_file=grouping_file,
            mask_btp=mask_btp,
            mask_mode=mask_mode,
            outdir=joinpath(dirname(String(idf_path)), "out"),
            angle_step=angle_step,
            return_meta=false,
        )
    else
        @warn "apply_grouping_masking not found; proceeding without grouping/masking"
        bank0.pixels
    end
    bank = DetectorBank(inst.name, pixels_gm)

    # --- decimation ---
    pixels = sample_pixels(bank, AngularDecimate(ψstride, ηstride))

    # --- axes ---
    Q_edges = collect(range(kern_Qmin, kern_Qmax; length=nQbins+1))
    ωlo = min(-2.0, kern_wmin)
    ωhi = max(Ei,  kern_wmax)
    ω_edges = collect(range(ωlo, ωhi; length=nωbins+1))

    # --- TOF window ---
    L2min = minimum(inst.L2); L2max = maximum(inst.L2)
    Ef_min, Ef_max = 1.0, Ei
    tmin = tof_from_EiEf(inst.L1, L2min, Ei, Ef_max)
    tmax = tof_from_EiEf(inst.L1, L2max, Ei, Ef_min)

    σt = σt_us * 1e-6

    ntof = let env = lowercase(String(ntof_env))
        n0 = (env in ("auto","suggest","0")) ? 0 : parse(Int, env)
        if n0 <= 0 && lowercase(res_mode) == "cdf" && σt_us > 0
            ntof_s, dt_s, maxd, dωbin = suggest_ntof(inst, pixels, Ei, tmin, tmax, ω_edges, σt;
                α=ntof_alpha, β=ntof_beta, ntof_min=ntof_min, ntof_max=ntof_max)
            @info "Auto-selected TOFTWIN_NTOF=$ntof_s (dt=$(round(1e6*dt_s,digits=2)) µs; max|dω/dt|=$(round(maxd,digits=4)) meV/s; Δωbin=$(round(dωbin,digits=4)) meV)"
            ntof_s
        elseif n0 <= 0
            500
        else
            n0
        end
    end

    tof_edges = collect(range(tmin, tmax; length=ntof+1))

    # --- resolution ---
    # Note: For σt_source="pychop", we build a TOF-dependent σt(t) curve and
    #       wrap it in GaussianTimingCDFResolution(σt_bins; ...).
    res_mode_l = lowercase(res_mode)
    σt_source_l = lowercase(String(σt_source))

    resolution = if res_mode_l == "none" || σt_us <= 0
        NoResolution()
    elseif res_mode_l == "gh"
        GaussianTimingResolution(σt; order=gh_order)
    elseif res_mode_l == "cdf"
        if σt_source_l == "pychop"
            # Build effective σt(t) so that σ_ω(t) ≈ ΔE_PyChop(ω(t))/2.355
            # (We treat the PyChop curve as FWHM by default; the oracle script
            # can be changed to emit σ directly if desired.)
            σt_bins = _pychop_effective_sigma_t_bins(inst, pixels, Ei, tof_edges;
                python=pychop_python,
                script=pychop_script,
                instrument=instr,
                variant=pychop_variant,
                freq_hz=pychop_freq_hz,
                tc_index=pychop_tc_index,
                use_tc_rss=pychop_use_tc_rss,
                delta_td_us=pychop_delta_td_us,
                npts=pychop_npts,
            )
            GaussianTimingCDFResolution(σt_bins; nsigma=nsigma)
        else
            GaussianTimingCDFResolution(σt; nsigma=nsigma)
        end
    else
        throw(ArgumentError("TOFTWIN_RES_MODE must be none|gh|cdf (got '$res_mode')"))
    end

    # --- optional CDF work ---
    cdf_work = if (resolution isa GaussianTimingCDFResolution)
        if isdefined(@__MODULE__, :precompute_tof_smear_work)
            @info "Using in-session precomputed CDF smear work"
            precompute_tof_smear_work(inst, pixels, Ei, tof_edges, resolution)
        else
            @warn "TOFtwin.precompute_tof_smear_work not found; continuing without precomputed work"
            nothing
        end
    else
        nothing
    end

    # --- Cv(flat) ---
    compute_Cv_flat() = predict_pixel_tof(inst;
        pixels=pixels, Ei_meV=Ei, tof_edges_s=tof_edges,
        model=(Q,ω)->1.0, resolution=resolution,
        cdf_work=cdf_work
    )

    Cv = if disk_cache && cache_Cv
        io = IOBuffer()
        serialize(io, cache_ver)
        serialize(io, String(instr))
        serialize(io, String(idf_path))
        st = try stat(String(idf_path)) catch; nothing end
        serialize(io, st === nothing ? nothing : (st.mtime, st.size))
        serialize(io, grouping)
        serialize(io, grouping_file === nothing ? "" : String(grouping_file))
        serialize(io, mask_btp)
        serialize(io, mask_mode)
        serialize(io, Float64(angle_step))
        serialize(io, Int32[p.id for p in pixels])
        serialize(io, Ei)
        serialize(io, tof_edges)
        serialize(io, resolution)
        path = _wf_cache_path(cache_dir, "Cv_flat", take!(io))
        @info "Cv cache: $path"
        _wf_load_or_compute(path, compute_Cv_flat; disk_cache=disk_cache)
    else
        compute_Cv_flat()
    end

    # --- reduce map ---
    compute_rmap() = build_reduce_map_powder(inst, pixels, Ei, tof_edges, Q_edges, ω_edges)

    rmap = if disk_cache && cache_rmap
        io = IOBuffer()
        serialize(io, cache_ver)
        serialize(io, String(instr))
        serialize(io, String(idf_path))
        st = try stat(String(idf_path)) catch; nothing end
        serialize(io, st === nothing ? nothing : (st.mtime, st.size))
        serialize(io, grouping)
        serialize(io, grouping_file === nothing ? "" : String(grouping_file))
        serialize(io, mask_btp)
        serialize(io, mask_mode)
        serialize(io, Float64(angle_step))
        serialize(io, Int32[p.id for p in pixels])
        serialize(io, Ei)
        serialize(io, tof_edges)
        serialize(io, Q_edges)
        serialize(io, ω_edges)
        serialize(io, resolution)
        path = _wf_cache_path(cache_dir, "reduce_map_powder", take!(io))
        @info "Reduce-map cache: $path"
        _wf_load_or_compute(path, compute_rmap; disk_cache=disk_cache)
    else
        compute_rmap()
    end

    return PowderCtx{typeof(resolution), typeof(cdf_work)}(instr, String(idf_path), inst, pixels, Ei, tof_edges,
                                                          Q_edges, ω_edges, resolution, cdf_work, Cv, rmap)
end

"""
    eval_powder(ctx, model; eps=1e-12, do_hist=true)

Evaluate one scattering model against an existing powder ctx.

Returns `(Cs, Hraw, Hmean, Hwt)` when `do_hist=true`.
"""
function eval_powder(ctx::PowderCtx, model; eps=1e-12, do_hist::Bool=true)
    Cs = predict_pixel_tof(ctx.inst;
        pixels=ctx.pixels, Ei_meV=ctx.Ei, tof_edges_s=ctx.tof_edges,
        model=model, resolution=ctx.resolution,
        cdf_work=ctx.cdf_work
    )

    if !do_hist
        return (Cs=Cs, Hraw=nothing, Hmean=nothing, Hwt=nothing)
    end

    Hraw = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hsum = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hwt  = Hist2D(ctx.Q_edges, ctx.ω_edges)

    reduce_three!(Hraw, Hsum, Hwt, ctx.pixels, ctx.rmap, Cs, ctx.Cv; eps=eps)

    Hmean = Hist2D(ctx.Q_edges, ctx.ω_edges)
    Hmean.counts .= Hsum.counts ./ (Hwt.counts .+ eps)

    return (Cs=Cs, Hraw=Hraw, Hmean=Hmean, Hwt=Hwt)
end
